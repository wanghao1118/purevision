from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window_slice(values: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    clipped = np.clip(values.astype(np.float32), lower, lower + width)
    return np.round((clipped - lower) * 255.0 / width).astype(np.uint8)


def build(source: Path, output: Path, center: float, width: float) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}")
    if width <= 0:
        raise ValueError("窗宽必须为正数")
    import pylidc as pl

    frame = pd.read_csv(source / "metadata.csv")
    required = {
        "sample_id", "patient_id", "scan_id", "max_slice_index", "image_path",
        "nodule_mask_path", "roi_row_start", "roi_row_stop", "roi_col_start",
        "roi_col_stop", "texture_native_score", "sphericity_native_score",
        "margin_native_score", "lobulation_native_score", "spiculation_native_score",
        "calcification_native_score",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"源元数据缺少字段：{sorted(missing)}")
    if frame.sample_id.duplicated().any():
        raise ValueError("源元数据存在重复 sample_id")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "nodule_masks").mkdir(exist_ok=True)
    rows = []
    crop_checks = 0
    for scan_id, group in frame.groupby("scan_id", sort=True):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).first()
        if scan is None:
            raise ValueError(f"找不到 scan_id={scan_id}")
        dicoms = scan.load_all_dicom_images(verbose=False)
        cache = {}
        for row in group.to_dict("records"):
            index = int(row["max_slice_index"])
            if not 0 <= index < len(dicoms):
                raise ValueError(f"切片索引越界：{row['sample_id']}")
            if index not in cache:
                dicom = dicoms[index]
                hu = dicom.pixel_array.astype(np.float32) * float(dicom.RescaleSlope)
                hu += float(dicom.RescaleIntercept)
                cache[index] = window_slice(hu, center, width)
            image = cache[index]
            height, width_px = image.shape
            r0, r1 = int(row["roi_row_start"]), int(row["roi_row_stop"])
            c0, c1 = int(row["roi_col_start"]), int(row["roi_col_stop"])
            if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width_px):
                raise ValueError(f"源裁剪坐标越界：{row['sample_id']}")
            old_image = np.asarray(Image.open(source / row["image_path"]).convert("L"))
            old_mask = np.asarray(Image.open(source / row["nodule_mask_path"]).convert("L"))
            if old_image.shape != (r1 - r0, c1 - c0) or old_mask.shape != old_image.shape:
                raise ValueError(f"源裁剪图像和掩码几何不一致：{row['sample_id']}")
            if not np.array_equal(image[r0:r1, c0:c1], old_image):
                raise ValueError(f"DICOM 重建与源裁剪像素不一致：{row['sample_id']}")
            crop_checks += 1
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[r0:r1, c0:c1] = old_mask
            sample_id = str(row["sample_id"])
            image_rel = Path("images") / f"{sample_id}.png"
            mask_rel = Path("nodule_masks") / f"{sample_id}.png"
            Image.fromarray(np.repeat(image[:, :, None], 3, axis=2), mode="RGB").save(
                output / image_rel, optimize=True
            )
            Image.fromarray(mask, mode="L").save(output / mask_rel, optimize=True)
            row.update({
                "image_path": image_rel.as_posix(),
                "nodule_mask_path": mask_rel.as_posix(),
                "lung_mask_path": "",
                "source_lung_roi_row_start": r0,
                "source_lung_roi_row_stop": r1,
                "source_lung_roi_col_start": c0,
                "source_lung_roi_col_stop": c1,
                "roi_row_start": 0,
                "roi_row_stop": height,
                "roi_col_start": 0,
                "roi_col_stop": width_px,
                "source_image_height": height,
                "source_image_width": width_px,
                "preprocessing_mode": "full_dicom_no_lung_crop",
            })
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "metadata.csv", index=False)
    for name in ("splits.json", "attribute_space.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)
    return {
        "数据集ID": "LIDC", "发布版": "lidc_full_fov_construction_v1",
        "来源根目录": str(source.resolve()),
        "标签来源": "LIDC-IDRI 医师 XML 评分与轮廓；本阶段不重新标注",
        "划分清单": str((output / "splits.json").resolve()),
        "划分清单SHA256": sha256(output / "splits.json") if (output / "splits.json").exists() else None,
        "阶段": "全幅 DICOM 重建；没有模型推理、mask 条件池化、监督质心评估或零样本分类",
        "样本数": len(rows), "源裁剪逐像素核验数": crop_checks,
        "预处理": "full_dicom_no_lung_crop",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    args = parser.parse_args()
    audit = build(args.source, args.output, args.window_center, args.window_width)
    (args.output / "构造记录_ZH.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
