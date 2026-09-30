from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from purevision.protocol import sha256_file, validate_dataset_protocol


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "examples" / "lidc_case_0079" / "case_zh.json"
INSTRUCTION = (
    "Identify the pulmonary nodule location and describe density, sphericity, "
    "margin, lobulation, spiculation, calcification, and size."
)


def verify_case(case: dict, config: dict, case_dir: Path) -> dict:
    dataset = validate_dataset_protocol(config)
    for key, expected in (
        ("dataset_id", dataset["dataset_id"]),
        ("dataset_release", dataset["release"]),
        ("source_root", dataset["source_root"]),
        ("split_manifest", dataset["split_manifest"]),
        ("split_manifest_sha256", dataset["split_manifest_sha256"]),
    ):
        if case[key] != expected:
            raise ValueError(f"病例与配置的 {key} 不一致")
    if case["sample_split"] != "test":
        raise ValueError("该真实病例必须属于固定 test 划分")
    image_path = case_dir / case["image"]
    mask_path = case_dir / case["nodule_mask"]
    if sha256_file(image_path) != case["image_sha256"]:
        raise ValueError("病例图像 SHA-256 不一致")
    if sha256_file(mask_path) != case["nodule_mask_sha256"]:
        raise ValueError("病例病灶 mask SHA-256 不一致")
    with Image.open(image_path) as opened:
        image_size = opened.size
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L")) > 0
    height, width = mask.shape
    if image_size != (width, height) or not mask.any():
        raise ValueError("图像与病灶 mask 尺寸不一致，或 mask 为空")
    rows, columns = np.nonzero(mask)
    cells = (rows * 4 // height) * 4 + (columns * 4 // width)
    counts = np.bincount(cells, minlength=16)
    index = int(counts.argmax())
    grid_cell = f"r{index // 4 + 1}c{index % 4 + 1}"
    if grid_cell != case["reference"]["grid_cell_4x4"]:
        raise ValueError(f"mask 阳性像素最多的 4x4 位置为 {grid_cell}，与病例标签不一致")
    occupied_cells = [
        {"grid_cell": f"r{cell // 4 + 1}c{cell % 4 + 1}", "pixels": int(count)}
        for cell, count in enumerate(counts) if count
    ]
    observed_counts = {item["grid_cell"]: item["pixels"] for item in occupied_cells}
    if observed_counts != case["reference"]["grid_cell_4x4_counts"]:
        raise ValueError("mask 网格像素计数与冻结病例标签不一致")
    if (len(occupied_cells) == 1) != case["reference"]["single_grid_cell"]:
        raise ValueError("mask 单格筛选标志与冻结病例标签不一致")
    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "grid_cell": grid_cell,
        "lesion_pixels": int(mask.sum()),
        "occupied_cells": occupied_cells,
        "single_cell": len(occupied_cells) == 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 LIDC-IDRI 真实病例的 MedGemma 1.5 配对测试")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "07_medgemma15_reproduction_smoke.yaml")
    parser.add_argument("--case", type=Path, default=CASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "lidc_case_0079")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fixture = verify_case(case, config, args.case.parent)
    print(f"病例核验通过：{case['sample_id']}，test，最多阳性像素位于 {fixture['grid_cell']}，单格={fixture['single_cell']}")
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    common = [
        "--config", str(args.config),
        "--image", str(fixture["image_path"]),
        "--instruction", INSTRUCTION,
        "--max-new-tokens", str(args.max_new_tokens),
        "--sample-id", case["sample_id"],
        "--split", case["sample_split"],
    ]
    native_path = args.output_dir / "native_result_zh.json"
    pure_path = args.output_dir / "purevision_result_zh.json"
    embedding_path = args.output_dir / "raw_patch_embeddings.pt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_native_baseline.py"), *common, "--output", str(native_path)],
        cwd=ROOT, env=environment, check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_inference.py"), *common,
         "--embedding-output", str(embedding_path), "--output", str(pure_path)],
        cwd=ROOT, env=environment, check=True,
    )
    pure = json.loads(pure_path.read_text(encoding="utf-8"))
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if len(pure["selected_patch_indices"]) != 8:
        raise ValueError("PureVision 未返回固定的 8 个病灶 patch")
    if native["image_sha256"] != case["image_sha256"] or pure["image_sha256"] != case["image_sha256"]:
        raise ValueError("生成结果未绑定本病例图像")
    summary = {
        "schema_version": 1,
        "记录类型": "LIDC-IDRI 真实病例配对测试结果",
        "dataset_id": case["dataset_id"],
        "dataset_release": case["dataset_release"],
        "source_root": case["source_root"],
        "label_provenance_zh": case["label_provenance_zh"],
        "split_manifest": case["split_manifest"],
        "split_manifest_sha256": case["split_manifest_sha256"],
        "sample_id": case["sample_id"],
        "sample_split": case["sample_split"],
        "病例标签中英对照": case["reference_display_zh_en"],
        "参考4x4位置": fixture["grid_cell"],
        "病灶mask阳性像素": fixture["lesion_pixels"],
        "病灶mask覆盖网格": fixture["occupied_cells"],
        "符合论文单格筛选": fixture["single_cell"],
        "流程说明": "同一真实病例先以原始 MedGemma 1.5 标准未修改推理生成，再以冻结的双视觉塔、共享对齐器、5x5/Top-8 选择、软融合及冻结 decoder 生成；不使用推理期 mask、监督质心评估、zero-shot subtype 或 t-SNE。",
        "原生结果SHA256": sha256_file(native_path),
        "PureVision结果SHA256": sha256_file(pure_path),
        "原始双塔特征SHA256": sha256_file(embedding_path),
        "说明": "病例级功能核验，不作为论文 VQA/RRG 总体指标。",
    }
    summary_path = args.output_dir / "SUMMARY_ZH.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"配对测试完成：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
