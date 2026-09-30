import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label

from run_knee_coronal_2d import (
    CHECKPOINT_SHA, COLORS, TISSUES, checksum, crop, from_model, load_model,
    read_image, solid, to_model, write_json,
)

CODE_FILES = ("build_knee_strict2d_dataset.py", "run_knee_coronal_2d.py")
SPLITS = ("train", "val", "test")
MODEL_REVISION = "ff5e0b5d1bf8b850ee6258210e05418889d7133f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def asset(root, relative):
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()), "Unsafe asset path")
    return path


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(array).save(temporary, format="PNG")
    temporary.replace(path)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def pack(masks):
    require(set(masks) == set(TISSUES), "Wrong anatomy channels")
    shape = masks["femur"].shape
    packed = np.zeros(shape, np.uint8)
    for index, tissue in enumerate(TISSUES):
        require(masks[tissue].dtype == bool and masks[tissue].shape == shape, "Invalid binary mask")
        packed |= masks[tissue].astype(np.uint8) << index
    return packed


def choose_masks(manual, predicted):
    require(set(manual).issubset(TISSUES), "Unknown manual channel")
    masks, sources = {}, {}
    if predicted is not None:
        require(predicted.shape == (384, 160) and predicted.dtype == np.uint8, "Invalid native prediction")
        require(np.isin(predicted, range(10)).all(), "Unknown model class")
    for tissue, ids in TISSUES.items():
        if tissue in manual:
            mask = manual[tissue]
            require(mask.dtype == bool and mask.shape == (384, 160), "Invalid manual mask")
            masks[tissue], sources[tissue] = mask.copy(), "manual"
        else:
            require(predicted is not None, "Missing manual channel requires a model prediction")
            masks[tissue], sources[tissue] = np.isin(predicted, ids), "model_prediction"
    return masks, sources


def components(mask):
    return int(label(mask, structure=np.ones((3, 3), np.uint8))[1])


def roi_quality(full, cropped, spacing):
    pixels = int(cropped.sum())
    count = components(cropped)
    reasons = []
    if pixels == 0:
        reasons.append("empty_medial_meniscus")
    if pixels and count != 1:
        reasons.append("disconnected_medial_meniscus")
    if int(full.sum()) != pixels:
        reasons.append("medial_meniscus_truncated_by_fixed_crop")
    area = pixels * float(spacing[0]) * float(spacing[1])
    if 0 < area < 3:
        reasons.append("medial_meniscus_area_below_original_3mm2_gate")
    return {"passes_original_single_slice_geometry_gate": not reasons, "reasons": reasons,
            "roi_pixels": pixels, "roi_area_mm2": area, "roi_components_8": count,
            "full_roi_pixels": int(full.sum()), "full_roi_components_8": components(full),
            "accuracy_expert_verified": False}


def load_inputs(source, row):
    for name in ("image", "native", "manual"):
        if row.get(name + "_path"):
            require(checksum(asset(source, row[name + "_path"])) == row[name + "_sha256"], "Input hash mismatch: " + name)
    gray = read_image(asset(source, row["image_path"]))
    archive = load_npz(asset(source, row["native_path"]))
    require(set(archive) == {"intensity"}, "Not a single native intensity plane")
    native = archive["intensity"]
    require(gray.shape == native.shape == (384, 160), "Wrong input shape")
    require(native.ndim == 2 and np.isfinite(native).all() and float(native.std()) > 1e-8, "Invalid single slice")
    box = row["geometry"]["crop_box_native_xyxy"]
    require(len(box) == 4 and all(isinstance(value, int) for value in box), "Noninteger crop")
    require(crop(gray, box).shape == (128, 128), "Crop dimensions changed")
    manual = load_npz(asset(source, row["manual_path"])) if row.get("manual_path") else {}
    return gray, native, manual


def export_sample(root, source, row, gray, native, masks, sources, inference, plan_sha):
    sid = row["sample_id"]
    require(sid and Path(sid).name == sid and "/" not in sid and "\\" not in sid, "Unsafe sample ID")
    base = Path("cases") / sid
    box = row["geometry"]["crop_box_native_xyxy"]
    views, hashes = {}, {}
    native_relative = (base / "native_intensity.npz").as_posix()
    save_npz(root / native_relative, full=native, crop=crop(native, box))
    hashes[native_relative] = checksum(root / native_relative)
    for view in ("full", "crop"):
        image = gray if view == "full" else crop(gray, box)
        selected = masks if view == "full" else {t: crop(m, box) for t, m in masks.items()}
        fields = {"lesion_mask_path": None, "native_intensity_path": native_relative, "native_intensity_key": view}
        arrays = {"image": image, "anatomy_mask": pack(selected),
                  "lesion_roi_proxy": selected["medial_meniscus"].astype(np.uint8),
                  "anatomy_color": solid(selected)}
        if view == "crop":
            overlay = np.repeat(image[..., None], 3, axis=2)
            color = arrays["anatomy_color"]
            foreground = np.any(color, axis=2)
            overlay[foreground] = np.rint(0.5 * overlay[foreground] + 0.5 * color[foreground]).astype(np.uint8)
            arrays["anatomy_overlay"] = overlay
        for name, array in arrays.items():
            relative = (base / f"{name}_{view}.png").as_posix()
            if name == "image" and view == "full":
                target = root / relative
                shutil.copyfile(asset(source, row["image_path"]), target.with_suffix(".png.tmp"))
                target.with_suffix(".png.tmp").replace(target)
            else:
                save_png(root / relative, array)
            hashes[relative], fields[name + "_path"] = checksum(root / relative), relative
        relative = (base / f"anatomy_channels_{view}.npz").as_posix()
        save_npz(root / relative, **selected)
        hashes[relative], fields["anatomy_channels_path"] = checksum(root / relative), relative
        views[view] = fields
    annotations = {}
    for tissue, mode in sources.items():
        original = row.get("manual_sources", {}).get(tissue.replace("_", " "), {})
        annotations[tissue] = {"source_type": mode, "manual_available": mode == "manual",
                               "quality_status": "source_manual_not_re_reviewed" if mode == "manual" else "unreviewed_pseudo_label",
                               "manual_reference": original.get("manual_reference") if mode == "manual" else None,
                               "provider": "3DReasonKnee manual annotation" if mode == "manual" else "aagatti/dosma_bones",
                               "source_label_ids": [1] if mode == "manual" else TISSUES[tissue]}
    qc = roi_quality(masks["medial_meniscus"], crop(masks["medial_meniscus"], box), row["geometry"]["pixel_spacing_mm_lr_si"])
    geometry = dict(row["geometry"])
    for name in ("roi_pixels", "roi_components_8"):
        geometry["previous_" + name] = geometry.get(name)
        geometry[name] = qc[name]
    record = {key: row[key] for key in ("sample_id", "patient_id", "split", "dimensions", "label_scope", "intensity_window")}
    record.update(views=views, **views["crop"], geometry=geometry, geometry_quality=qc,
                  anatomy_mask_encoding="independent_channel_bitset", annotation_sources=annotations,
                  spatial_resampling=False, lesion_mask_status="not_available_in_source",
                  lesion_roi_proxy_semantics="entire_medial_meniscus_not_extruded_portion",
                  mask_quality_status="manual_first_with_unreviewed_model_fallback",
                  slice_specific_expert_validation=False, model_inference=inference, sha256=hashes,
                  source_build_plan_sha256=plan_sha,
                  metadata_path=(base / "metadata.json").as_posix(),
                  source_input={key: row.get(key) for key in ("image_path", "image_sha256", "native_path", "native_sha256", "manual_path", "manual_sha256")})
    verify_record(root, record, row, source)
    write_json(root / record["metadata_path"], record)
    return record


def verify_record(root, record, row, source):
    require(record["sample_id"] == row["sample_id"] and record["patient_id"] == row["patient_id"], "Sample identity changed")
    require(record["dimensions"] == row["dimensions"] and record["split"] == row["split"], "Labels or split changed")
    require(record["geometry"]["crop_box_native_xyxy"] == row["geometry"]["crop_box_native_xyxy"], "Crop moved")
    require(record["lesion_mask_path"] is None and not record["spatial_resampling"], "Incorrect lesion/resampling claim")
    require(record["anatomy_mask_encoding"] == "independent_channel_bitset", "Wrong mask encoding")
    require(set(record["annotation_sources"]) == set(TISSUES), "Missing source provenance")
    for relative, digest in record["sha256"].items():
        require(checksum(asset(root, relative)) == digest, "Output hash mismatch: " + relative)
    require(record["sha256"][record["views"]["full"]["image_path"]] == row["image_sha256"], "Full image bytes changed")
    box = row["geometry"]["crop_box_native_xyxy"]
    loaded = {}
    native = load_npz(asset(root, record["views"]["full"]["native_intensity_path"]))
    require(set(native) == {"full", "crop"}, "Unexpected intensity arrays")
    for view, shape in (("full", (384, 160)), ("crop", (128, 128))):
        fields = record["views"][view]
        image = read_image(asset(root, fields["image_path"]))
        packed = read_image(asset(root, fields["anatomy_mask_path"]))
        roi = read_image(asset(root, fields["lesion_roi_proxy_path"]))
        masks = load_npz(asset(root, fields["anatomy_channels_path"]))
        require(image.shape == packed.shape == roi.shape == native[view].shape == shape, "Asset shape mismatch")
        require(np.array_equal(packed, pack(masks)), "Bitset and independent channels disagree")
        require(np.array_equal(roi, masks["medial_meniscus"]), "Proxy ROI differs from medial meniscus")
        require(np.array_equal(read_image(asset(root, fields["anatomy_color_path"])), solid(masks)), "Color mask differs")
        loaded[view] = {"image": image, "packed": packed, "roi": roi, "native": native[view], "masks": masks}
    for name in ("image", "packed", "roi", "native"):
        require(np.array_equal(crop(loaded["full"][name], box), loaded["crop"][name]), "Image/mask crop mismatch: " + name)
    for tissue in TISSUES:
        require(np.array_equal(crop(loaded["full"]["masks"][tissue], box), loaded["crop"]["masks"][tissue]), "Channel crop mismatch")
    manual = {}
    if row.get("manual_path"):
        require(checksum(asset(source, row["manual_path"])) == row["manual_sha256"], "Manual input changed")
        manual = load_npz(asset(source, row["manual_path"]))
    for tissue in TISSUES:
        expected = "manual" if tissue in manual else "model_prediction"
        require(record["annotation_sources"][tissue]["source_type"] == expected, "Wrong manual priority")
        if tissue in manual:
            require(np.array_equal(manual[tissue], loaded["full"]["masks"][tissue]), "Human annotation altered")
    qc = roi_quality(loaded["full"]["roi"].astype(bool), loaded["crop"]["roi"].astype(bool), row["geometry"]["pixel_spacing_mm_lr_si"])
    require(record["geometry_quality"] == qc, "Stale geometry flags")
    require(record["model_inference"]["performed"] == (len(manual) != len(TISSUES)), "Incorrect inference receipt")
    return True


def prepare(args):
    root, source = args.output.resolve(), args.input.resolve()
    require(not root.exists(), "Output exists; resume with build")
    require(not root.is_relative_to(source) and not source.is_relative_to(root), "Separate output directory required")
    require(checksum(args.model) == CHECKPOINT_SHA, "Unapproved checkpoint")
    rows = read_rows(source / "samples_native_all.jsonl")
    require(rows and len(rows) == len({r["sample_id"] for r in rows}), "Empty or duplicate plan")
    groups = {s: {r["patient_id"] for r in rows if r["split"] == s} for s in SPLITS}
    require(all(r["split"] in SPLITS for r in rows), "Invalid split")
    require(not any(groups[a] & groups[b] for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]), "Patient leakage")
    definitions = read_json(source / "dimensions.json")
    choices = {g["group"]: {c["value"]: c["label"] for c in g["choices"]} for g in definitions["attributes"]}
    require(all(set(r["dimensions"]) == set(choices) for r in rows), "Wrong dimensions")
    for row in rows:
        for group, value in row["dimensions"].items():
            require(choices[group].get(value["value"]) == value["label"], "Unknown original class")
    pilot = read_json(args.pilot / "evaluation.json")
    require(pilot["backend"] == "tensorflow" and pilot["checkpoint_sha256"] == CHECKPOINT_SHA,
            "Pilot is not the approved native TensorFlow model")
    require(pilot["input_is_strictly_2d"] and not pilot["spatial_resize"], "Pilot contract differs")
    code = root / "provenance/code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copyfile(Path(__file__).parent / name, code / name)
    for name in ("dimensions.json", "anatomy_labels.json"):
        shutil.copyfile(source / name, root / name)
    anatomy = read_json(root / "anatomy_labels.json")
    require([c["npz_key"] for c in anatomy["channels"]] == list(TISSUES), "Inherited bit order changed")
    for name in ("evaluation.json", "audit.json"):
        shutil.copyfile(args.pilot / name, root / "provenance" / ("pilot_" + name))
    shutil.copyfile(source / "samples_native_all.jsonl", root / "provenance/build_plan.jsonl")
    config = {"schema_version": 4, "source": str(source), "model": str(args.model.resolve()),
              "samples_expected": len(rows), "plan_sha256": checksum(root / "provenance/build_plan.jsonl"),
              "model_checkpoint_sha256": CHECKPOINT_SHA, "model_revision": MODEL_REVISION,
              "inference": "single native 2D slice; full segmentation then exact crop; no resize",
              "manual_policy": "per-structure priority including empty/overlapping annotations; never repair",
              "user_approved_full_run": True, "expert_quality_approved": False,
              "model_labels_are_pseudo_labels": True, "lesion_ground_truth_available": False,
              "code_sha256": {name: checksum(code / name) for name in CODE_FILES},
              "input_metadata_sha256": {name: checksum(root / name) for name in ("dimensions.json", "anatomy_labels.json")},
              "prepared_unix": time.time()}
    write_json(root / "build_config.json", config)
    write_json(root / "build_status.json", {"state": "prepared", "completed": 0, "expected": len(rows)})
    print(json.dumps({"root": str(root), "samples": len(rows)}), flush=True)


def frozen(root):
    config = read_json(root / "build_config.json")
    require(checksum(root / "provenance/build_plan.jsonl") == config["plan_sha256"], "Plan modified")
    for name, digest in config["code_sha256"].items():
        require(checksum(root / "provenance/code" / name) == digest, "Frozen code modified")
        require(checksum(Path(__file__).parent / name) == digest, "Run with frozen production code")
    for name, digest in config["input_metadata_sha256"].items():
        require(checksum(root / name) == digest, "Frozen labels modified")
    return config


README = """# 3DReasonKnee 单切片数据构造说明

数据集 ID 为 3DReasonKnee，发布版为 medial_meniscus_strict2d_crop128_v4。
来源根目录、患者级 train/val/test 清单和标签来源见 build_config.json、
manifest/{train,val,test}.jsonl 与各病例 metadata.json。发布后应计算并保存
manifest/all.jsonl 的 SHA-256；不得仅凭目录名推断来源身份。

每例包含一张原生近冠状位 MRI 切片、无插值整数裁剪图、原始强度数组和八个
可重叠的二值解剖通道。anatomy_mask_*.png 是 bitset，不是互斥类别 ID；
按 anatomy_labels.json 的 bit_index 解码。人工标签优先，缺失结构才由冻结
二维模型预测；模型伪标签未经专家逐像素复核。

内侧半月板 (medial meniscus) 的 lesion_roi_proxy_* 表示整块半月板，
不是外突病灶 (extrusion lesion) 真值；lesion_mask_path 为 null。
内侧半月板外突等级 (medial meniscus medial extrusion grade) 和形态
(meniscal morphology) 是继承的检查级区域 MOAKS 标签，不是逐切片复核诊断。
评测者可将整块内侧半月板代理 ROI 约定为定位参考真值，以阳性像素
最多的 4×4 网格作为答案；不得称其为专家逐像素外突病灶标注。

full 与 crop 图像未做空间重采样。构造时先在原生切片上得到掩码，再使用相同
裁剪坐标。manifest/all.jsonl 保留所有例，single_connected/ 仅是几何筛选，
不等于分割正确性。dataset.complete 只证明技术完整性，不证明临床准确性。

本构造阶段不执行 MedGemma 原始预训练权重标准未修改推理、mask 条件池化、
监督质心评估、零样本分类或 t-SNE；若后续绘制 t-SNE，簇间空隙不得当作
原始嵌入空间的定量距离。
"""


def finalize(root, source, config, plan):
    records = []
    for index, row in enumerate(plan):
        record = read_json(root / "cases" / row["sample_id"] / "metadata.json")
        require(record["source_build_plan_sha256"] == config["plan_sha256"], "Record has different plan")
        verify_record(root, record, row, source)
        records.append(record)
        if (index + 1) % 1000 == 0:
            write_json(root / "build_status.json", {"state": "verifying", "completed": len(plan), "verified": index+1, "expected": len(plan)})
            print(json.dumps({"verified": index+1}), flush=True)
    require(len(records) == config["samples_expected"], "Incomplete dataset")
    summary = {"samples": len(records), "patients": len({r["patient_id"] for r in records}),
               "manual_cases": sum(any(a["source_type"] == "manual" for a in r["annotation_sources"].values()) for r in records),
               "model_inference_cases": sum(r["model_inference"]["performed"] for r in records),
               "structure_sources": {t: dict(Counter(r["annotation_sources"][t]["source_type"] for r in records)) for t in TISSUES},
               "geometry_review_reasons": dict(Counter(x for r in records for x in r["geometry_quality"]["reasons"])),
               "single_connected": sum(r["geometry_quality"]["passes_original_single_slice_geometry_gate"] for r in records),
               "positive_extrusion_all": sum(r["dimensions"]["medial_meniscus_medial_extrusion"]["value"] > 0 for r in records),
               "full_shape_hw": [384, 160], "crop_shape_hw": [128, 128], "spatial_resampling": False,
               "lesion_ground_truth_available": False, "model_masks_expert_reviewed": False, "splits": {}}
    write_rows(root / "manifest/all.jsonl", records)
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        ready = [r for r in selected if r["geometry_quality"]["passes_original_single_slice_geometry_gate"]]
        review = [r for r in selected if not r["geometry_quality"]["passes_original_single_slice_geometry_gate"]]
        positive = [r for r in ready if r["dimensions"]["medial_meniscus_medial_extrusion"]["value"] > 0]
        for directory, group in (("manifest", selected), ("single_connected", ready), ("needs_geometry_review", review), ("positive_only", positive)):
            write_rows(root / directory / f"{split}.jsonl", group)
        summary["splits"][split] = {"samples": len(selected), "single_connected": len(ready), "needs_geometry_review": len(review),
                                    "positive_only": len(positive),
                                    "class_counts": {key: dict(Counter(str(r["dimensions"][key]["value"]) for r in selected)) for key in records[0]["dimensions"]}}
    write_json(root / "dataset_summary.json", summary)
    write_json(root / "validation.json", {"records_verified": len(records), "all_output_hashes_checked": True,
               "full_images_byte_identical_to_source": True, "image_mask_crop_exact": True, "independent_channels_equal_bitsets": True,
               "manual_masks_unchanged": True, "original_labels_and_splits_preserved": True,
               "no_3d_inference": True, "expert_accuracy_validation": False})
    (root / "README.md").write_text(README)
    frozen(root)
    paths = [p for p in root.rglob("*") if p.is_file() and p.name not in ("build.lock", "build_status.json", "dataset.complete", "storage.json")]
    write_json(root / "storage.json", {"files": len(paths), "bytes": sum(p.stat().st_size for p in paths),
                                       "excludes": ["build.lock", "build_status.json", "dataset.complete", "storage.json"]})
    metadata = [p for p in paths if p.relative_to(root).parts[0] != "cases"]
    write_json(root / "dataset.complete", {"status": "complete", "samples": len(records),
               "meaning": "technical integrity complete; model masks remain unreviewed pseudo-labels",
               "metadata_sha256": {p.relative_to(root).as_posix(): checksum(p) for p in metadata + [root / "storage.json"]}})
    write_json(root / "build_status.json", {"state": "complete", "completed": len(records), "expected": len(records)})
    print(json.dumps(summary), flush=True)
    return summary


def build(args):
    import fcntl
    root = args.output.resolve()
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "dataset.complete").exists(), "Dataset already complete; use verify")
        config = frozen(root)
        source = args.cache_input.resolve() if args.cache_input else Path(config["source"])
        require(checksum(source / "samples_native_all.jsonl") == config["plan_sha256"], "Cache has a different plan")
        plan = read_rows(root / "provenance/build_plan.jsonl")
        model, completed, new_count = None, 0, 0
        started = time.monotonic()
        try:
            for row in sorted(plan, key=lambda r: not bool(r.get("manual_path"))):
                existing = root / "cases" / row["sample_id"] / "metadata.json"
                if existing.exists():
                    record = read_json(existing)
                    require(record["source_build_plan_sha256"] == config["plan_sha256"], "Resume plan changed")
                    verify_record(root, record, row, source)
                else:
                    if args.limit is not None and new_count >= args.limit:
                        break
                    gray, native, manual = load_inputs(source, row)
                    require(set(manual).issubset(TISSUES), "Unknown manual channels")
                    predicted = None
                    inference = {"performed": False, "input": "single_native_2d_slice_only", "backend": "tensorflow",
                                 "checkpoint_sha256": CHECKPOINT_SHA, "revision": MODEL_REVISION, "seconds": 0.0,
                                 "normalization": "single-slice mean/std", "spatial_resize": False,
                                 "model_input_padding_si": [64, 64], "manual_used_as_prompt": False}
                    if len(manual) < len(TISSUES):
                        if model is None:
                            model = load_model(Path(config["model"]), backend="tensorflow")
                        tic = time.monotonic()
                        probabilities = model(to_model(native), training=False).numpy()
                        require(probabilities.shape == (1, 10, 160, 512) and np.isfinite(probabilities).all(), "Invalid model output")
                        predicted = from_model(probabilities.argmax(axis=1)[0])
                        inference.update(performed=True, seconds=time.monotonic() - tic)
                    masks, sources = choose_masks(manual, predicted)
                    record = export_sample(root, source, row, gray, native, masks, sources, inference, config["plan_sha256"])
                    new_count += 1
                completed += 1
                if completed == 1 or completed % 25 == 0:
                    elapsed = time.monotonic() - started
                    status = {"state": "building", "completed": completed, "expected": len(plan), "new_this_run": new_count,
                              "elapsed_seconds_this_run": elapsed, "latest_sample": row["sample_id"],
                              "estimated_remaining_seconds": (len(plan)-completed) * elapsed / new_count if new_count else None}
                    write_json(root / "build_status.json", status)
                    print(json.dumps(status), flush=True)
            if completed == len(plan):
                write_json(root / "build_status.json", {"state": "verifying", "completed": completed, "expected": len(plan)})
                finalize(root, source, config, plan)
            else:
                write_json(root / "build_status.json", {"state": "paused_after_limit", "completed": completed, "expected": len(plan)})
                print(json.dumps({"paused_after_limit": completed}), flush=True)
        except BaseException as error:
            write_json(root / "build_status.json", {"state": "failed", "completed": completed, "expected": len(plan),
                       "error": str(error), "error_type": type(error).__name__})
            raise


def verify(args):
    root = args.output.resolve()
    config = frozen(root)
    marker = read_json(root / "dataset.complete")
    require(marker["status"] == "complete" and marker["samples"] == config["samples_expected"], "Invalid completion marker")
    for relative, digest in marker["metadata_sha256"].items():
        require(checksum(asset(root, relative)) == digest, "Published metadata changed: " + relative)
    source = args.cache_input.resolve() if args.cache_input else Path(config["source"])
    plan = read_rows(root / "provenance/build_plan.jsonl")
    records = read_rows(root / "manifest/all.jsonl")
    require(len(records) == len(plan), "Wrong manifest size")
    for index, (row, record) in enumerate(zip(plan, records)):
        require(record == read_json(asset(root, record["metadata_path"])), "Case and manifest disagree")
        require(record["source_build_plan_sha256"] == config["plan_sha256"], "Published plan changed")
        verify_record(root, record, row, source)
        if (index + 1) % 1000 == 0:
            print(json.dumps({"verified_published_records": index + 1}), flush=True)
    print(json.dumps({"verified_published_records": len(records), "status": "passed"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    for name in ("build", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--cache-input", type=Path)
        if name == "build":
            p.add_argument("--limit", type=int)
    args = parser.parse_args()
    {"prepare": prepare, "build": build, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
