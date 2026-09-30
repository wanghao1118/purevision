from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

SPLITS = ("train", "val", "test")
GROUPS = ("medial_meniscus_body_morphology", "medial_meniscus_medial_extrusion")
STRUCTURAL_IDS = (40, 41, 44, *range(47, 65))
SOURCE_CARD = "https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    header = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def read_png(path):
    with Image.open(path) as image:
        return np.array(image)


def connected(binary):
    return int(ndimage.label(binary, structure=np.ones((3, 3), bool))[1])


def orientation(image):
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    return transform, affine


def canonical_array(array, transform):
    return nib.orientations.apply_orientation(array, transform)


def plane(array, index):
    return np.ascontiguousarray(array[:, index, :].T[::-1, ::-1])


def check_grids(image, mask):
    require(image.shape == mask.shape and len(image.shape) == 3, "Source image-mask shape mismatch")
    image_transform, image_affine = orientation(image)
    mask_transform, mask_affine = orientation(mask)
    require(np.array_equal(image_transform, mask_transform), "Source image-mask axis mismatch")
    corners = np.array([(*xyz, 1.0) for xyz in itertools.product(*[(0, n - 1) for n in image.shape])]).T
    maximum_error = float(np.max(np.linalg.norm(((image.affine - mask.affine) @ corners)[:3], axis=0)))
    require(maximum_error < 0.001, f"Source grid error exceeds 0.001 mm: {maximum_error}")
    return image_transform, image_affine, mask_affine, maximum_error


def select_native(mask, spacing):
    occupied = np.flatnonzero(np.any(mask == 62, axis=(0, 2)))
    if len(occupied) == 0:
        return {"accepted": False, "reason": "missing_native_central_tibia"}
    center = float(occupied[0] + occupied[-1]) / 2
    candidates = [int(i) for i in occupied if abs(i - center) * spacing[1] <= 3.0 + 1e-8]
    inspected, eligible = [], []
    for index in candidates:
        roi = plane(mask, index) == 44
        count = connected(roi)
        area = float(roi.sum()) * spacing[0] * spacing[2]
        record = {"index": index, "components_8": count, "pixels": int(roi.sum()),
                  "area_mm2": area, "distance_from_center_mm": abs(index - center) * spacing[1]}
        inspected.append(record)
        if count == 1 and area >= 3.0:
            eligible.append(record)
    result = {"accepted": bool(eligible), "reason": "accepted" if eligible else "no_eligible_native_slice",
              "anatomical_center_index": center, "central_tibia_bounds": [int(occupied[0]), int(occupied[-1])], "candidates": inspected}
    if eligible:
        result["selected"] = min(eligible, key=lambda r: (r["distance_from_center_mm"], -r["area_mm2"], r["index"]))
    return result


def box_for_roi(mask, side):
    require(isinstance(side, int) and not isinstance(side, bool) and side > 0, "Invalid crop size")
    y, x = np.where(mask == 44)
    require(len(x) > 0, "Missing meniscus")
    left = int(np.floor((int(x.min()) + int(x.max()) + 1 - side) / 2))
    top = int(np.floor((int(y.min()) + int(y.max()) + 1 - side) / 2))
    require(left <= x.min() and top <= y.min() and left + side > x.max() and top + side > y.max(), "Fixed crop truncates ROI")
    return [left, top, left + side, top + side]


def crop_exact(array, box):
    x0, y0, x1, y1 = map(int, box)
    require(x1 > x0 and y1 > y0 and array.ndim == 2, "Invalid integer crop")
    output = np.zeros((y1 - y0, x1 - x0), dtype=array.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(array.shape[1], x1), min(array.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        output[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = array[sy0:sy1, sx0:sx1]
    return output


def display_window(native):
    require(np.isfinite(native).all(), "Nonfinite source intensities")
    values = native[native > 0].astype(np.float32)
    require(values.size > 0, "Empty source image")
    low, high = np.percentile(values, [0.5, 99.5]).tolist()
    require(high > low, "Constant source image")
    output = np.rint(np.clip((native.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return output, [low, high]


def anatomy_and_roi(segmentation):
    require(segmentation.dtype == np.uint8 and int(segmentation.max()) <= 64, "Invalid source labels")
    anatomy = np.where(np.isin(segmentation, STRUCTURAL_IDS), segmentation, 0).astype(np.uint8)
    return anatomy, (segmentation == 44).astype(np.uint8)


def relocate(old_path, datasets):
    relative = Path(old_path).relative_to("/mnt/sda/moonsword/datasets")
    new_path = (datasets / relative).resolve()
    require(datasets.resolve() in new_path.parents, "Invalid relocated source path")
    return new_path


def sample_relative(folder, sid, suffix):
    require(sid and all(c.isalnum() or c in "_-" for c in sid), "Unsafe sample ID")
    return f"{folder}/{sid[-2:]}/{sid}{suffix}"


def verify_sample(root, row):
    require(row["lesion_mask_path"] is None and row["spatial_resampling"] is False, "Invalid mask or resampling claim")
    require(row["mask_quality_status"] == "unreviewed_source_prediction", "Unexpected mask-quality claim")
    require(set(row["dimensions"]) == set(GROUPS), "Unexpected dimensions")
    for relative, digest in row["sha256"].items():
        path = (root / relative).resolve()
        require(root.resolve() in path.parents and not Path(relative).is_absolute(), "Unsafe asset path")
        require(sha256(path) == digest, f"Changed asset: {relative}")
    loaded = {}
    for view in ("full", "crop"):
        paths = row["views"][view]
        arrays = {name: read_png(root / paths[name + "_path"]) for name in ("image", "source_segmentation", "anatomy_mask", "lesion_roi_proxy")}
        arrays["native_intensity"] = np.load(root / paths["native_intensity_path"], allow_pickle=False)
        shape = tuple(row["geometry"][view + "_shape_hw"])
        require(all(array.shape == shape for array in arrays.values()), "Image-mask shape mismatch")
        anatomy, roi = anatomy_and_roi(arrays["source_segmentation"])
        require(np.array_equal(anatomy, arrays["anatomy_mask"]), "Anatomy mismatch")
        require(np.array_equal(roi, arrays["lesion_roi_proxy"]) and connected(roi) == 1, "ROI mismatch/connectivity failure")
        require(array_sha(arrays["native_intensity"]) == row["native_array_sha256"][view], "Native intensity mismatch")
        loaded[view] = arrays
    box = row["geometry"]["crop_box_native_xyxy"]
    require(box[2] - box[0] == box[3] - box[1] == row["geometry"]["crop_shape_hw"][0], "Crop size mismatch")
    for name in loaded["full"]:
        require(np.array_equal(crop_exact(loaded["full"][name], box), loaded["crop"][name]), f"Nonexact crop: {name}")
    require(loaded["full"]["lesion_roi_proxy"].sum() == loaded["crop"]["lesion_roi_proxy"].sum(), "Truncated ROI")
    display, window = display_window(loaded["full"]["native_intensity"])
    require(np.array_equal(display, loaded["full"]["image"]) and window == row["intensity_window"], "Display intensity conversion mismatch")


def build_sample(task):
    original, datasets, output, side = task
    sid = original["sample_id"]
    cached_path = output / sample_relative("records", sid, ".json")
    paths = {kind: relocate(original["source_provenance"][f"source_{kind}_record"]["path"], datasets) for kind in ("volume", "mask")}
    signature = {kind: {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for kind, path in paths.items()}
    if cached_path.exists():
        cached = read_json(cached_path)
        require(cached["source_signature"] == signature, "Source changed since cached native export")
        if cached["accepted"]:
            require(cached["row"]["dimensions"] == original["dimensions"], "Changed cached labels")
            verify_sample(output, cached["row"])
        return cached
    image, mask = nib.load(paths["volume"]), nib.load(paths["mask"])
    transform, image_affine, mask_affine, grid_error = check_grids(image, mask)
    raw_mask = np.asanyarray(mask.dataobj)
    require(raw_mask.dtype == np.uint8, "Unexpected mask datatype")
    ras_mask = canonical_array(raw_mask, transform)
    spacing = tuple(float(np.linalg.norm(mask_affine[:3, axis])) for axis in range(3))
    selection = select_native(ras_mask, spacing)
    result = {"sample_id": sid, "patient_id": original["patient_id"], "split": original["split"],
              "accepted": selection["accepted"], "source_signature": signature, "selection": selection}
    if not selection["accepted"]:
        write_json(cached_path, result)
        return result
    index = selection["selected"]["index"]
    segmentation = plane(ras_mask, index)
    raw_image = np.asanyarray(image.dataobj)
    native = plane(canonical_array(raw_image, transform), index)
    display, window = display_window(native)
    anatomy, roi = anatomy_and_roi(segmentation)
    box = box_for_roi(segmentation, side)
    arrays = {"native_intensity": native, "image": display, "source_segmentation": segmentation,
              "anatomy_mask": anatomy, "lesion_roi_proxy": roi}
    views, hashes, array_hashes = {}, {}, {}
    for view in ("full", "crop"):
        views[view] = {"lesion_mask_path": None}
        for name, full_array in arrays.items():
            array = full_array if view == "full" else crop_exact(full_array, box)
            suffix = ".npy" if name == "native_intensity" else ".png"
            relative = sample_relative(f"{view}/{name}", sid, suffix)
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "native_intensity":
                np.save(path, array, allow_pickle=False)
                require(np.array_equal(np.load(path, allow_pickle=False), array), "Native array roundtrip failure")
                array_hashes[view] = array_sha(array)
            else:
                Image.fromarray(array).save(path)
                require(np.array_equal(read_png(path), array), "PNG roundtrip failure")
            views[view][name + "_path"] = relative
            hashes[relative] = sha256(path)
    y, x = np.where(roi)
    centroid = np.array([ras_mask.shape[0] - 1 - x.mean(), index, ras_mask.shape[2] - 1 - y.mean(), 1])
    world_centroid = (image_affine @ centroid)[:3]
    oblique = not np.allclose(image_affine[:3, :3], np.diag(np.diag(image_affine[:3, :3])), atol=1e-4, rtol=0)
    row = {
        **{key: original[key] for key in ("sample_id", "patient_id", "split", "timepoint", "knee_side_code", "dimensions", "label_scope", "slice_specific_expert_validation")},
        **{key: value for key, value in views["crop"].items()}, "views": views,
        "spatial_resampling": False, "mask_quality_status": "unreviewed_source_prediction",
        "mask_source": "official nnU-Net output, not per-case manual ground truth",
        "lesion_mask_status": "not_available_in_source",
        "lesion_roi_proxy_semantics": "source_label_44_entire_medial_meniscus_not_extruded_portion",
        "geometry": {
            "plane": "acquisition_grid_near_coronal", "native_ras_ap_index": index,
            "full_shape_hw": list(native.shape), "crop_shape_hw": [side, side],
            "crop_box_native_xyxy": box, "pixel_spacing_mm_lr_si": [spacing[0], spacing[2]],
            "boundary_padded": box[0] < 0 or box[1] < 0 or box[2] > native.shape[1] or box[3] > native.shape[0],
            "raw_volume_shape": list(image.shape), "orientation_permutation_and_flips": transform.tolist(),
            "image_native_ras_affine": image_affine.tolist(), "mask_native_ras_affine": mask_affine.tolist(),
            "source_grid_max_corner_error_mm": grid_error, "oblique_acquisition": oblique,
            "roi_pixels": int(roi.sum()), "roi_components_8": 1,
            "roi_centroid_world_mm": world_centroid.tolist(),
            "previous_reformatted_ap_index": original["geometry"]["coronal_index_ras"],
            "previous_reformatted_plane_world_y_mm": original["geometry"]["coronal_world_y_mm"],
            "native_roi_centroid_to_previous_plane_y_mm": float(world_centroid[1] - original["geometry"]["coronal_world_y_mm"]),
        },
        "intensity_window": window, "source_signature": signature,
        "native_array_sha256": array_hashes, "sha256": hashes,
    }
    require(connected(crop_exact(roi, box)) == 1 and crop_exact(roi, box).sum() == roi.sum(), "Crop lost ROI")
    result["row"] = row
    write_json(cached_path, result)
    return result


def summary(rows):
    return {"samples": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "grade_1_to_3_samples": sum(r["dimensions"][GROUPS[1]]["value"] > 0 for r in rows),
            "class_counts": {g: dict(sorted(Counter(str(r["dimensions"][g]["value"]) for r in rows).items())) for g in GROUPS},
            "native_full_shapes_hw": dict(Counter("x".join(map(str, r["geometry"]["full_shape_hw"])) for r in rows)),
            "boundary_padded": sum(r["geometry"]["boundary_padded"] for r in rows),
            "oblique_acquisitions": sum(r["geometry"]["oblique_acquisition"] for r in rows)}


def preview(root, rows):
    selected = [next(r for r in rows if r["dimensions"][GROUPS[1]]["value"] == grade) for grade in range(4)]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    panel = max(220, max(r["geometry"]["full_shape_hw"][1] for r in selected))
    row_height = max(r["geometry"]["full_shape_hw"][0] for r in selected) + 70
    sheet = Image.new("RGB", (panel * 4, row_height * 4 + 85), "white")
    draw = ImageDraw.Draw(sheet)
    headings = ("Native image + crop box", "Source structural boundaries", "128px crop, no resize", "128px crop + source masks")
    for col, title in enumerate(headings):
        draw.text((col * panel + 4, 10), title, fill="black", font=font)
    draw.text((4, 35), "1 source pixel = 1 preview pixel. Native anisotropic spacing is NOT stretched. Masks are predictions.", fill="black", font=font)
    colors = [(238, 55, 145), (65, 175, 232), (245, 193, 49), (41, 180, 140)]
    label_groups = [(44,), (49,), tuple(range(50, 56)), tuple(range(58, 65))]
    for row_index, row in enumerate(selected):
        full = read_png(root / row["views"]["full"]["image_path"])
        seg = read_png(root / row["views"]["full"]["source_segmentation_path"])
        plain = Image.fromarray(full).convert("RGB")
        ImageDraw.Draw(plain).rectangle(row["geometry"]["crop_box_native_xyxy"], outline=(35, 145, 255), width=1)
        contour = np.repeat(full[..., None], 3, axis=2)
        for ids, color in zip(label_groups, colors):
            binary = np.isin(seg, ids)
            boundary = binary & ~ndimage.binary_erosion(binary)
            contour[boundary] = color
        crop = Image.open(root / row["image_path"]).convert("RGB")
        crop_contour = Image.fromarray(crop_exact(contour[:, :, 0], row["geometry"]["crop_box_native_xyxy"]))
        crop_rgb = np.stack([crop_exact(contour[:, :, channel], row["geometry"]["crop_box_native_xyxy"]) for channel in range(3)], axis=-1)
        crop_contour = Image.fromarray(crop_rgb)
        panels = (plain, Image.fromarray(contour), crop, crop_contour)
        y = 65 + row_index * row_height
        for col, image in enumerate(panels):
            sheet.paste(image, (col * panel + (panel - image.width) // 2, y))
        label = f"G{row['dimensions'][GROUPS[1]]['value']} | {row['sample_id']} | native index {row['geometry']['native_ras_ap_index']}"
        draw.text((4, y + row_height - 50), label, fill="black", font=font)
        draw.text((4, y + row_height - 30), "Pink: meniscus 44; blue: tibial cartilage 49; yellow: femoral region union; green: tibial region union", fill="black", font=font)
    sheet.save(root / "preview_native.png")
    write_json(root / "preview_samples.json", [{"sample_id": r["sample_id"], "geometry": r["geometry"], "dimensions": r["dimensions"]} for r in selected])


def verify(root, workers):
    marker = read_json(root / "dataset.complete") if (root / "dataset.complete").exists() else None
    if marker:
        for name, digest in marker["metadata_sha256"].items():
            require(sha256(root / name) == digest, f"Changed metadata: {name}")
    seen_samples, seen_patients, total = set(), set(), 0
    for split in SPLITS:
        rows = read_jsonl(root / "samples" / f"{split}.jsonl")
        ids, patients = [r["sample_id"] for r in rows], {r["patient_id"] for r in rows}
        require(len(ids) == len(set(ids)) and not seen_samples.intersection(ids), "Duplicate samples")
        require(not seen_patients.intersection(patients) and all(r["split"] == split for r in rows), "Split/patient leakage")
        selected = read_jsonl(root / "positive_only" / f"{split}.jsonl")
        require(selected == [r for r in rows if r["dimensions"][GROUPS[1]]["value"] > 0], "Positive index mismatch")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(lambda r: verify_sample(root, r), rows):
                total += 1
        seen_samples.update(ids)
        seen_patients.update(patients)
        print(json.dumps({"verified": split, "samples": len(rows)}), flush=True)
    require(total == read_json(root / "dataset_summary.json")["total"]["samples"], "Incomplete export")
    return {"status": "pass", "samples": total, "data_assets_verified": total * 10,
            "no_spatial_resize_or_resampling": True, "native_intensity_roundtrip": True,
            "identical_image_mask_crop_coordinates": True, "all_roi_complete_and_8_connected": True,
            "patient_disjoint": True, "segmentation_accuracy_measured": False,
            "quality_status": "geometry_verified_anatomical_accuracy_unreviewed"}


def build(args):
    source, root, datasets = args.source.resolve(), args.output.resolve(), args.datasets.resolve()
    require(source != root and source not in root.parents, "Output must be independent")
    cohorts = {s: read_jsonl(source / "samples" / f"{s}.jsonl") for s in SPLITS}
    previous_marker = read_json(source / "dataset.complete")
    for s in SPLITS:
        require(sha256(source / "samples" / f"{s}.jsonl") == previous_marker["metadata_sha256"][f"samples/{s}.jsonl"], "Changed source manifest")
    if args.preview_only:
        ids = {r["sample_id"] for r in read_json(source / "preview_samples.json")}
        cohorts = {s: [r for r in rows if r["sample_id"] in ids] for s, rows in cohorts.items()}
    contract = {"version": 2, "source_root": str(source), "source_marker_sha256": sha256(source / "dataset.complete"),
                "datasets_root": str(datasets), "crop_side_native_pixels": args.side, "spatial_resampling": False,
                "orientation": "transpose_and_flip_only", "selection": "native_central_tibia_midpoint_pm3mm_single_component_at_least3mm2",
                "selection_uses_diagnostic_labels": False, "previous_scan_ids_and_labels_retained_if_native_eligible": True,
                "same_exact_reformatted_plane_claimed": False, "preview_only": args.preview_only,
                "structural_mask_ids": list(STRUCTURAL_IDS), "prediction_source": SOURCE_CARD,
                "mask_accuracy_reviewed": False, "script_sha256": sha256(Path(__file__))}
    if root.exists():
        require((root / "dataset_contract.json").exists() and read_json(root / "dataset_contract.json") == contract, "Existing native export contract differs")
        if (root / "dataset.complete").exists():
            print(json.dumps(verify(root, args.workers)), flush=True)
            return
    else:
        root.mkdir(parents=True)
        write_json(root / "dataset_contract.json", contract)
    labels = read_json(source / "source_mask_labels.json")
    write_json(root / "source_mask_labels.json", labels)
    write_json(root / "anatomy_labels.json", {k: v for k, v in labels.items() if v == 0 or v in STRUCTURAL_IDS})
    shutil.copyfile(source / "dimensions.json", root / "dimensions.json")
    (root / "provenance").mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), root / "provenance" / Path(__file__).name)
    results, exclusions = {}, []
    started = time.monotonic()
    for split, originals in cohorts.items():
        rows = []
        tasks = ((r, datasets, root, args.side) for r in originals)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, result in enumerate(pool.map(build_sample, tasks), 1):
                if result["accepted"]:
                    rows.append(result["row"])
                else:
                    exclusions.append(result)
                if count % 100 == 0 or count == len(originals):
                    print(json.dumps({"split": split, "processed": count, "source_total": len(originals), "accepted": len(rows),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        results[split] = rows
        write_jsonl(root / "samples" / f"{split}.jsonl", rows)
        write_jsonl(root / "positive_only" / f"{split}.jsonl", [r for r in rows if r["dimensions"][GROUPS[1]]["value"] > 0])
    all_rows = [r for rows in results.values() for r in rows]
    write_jsonl(root / "excluded_native_geometry.jsonl", exclusions)
    report = {"total": summary(all_rows), "splits": {s: summary(rows) for s, rows in results.items()},
              "source_samples": sum(map(len, cohorts.values())), "native_exclusions": len(exclusions),
              "exclusion_reasons": dict(Counter(r["selection"]["reason"] for r in exclusions)),
              "crop_side_native_pixels": args.side, "segmentation_quality": "source_nnUNet_predictions_not_expert_verified",
              "native_vs_prior_roi_centroid_y_difference_mm": dict(zip(("min", "median", "p95", "max"), map(float, np.percentile([abs(r["geometry"]["native_roi_centroid_to_previous_plane_y_mm"]) for r in all_rows], [0, 50, 95, 100]))))}
    write_json(root / "dataset_summary.json", report)
    preview(root, results["test"])
    (root / "README.md").write_text(
        "# Native-grid medial meniscus subset\n\n"
        "No spatial resize, affine resampling, interpolation, mask smoothing or mask correction is performed. "
        "MRI and mask use the same original voxel grid and the same integer crop box. "
        "Only shared axis permutation/reversal changes orientation. Original oblique acquisition angles are preserved.\n\n"
        f"Default inputs are {args.side}x{args.side} native-pixel crops. Full native planes are retained. "
        "PNG images apply intensity windowing only; native_intensity_path NPY files retain exact source intensity values and dtype. "
        "All mask PNGs retain integer labels. All loader paths are relative to this root.\n\n"
        "The old orthogonal-RAS 896x896 images are NOT used. Old resampled layer indices are NOT reused as raw-layer indices. "
        "The same anatomical center/radius/connectivity rule is applied anew in the native grid. "
        "Examinations without an eligible native slice are listed in excluded_native_geometry.jsonl. "
        "For retained examinations, labels and patient splits are unchanged. Source records are in records/.\n\n"
        "The masks are official nnU-Net predictions, NOT manually verified ground truth. "
        "Geometry and pixel consistency do NOT establish segmentation accuracy. "
        "The label-44 meniscus ROI is a proxy, not the extruded portion; lesion_mask_path stays null. "
        "Structural anatomy excludes finding-specific IDs 42,43,45,46; the unchanged source_segmentation retains them. "
        "Femoral/tibial subregion boundaries are not necessarily whole-bone boundaries. "
        "Background/other labels do not establish absence of anatomy or disease.\n\n"
        "MRI pixels are anisotropic: pixel aspect ratio is kept without stretching. "
        "The preview shows each source pixel once and does not enlarge crops; grouped bone outlines are visualization only. "
        "Source prediction errors are not repaired or smoothed away. The two class dimensions retain regional examination labels, "
        "not slice-reviewed diagnoses.\n\n"
        f"Official source: {SOURCE_CARD}\n\n"
        "Verify: python provenance/build_knee_native_subset.py verify --output DATASET_ROOT\n",
        encoding="utf-8")
    validation = verify(root, args.workers)
    write_json(root / "validation.json", validation)
    files = [p for p in root.rglob("*") if p.is_file()]
    write_json(root / "storage.json", {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
               "excludes": ["storage.json", "dataset.complete"]})
    meta = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".py")]
    write_json(root / "dataset.complete", {"status": "complete", "metadata_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in meta}})
    print(json.dumps({"output": str(root), "summary": report, "validation": validation}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Native-grid MRI/mask crops without spatial resize or resampling")
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    require(args.workers > 0 and args.side > 0, "Invalid configuration")
    if args.mode == "verify":
        print(json.dumps(verify(args.output.resolve(), args.workers)), flush=True)
    else:
        require(args.source is not None and args.datasets is not None, "Source roots required")
        build(args)


if __name__ == "__main__":
    main()
