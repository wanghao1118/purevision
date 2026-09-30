from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


VIEW = re.compile(r"(P_\d{5}_(?:LEFT|RIGHT)_(?:CC|MLO))")
FILES = (
    "image.png",
    "lesion_mask.png",
    "anatomy_mask.png",
    "anatomy_training_labels.png",
    "breast_mask.png",
    "pectoral_pseudomask.png",
    "diagnostic_text.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_rows(metadata_root: Path) -> dict[tuple[str, str, str], list[dict]]:
    result = defaultdict(list)
    for lesion_type in ("mass", "calc"):
        for partition in ("train", "test"):
            path = metadata_root / f"{lesion_type}_case_description_{partition}_set.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    full_id = Path(row["image file path"]).parts[0]
                    match = VIEW.search(full_id)
                    if match is None:
                        raise ValueError(f"Invalid source image ID: {full_id}")
                    result[(lesion_type, partition, match.group(1))].append(row)
    return result


def series_directories(root: Path) -> dict[tuple[str, str], Path]:
    result = {}
    for row in jsonl(root / "metadata" / "verified_instances.jsonl"):
        path = root / row["path"]
        key = (path.parts[-3], path.parts[-2])
        previous = result.get(key)
        if previous is not None and previous != path.parent:
            raise ValueError(f"Series UID maps to multiple directories: {key}")
        result[key] = path.parent
    return result


def source_series_dir(root: Path, csv_path: str, index: dict[tuple[str, str], Path]) -> Path:
    source = Path(csv_path)
    if len(source.parts) < 4:
        raise ValueError(f"Invalid DICOM source path: {csv_path}")
    path = root / "images" / source.parent
    if not path.is_dir():
        path = index.get((source.parts[1], source.parts[2]))
        if path is None or not path.is_dir():
            raise FileNotFoundError(f"No verified DICOM series for {csv_path}")
    return path


def only_dicom(directory: Path) -> Path:
    paths = list(directory.glob("*.dcm"))
    if len(paths) != 1:
        raise ValueError(f"Expected one DICOM in {directory}, found {len(paths)}")
    return paths[0]


def plan_samples(source_root: Path, v1_root: Path, limit: int | None) -> list[dict]:
    rows = source_rows(source_root / "metadata")
    index = series_directories(source_root)
    plan = []
    for sample in jsonl(v1_root / "manifest" / "all.jsonl"):
        source_id = sample["source_full_image_id"]
        match = VIEW.search(source_id)
        if match is None:
            raise ValueError(f"Invalid manifest image ID: {source_id}")
        lesion_type = "mass" if sample["source_labels"]["categorical"]["lesion_type"] == 0 else "calc"
        partition = "test" if sample["split"] == "test" else "train"
        candidates = rows[(lesion_type, partition, match.group(1))]
        exact = [row for row in candidates if Path(row["image file path"]).parts[0] == source_id]
        chosen = exact if exact else candidates
        if len(chosen) != 1:
            raise ValueError(f"Ambiguous CSV match for {sample['sample_id']}: {len(chosen)}")
        csv_row = chosen[0]
        full_dicom = only_dicom(source_series_dir(source_root, csv_row["image file path"], index))
        roi_series = source_series_dir(source_root, csv_row["ROI mask file path"], index)
        plan.append({
            "sample": sample,
            "full_dicom": str(full_dicom),
            "roi_series": str(roi_series),
            "source_csv_lesion_type": lesion_type,
            "source_csv_partition": partition,
            "source_csv_abnormality_id": csv_row["abnormality id"],
            "source_csv_image_path": csv_row["image file path"],
            "source_csv_roi_path": csv_row["ROI mask file path"],
        })
    if len(plan) != 2591 or len({item["sample"]["sample_id"] for item in plan}) != len(plan):
        raise ValueError(f"Unexpected plan size or duplicate IDs: {len(plan)}")
    return plan[:limit] if limit else plan


def render_uint8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low:
        raise ValueError("DICOM has no usable intensity range")
    return np.round(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.astype(np.float64).ravel(), right.astype(np.float64).ravel())[0, 1])


def render_matching_reference(dataset: pydicom.Dataset, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = dataset.pixel_array
    if raw.ndim != 2:
        raise ValueError("Expected a two-dimensional DICOM image")
    candidates = {"raw": raw}
    try:
        candidates["voi_lut"] = apply_voi_lut(raw, dataset)
    except (AttributeError, ValueError, TypeError):
        pass
    best = None
    for transform, values in candidates.items():
        normalized = render_uint8(values)
        for inverted in (False, True):
            image = 255 - normalized if inverted else normalized
            resized = np.asarray(
                Image.fromarray(image).resize((896, 896), Image.Resampling.BICUBIC)
            )
            score = correlation(resized, reference)
            if best is None or score > best[0]:
                best = (score, image, transform, inverted)
    assert best is not None
    if best[0] < 0.98:
        raise ValueError(f"Native image does not align with 896 cache: r={best[0]:.6f}")
    return best[1], {
        "transform": best[2],
        "inverted": best[3],
        "percentile_window": [0.5, 99.5],
        "correlation_to_reference_896": best[0],
    }


def load_native_roi(series: Path, image_shape: tuple[int, int], reference: np.ndarray) -> tuple[np.ndarray, Path]:
    def matching(paths):
        valid = []
        for path in paths:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            if (int(header.Rows), int(header.Columns)) != image_shape:
                continue
            mask = pydicom.dcmread(path).pixel_array > 0
            if not mask.any():
                continue
            resized = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize((896, 896), Image.Resampling.NEAREST)
            ) > 0
            if np.array_equal(resized, reference):
                valid.append((mask, path))
        return valid

    valid = matching(series.glob("*.dcm"))
    if not valid:
        case_dir = series.parent.parent
        sibling_dicoms = (
            path for path in case_dir.rglob("*.dcm") if path.parent != series
        )
        valid = matching(sibling_dicoms)
    if len(valid) != 1:
        raise ValueError(f"Expected one native ROI matching 896 cache in {series}, found {len(valid)}")
    return valid[0]


def native_anatomy(
    image: np.ndarray, lesion: np.ndarray, old_anatomy: np.ndarray, revision: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    if revision.is_dir():
        breast = np.asarray(Image.open(revision / "native_breast_mask.png").convert("L")) > 0
        pectoral = np.asarray(Image.open(revision / "native_pectoral_pseudomask.png").convert("L")) > 0
        old_roi = np.asarray(Image.open(revision / "native_lesion_roi.png").convert("L")) > 0
        if not np.array_equal(old_roi, lesion):
            raise ValueError(f"Source native ROI differs from frozen revision: {revision}")
        source = "existing_native_mask_revision_20260916"
    else:
        from scripts.revise_cbis_native_anatomy_masks import grabcut_breast_mask

        try:
            breast = grabcut_breast_mask(image)
            source = "native_seeded_grabcut_plus_reprojected_v1_pectoral_pseudomask"
        except ValueError as error:
            if str(error) != "Cannot determine breast chest-wall side":
                raise
            breast = grabcut_intensity_seed_fallback(image)
            source = "native_intensity_seed_grabcut_plus_reprojected_v1_pectoral_pseudomask"
        pectoral = cv2.resize(
            (old_anatomy == 2).astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        pectoral &= breast
    if breast.shape != image.shape or pectoral.shape != image.shape or not breast.any():
        raise ValueError("Native anatomy masks have invalid geometry or are empty")
    if (lesion & breast).sum() / lesion.sum() < 0.975:
        raise ValueError("Native breast foreground excludes over 2.5% of source ROI")
    return breast, pectoral, source


def grabcut_intensity_seed_fallback(image: np.ndarray) -> np.ndarray:
    from scripts.revise_cbis_native_anatomy_masks import (
        breast_seed_without_scanner_frame,
        scanner_footer_start,
    )

    otsu, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough = image > max(12, round(otsu * 0.25))
    footer_start = scanner_footer_start(image, rough)
    rough[footer_start:] = False
    rough = breast_seed_without_scanner_frame(rough)
    seed_radius = max(5, round(min(image.shape) * 0.008))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (seed_radius * 2 + 1, seed_radius * 2 + 1)
    )
    sure_foreground = cv2.erode(rough.astype(np.uint8), kernel) > 0
    sure_background = cv2.dilate(rough.astype(np.uint8), kernel) == 0
    if not sure_foreground.any() or not sure_background.any():
        raise ValueError("Intensity-seeded GrabCut requires both foreground and background")
    seeds = np.full(image.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    seeds[rough] = cv2.GC_PR_FGD
    seeds[sure_background] = cv2.GC_BGD
    seeds[sure_foreground] = cv2.GC_FGD
    seeds[footer_start:] = cv2.GC_BGD
    models = (np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64))
    cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), seeds, None, *models, 3, cv2.GC_INIT_WITH_MASK)
    selected = np.isin(seeds, (cv2.GC_FGD, cv2.GC_PR_FGD))
    selected[footer_start:] = False
    from scripts.revise_cbis_native_anatomy_masks import largest_filled_component

    return largest_filled_component(selected)


def validate_case(case_dir: Path) -> dict:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    for name in FILES:
        if sha256(case_dir / name) != metadata["sha256"][name]:
            raise ValueError(f"Hash mismatch: {case_dir / name}")
    image = np.asarray(Image.open(case_dir / "image.png").convert("L"))
    lesion = np.asarray(Image.open(case_dir / "lesion_mask.png").convert("L")) > 0
    anatomy = np.asarray(Image.open(case_dir / "anatomy_mask.png").convert("L"))
    training = np.asarray(Image.open(case_dir / "anatomy_training_labels.png").convert("L"))
    breast = np.asarray(Image.open(case_dir / "breast_mask.png").convert("L")) > 0
    pectoral = np.asarray(Image.open(case_dir / "pectoral_pseudomask.png").convert("L")) > 0
    if any(values.shape != image.shape for values in (lesion, anatomy, training, breast, pectoral)):
        raise ValueError(f"Geometry mismatch: {case_dir}")
    expected = np.zeros(image.shape, dtype=np.uint8)
    expected[breast] = 1
    expected[pectoral] = 2
    if not lesion.any() or not np.array_equal(anatomy, expected):
        raise ValueError(f"Invalid lesion/anatomy mask: {case_dir}")
    expected[lesion] = 255
    if not np.array_equal(training, expected):
        raise ValueError(f"Invalid training labels: {case_dir}")
    if metadata["geometry_hw"] != list(image.shape):
        raise ValueError(f"Metadata geometry mismatch: {case_dir}")
    return metadata


def build_case(plan: dict, v1_root: str, revision_root: str, output_root: str) -> dict:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    sample = plan["sample"]
    sample_id = sample["sample_id"]
    v1_case = Path(v1_root) / "cases" / sample_id
    case_dir = Path(output_root) / "cases" / sample_id
    if case_dir.exists():
        return validate_case(case_dir)

    full_dicom = Path(plan["full_dicom"])
    dataset = pydicom.dcmread(full_dicom)
    reference = np.asarray(Image.open(v1_case / "image.png").convert("L"))
    old_roi = np.asarray(Image.open(v1_case / "lesion_mask.png").convert("L")) > 0
    old_anatomy = np.asarray(Image.open(v1_case / "anatomy_mask.png").convert("L"))
    image, rendering = render_matching_reference(dataset, reference)
    lesion, roi_dicom = load_native_roi(Path(plan["roi_series"]), image.shape, old_roi)
    revision = Path(revision_root) / sample_id
    breast, pectoral, anatomy_source = native_anatomy(image, lesion, old_anatomy, revision)
    anatomy = np.zeros(image.shape, dtype=np.uint8)
    anatomy[breast] = 1
    anatomy[pectoral] = 2
    training = anatomy.copy()
    training[lesion] = 255

    case_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(case_dir / "image.png", compress_level=1)
    Image.fromarray(lesion.astype(np.uint8) * 255).save(case_dir / "lesion_mask.png", compress_level=1)
    Image.fromarray(anatomy).save(case_dir / "anatomy_mask.png", compress_level=1)
    Image.fromarray(training).save(case_dir / "anatomy_training_labels.png", compress_level=1)
    Image.fromarray(breast.astype(np.uint8) * 255).save(case_dir / "breast_mask.png", compress_level=1)
    Image.fromarray(pectoral.astype(np.uint8) * 255).save(case_dir / "pectoral_pseudomask.png", compress_level=1)
    shutil.copy2(v1_case / "diagnostic_text.txt", case_dir / "diagnostic_text.txt")
    metadata = {
        "sample_id": sample_id,
        "patient_id": sample["patient_id"],
        "split": sample["split"],
        "image_view": sample["image_view"],
        "source_labels": sample["source_labels"],
        "geometry_hw": list(image.shape),
        "source_full_dicom": str(full_dicom),
        "source_full_dicom_sha256": sha256(full_dicom),
        "source_roi_dicom": str(roi_dicom),
        "source_roi_dicom_sha256": sha256(roi_dicom),
        "source_csv_image_path": plan["source_csv_image_path"],
        "source_csv_roi_path": plan["source_csv_roi_path"],
        "source_csv_abnormality_id": plan["source_csv_abnormality_id"],
        "rendering": rendering,
        "anatomy_source": anatomy_source,
        "grabcut_rng_seed": None if revision.is_dir() else 0,
        "pectoral_source": sample["pectoral_source"],
        "pectoral_checkpoint_sha256": sample["pectoral_checkpoint_sha256"],
        "diagnostic_text_source": sample["diagnostic_text_source"],
        "anatomy_classes": {"0": "background", "1": "breast_tissue", "2": "pectoral_muscle"},
        "anatomy_training_ignore_label": 255,
        "lesion_pixels": int(lesion.sum()),
        "lesion_inside_breast_fraction": float((lesion & breast).sum() / lesion.sum()),
        "breast_pixels": int(breast.sum()),
        "pectoral_pixels": int(pectoral.sum()),
        "sha256": {name: sha256(case_dir / name) for name in FILES},
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_case(case_dir)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--retry-errors", type=Path)
    parser.add_argument("--errors-output", type=Path)
    parser.add_argument("--finalize-accepted", action="store_true")
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    plan = plan_samples(args.source_root, args.v1_root, args.limit)
    if args.retry_errors:
        if not args.errors_output:
            parser.error("--retry-errors requires --errors-output")
        retry_ids = {row["sample_id"] for row in jsonl(args.retry_errors)}
        retry_plan = [item for item in plan if item["sample"]["sample_id"] in retry_ids]
        if len(retry_plan) != len(retry_ids):
            raise ValueError("Retry IDs do not match source plan")
        errors = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(build_case, item, str(args.v1_root), str(args.revision_root), str(args.output)): item
                for item in retry_plan
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception as error:
                    errors.append({"sample_id": item["sample"]["sample_id"], "error": repr(error)})
        errors.sort(key=lambda row: row["sample_id"])
        write_jsonl(args.errors_output, errors)
        print(json.dumps({"retried": len(retry_plan), "recovered": len(retry_plan) - len(errors),
                          "excluded_candidates": len(errors)}), flush=True)
        return

    excluded = {row["sample_id"]: row for row in jsonl(args.exclusions)} if args.exclusions else {}
    if args.finalize_accepted and not args.exclusions:
        parser.error("--finalize-accepted requires --exclusions")
    if excluded and not (args.finalize_accepted or args.verify_only):
        parser.error("--exclusions requires --finalize-accepted or --verify-only")
    planned_ids = {item["sample"]["sample_id"] for item in plan}
    if excluded.keys() - planned_ids:
        raise ValueError("Exclusions contain IDs outside source plan")
    accepted_plan = [item for item in plan if item["sample"]["sample_id"] not in excluded]
    if args.verify_only or args.finalize_accepted:
        for sample_id in excluded:
            if (args.output / "cases" / sample_id).exists():
                raise ValueError(f"Excluded case has output files: {sample_id}")
        manifest = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(validate_case, args.output / "cases" / item["sample"]["sample_id"]): item
                for item in accepted_plan
            }
            for future in as_completed(futures):
                manifest.append(future.result())
                if len(manifest) % 100 == 0:
                    print(json.dumps({"verified": len(manifest)}), flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "cases").mkdir(exist_ok=True)
        manifest = []
        errors = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(build_case, item, str(args.v1_root), str(args.revision_root), str(args.output)): item
                for item in plan
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    manifest.append(future.result())
                except Exception as error:
                    errors.append({"sample_id": item["sample"]["sample_id"], "error": repr(error)})
                if (len(manifest) + len(errors)) % 20 == 0:
                    print(json.dumps({"processed": len(manifest) + len(errors), "ok": len(manifest), "failed": len(errors)}), flush=True)
        if errors:
            write_jsonl(args.output / "build_errors.jsonl", errors)
            raise RuntimeError(f"{len(errors)} native cases failed; see build_errors.jsonl")
    if not args.verify_only:
        manifest.sort(key=lambda row: row["sample_id"])
        write_jsonl(args.output / "manifest.jsonl", manifest)
        summary = {
            "samples": len(manifest),
            "candidate_samples": len(plan),
            "excluded_samples": len(excluded),
            "patients": len({row["patient_id"] for row in manifest}),
            "split_samples": dict(Counter(row["split"] for row in manifest)),
            "anatomy_sources": dict(Counter(row["anatomy_source"] for row in manifest)),
            "image_shape_min_hw": [min(row["geometry_hw"][i] for row in manifest) for i in range(2)],
            "image_shape_max_hw": [max(row["geometry_hw"][i] for row in manifest) for i in range(2)],
            "source_v1_root": str(args.v1_root),
            "source_dicom_root": str(args.source_root),
            "source_revision_root": str(args.revision_root),
            "checkpoint_sha256": sorted({row["pectoral_checkpoint_sha256"] for row in manifest}),
            "pilot_limit": args.limit,
            "minimum_source_roi_inside_breast_fraction": 0.975,
        }
        if excluded:
            write_jsonl(args.output / "excluded_cases.jsonl", sorted(excluded.values(), key=lambda row: row["sample_id"]))
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if len(manifest) != len(accepted_plan):
        raise ValueError("Native case count differs from accepted plan")
    if not args.verify_only:
        (args.output / "dataset.complete").write_text("complete\n", encoding="ascii")
    print(json.dumps({"verified": len(manifest), "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
