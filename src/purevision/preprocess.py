from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .attributes import AttributeSpace, majority_calcification_group
from .config import load_config, with_path_overrides

for _name, _value in (("int", np.int64), ("float", np.float64), ("bool", np.bool_)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)


SEMANTIC_FEATURES = (
    "texture",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
)

METADATA_FIELDS = (
    "sample_id",
    "patient_id",
    "scan_id",
    "series_instance_uid",
    "lesion_index",
    "reader_count",
    "image_path",
    "nodule_mask_path",
    "lung_mask_path",
    "position",
    "side",
    "patient_x_mm",
    "height_zone",
    "relative_lung_z",
    "max_slice_index",
    "max_slice_z_mm",
    "lung_inferior_z_mm",
    "lung_superior_z_mm",
    "lung_effective_slices",
    "roi_row_start",
    "roi_row_stop",
    "roi_col_start",
    "roi_col_stop",
    "diameter_mm_mean",
    "diameter_mm_std",
    "surface_area_mm2_mean",
    "volume_mm3_mean",
    *tuple(f"{feature}_mean" for feature in SEMANTIC_FEATURES),
    "calcification_mean",
    "calcification_group",
    "density_level",
    "sphericity_level",
    "margin_level",
    "lobulation_level",
    "spiculation_level",
)

SCAN_FIELDS = (
    "patient_id",
    "scan_id",
    "series_instance_uid",
    "status",
    "samples",
    "ambiguous_clusters_skipped",
    "lung_effective_slices",
    "elapsed_seconds",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D LIDC lesion samples")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def completed_scan_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["scan_id"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def annotation_statistics(annotations: list[Any]) -> dict[str, Any]:
    diameters = np.asarray([annotation.diameter for annotation in annotations], dtype=float)
    row: dict[str, Any] = {
        "reader_count": len(annotations),
        "diameter_mm_mean": float(diameters.mean()),
        "diameter_mm_std": float(diameters.std()),
        "surface_area_mm2_mean": float(np.mean([a.surface_area for a in annotations])),
        "volume_mm3_mean": float(np.mean([a.volume for a in annotations])),
    }
    for feature in SEMANTIC_FEATURES:
        row[f"{feature}_mean"] = float(
            np.mean([getattr(annotation, feature) for annotation in annotations])
        )
    calcification_codes = [int(annotation.calcification) for annotation in annotations]
    row["calcification_mean"] = float(np.mean(calcification_codes))


    row["calcification_group"] = majority_calcification_group(calcification_codes)
    return row


def majority_mask(annotations: list[Any]) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    boxes = [annotation.bbox() for annotation in annotations]
    starts = np.asarray([[int(axis.start) for axis in box] for box in boxes])
    stops = np.asarray([[int(axis.stop) for axis in box] for box in boxes])
    union_start = starts.min(axis=0)
    union_stop = stops.max(axis=0)
    votes = np.zeros(tuple((union_stop - union_start).tolist()), dtype=np.uint8)

    for annotation, box in zip(annotations, boxes):
        local = tuple(
            slice(int(axis.start) - int(union_start[i]), int(axis.stop) - int(union_start[i]))
            for i, axis in enumerate(box)
        )
        votes[local] += annotation.boolean_mask().astype(np.uint8)

    mask = votes > (len(annotations) / 2.0)
    if not mask.any():
        mask = votes >= max(1, math.ceil(len(annotations) / 2.0))
    bbox = tuple(
        slice(int(union_start[i]), int(union_stop[i])) for i in range(3)
    )
    return mask, bbox


def full_mask_on_slice(
    mask: np.ndarray,
    bbox: tuple[slice, slice, slice],
    slice_index: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(image_shape, dtype=np.uint8)
    local_z = slice_index - int(bbox[2].start)
    if 0 <= local_z < mask.shape[2]:
        result[bbox[0], bbox[1]] = mask[:, :, local_z].astype(np.uint8)
    return result


def max_area_slice(mask: np.ndarray, bbox: tuple[slice, slice, slice]) -> int:
    areas = mask.sum(axis=(0, 1))
    return int(bbox[2].start) + int(np.argmax(areas))


def load_scan(scan: Any) -> tuple[np.ndarray, dict[str, Any]]:
    images = scan.load_all_dicom_images(verbose=False)
    if not images:
        raise ValueError("Scan has no readable DICOM images")

    volume = np.stack(
        [
            image.pixel_array.astype(np.float32) * float(image.RescaleSlope)
            + float(image.RescaleIntercept)
            for image in images
        ],
        axis=-1,
    ).astype(np.int16)
    first = images[0]
    geometry = {
        "rows": int(first.Rows),
        "columns": int(first.Columns),
        "pixel_spacing": np.asarray(first.PixelSpacing, dtype=float),
        "orientation": np.asarray(first.ImageOrientationPatient, dtype=float),
        "origins": np.stack(
            [np.asarray(image.ImagePositionPatient, dtype=float) for image in images]
        ),
    }
    return volume, geometry


def classify_side(
    centroid: np.ndarray, geometry: dict[str, Any], slice_index: int
) -> tuple[str, float]:
    row_direction = geometry["orientation"][:3]
    column_direction = geometry["orientation"][3:]
    spacing = geometry["pixel_spacing"]
    origin = geometry["origins"][slice_index]
    patient_x = (
        origin[0]
        + centroid[1] * spacing[1] * row_direction[0]
        + centroid[0] * spacing[0] * column_direction[0]
    )
    return ("left" if patient_x >= 0 else "right"), float(patient_x)


def classify_position(side: str, relative_lung_z: float) -> tuple[str, str]:
    relative_lung_z = float(np.clip(relative_lung_z, 0.0, 1.0))
    if side == "left":
        height = "lower" if relative_lung_z < 0.5 else "upper"
    else:
        if relative_lung_z < 1.0 / 3.0:
            height = "lower"
        elif relative_lung_z < 2.0 / 3.0:
            height = "middle"
        else:
            height = "upper"
    return f"{side}_{height}", height


def lung_window(volume_slice: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(volume_slice.astype(np.float32), lower, upper)
    return np.round((clipped - lower) * 255.0 / (upper - lower)).astype(np.uint8)


def square_lung_roi(mask_zyx: np.ndarray, margin_fraction: float) -> tuple[int, int, int, int]:
    projection = mask_zyx.any(axis=0)
    rows, cols = np.where(projection)
    if rows.size == 0:
        raise ValueError("Lung segmentation is empty")
    height, width = projection.shape
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    margin = math.ceil(max(r1 - r0, c1 - c0) * margin_fraction)
    side = min(max(r1 - r0, c1 - c0) + 2 * margin, min(height, width))
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    start_r = max(0, min(height - side, center_r - side // 2))
    start_c = max(0, min(width - side, center_c - side // 2))
    return start_r, start_r + side, start_c, start_c + side


def lung_segment(scan: Any, volume: np.ndarray, inferer: Any) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.moveaxis(volume, -1, 0))
    image.SetSpacing(
        (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_spacing))
    )
    segmentation = np.asarray(inferer.apply(image), dtype=np.uint8)
    if segmentation.shape != (volume.shape[2], volume.shape[0], volume.shape[1]):
        raise ValueError(
            f"Unexpected lung mask shape {segmentation.shape} for volume {volume.shape}"
        )
    return segmentation


def save_sample_images(
    output_dir: Path,
    sample_id: str,
    image: np.ndarray,
    nodule_mask: np.ndarray,
    lung_mask: np.ndarray,
) -> tuple[str, str, str]:
    relative_paths = (
        Path("images") / f"{sample_id}.png",
        Path("nodule_masks") / f"{sample_id}.png",
        Path("lung_masks") / f"{sample_id}.png",
    )
    for path in relative_paths:
        (output_dir / path).parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(output_dir / relative_paths[0], optimize=True)
    Image.fromarray((nodule_mask > 0).astype(np.uint8) * 255, mode="L").save(
        output_dir / relative_paths[1], optimize=True
    )
    Image.fromarray((lung_mask > 0).astype(np.uint8) * 255, mode="L").save(
        output_dir / relative_paths[2], optimize=True
    )
    return tuple(path.as_posix() for path in relative_paths)


def main() -> int:
    args = parse_args()
    config = with_path_overrides(load_config(args.config), processed=args.output_dir)
    data_config = config["data"]
    output_dir = Path(config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    scans_path = output_dir / "preprocessing_scans.csv"
    if args.overwrite:
        for path in (metadata_path, scans_path):
            path.unlink(missing_ok=True)

    import pylidc as pl
    from lungmask import LMInferer

    attribute_space = AttributeSpace(config["attribute_space"])
    attribute_space.save(output_dir / "attribute_space.json")
    inferer = LMInferer(
        modelname="R231",
        fillmodel=None,
        force_cpu=args.device == "cpu",
        batch_size=20,
    )

    completed = completed_scan_ids(scans_path)
    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id, pl.Scan.id).all()
    if args.start_index < 0 or args.start_index > len(scans):
        raise ValueError(f"start-index must be between 0 and {len(scans)}")
    scans = scans[args.start_index :]
    if args.limit is not None:
        scans = scans[: args.limit]
    final_ordinal = args.start_index + len(scans)

    for ordinal, scan in enumerate(scans, start=args.start_index + 1):
        if scan.id in completed:
            continue
        started = time.time()
        sample_rows: list[dict[str, Any]] = []
        effective_count = 0
        ambiguous_clusters_skipped = 0
        try:
            volume, geometry = load_scan(scan)
            lung_mask_zyx = lung_segment(scan, volume, inferer)
            effective_indices = np.flatnonzero(lung_mask_zyx.any(axis=(1, 2)))
            if effective_indices.size < 2:
                raise ValueError("Lung model found fewer than two effective slices")
            effective_count = int(effective_indices.size)
            slice_zvals = np.asarray(scan.slice_zvals, dtype=float)
            effective_z = slice_zvals[effective_indices]
            inferior_z = float(effective_z.min())
            superior_z = float(effective_z.max())
            roi = square_lung_roi(
                lung_mask_zyx > 0, float(data_config["roi_margin_fraction"])
            )
            clusters = scan.cluster_annotations(verbose=False)

            for lesion_index, annotations in enumerate(clusters):
                if len(annotations) > 4:
                    ambiguous_clusters_skipped += 1
                    continue
                statistics = annotation_statistics(annotations)
                if statistics["diameter_mm_mean"] <= float(data_config["diameter_threshold_mm"]):
                    continue
                consensus_mask, bbox = majority_mask(annotations)
                selected_slice = max_area_slice(consensus_mask, bbox)
                full_nodule_mask = full_mask_on_slice(
                    consensus_mask, bbox, selected_slice, volume.shape[:2]
                )
                if not full_nodule_mask.any():
                    raise ValueError("Consensus mask is empty on its selected slice")

                points = np.argwhere(consensus_mask)
                centroid = points.mean(axis=0) + np.asarray(
                    [bbox[0].start, bbox[1].start, bbox[2].start], dtype=float
                )
                side, patient_x = classify_side(centroid, geometry, selected_slice)
                selected_z = float(slice_zvals[selected_slice])
                relative_lung_z = (selected_z - inferior_z) / (superior_z - inferior_z)
                position, height_zone = classify_position(side, relative_lung_z)

                r0, r1, c0, c1 = roi
                windowed = lung_window(
                    volume[:, :, selected_slice],
                    float(data_config["lung_window_center"]),
                    float(data_config["lung_window_width"]),
                )[r0:r1, c0:c1]
                nodule_crop = full_nodule_mask[r0:r1, c0:c1]
                lung_crop = (lung_mask_zyx[selected_slice] > 0)[r0:r1, c0:c1]

                sample_id = f"{scan.patient_id}_s{scan.id}_n{lesion_index}"
                image_path, nodule_path, lung_path = save_sample_images(
                    output_dir, sample_id, windowed, nodule_crop, lung_crop
                )
                row = attribute_space.enrich_row(
                    {
                        "sample_id": sample_id,
                        "patient_id": scan.patient_id,
                        "scan_id": scan.id,
                        "series_instance_uid": scan.series_instance_uid,
                        "lesion_index": lesion_index,
                        "image_path": image_path,
                        "nodule_mask_path": nodule_path,
                        "lung_mask_path": lung_path,
                        "position": position,
                        "side": side,
                        "patient_x_mm": patient_x,
                        "height_zone": height_zone,
                        "relative_lung_z": float(np.clip(relative_lung_z, 0.0, 1.0)),
                        "max_slice_index": selected_slice,
                        "max_slice_z_mm": selected_z,
                        "lung_inferior_z_mm": inferior_z,
                        "lung_superior_z_mm": superior_z,
                        "lung_effective_slices": effective_count,
                        "roi_row_start": r0,
                        "roi_row_stop": r1,
                        "roi_col_start": c0,
                        "roi_col_stop": c1,
                        **statistics,
                    }
                )
                sample_rows.append(row)

            scan_row = {
                "patient_id": scan.patient_id,
                "scan_id": scan.id,
                "series_instance_uid": scan.series_instance_uid,
                "status": "ok",
                "samples": len(sample_rows),
                "ambiguous_clusters_skipped": ambiguous_clusters_skipped,
                "lung_effective_slices": effective_count,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": "",
            }
        except Exception as exc:
            scan_row = {
                "patient_id": scan.patient_id,
                "scan_id": scan.id,
                "series_instance_uid": scan.series_instance_uid,
                "status": "error",
                "samples": 0,
                "ambiguous_clusters_skipped": ambiguous_clusters_skipped,
                "lung_effective_slices": effective_count,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(traceback.format_exc(limit=5), file=sys.stderr, flush=True)

        append_csv(metadata_path, METADATA_FIELDS, sample_rows)
        append_csv(scans_path, SCAN_FIELDS, [scan_row])
        print(
            json.dumps(
                {
                    "event": "scan_complete",
                    "ordinal": ordinal,
                    "total": final_ordinal,
                    "patient_id": scan.patient_id,
                    "status": scan_row["status"],
                    "samples": len(sample_rows),
                    "seconds": scan_row["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
