from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from purevision.anatomy_dataset import AnatomyPatchDataset
from purevision.anatomy_model import load_anatomy_patch_encoder
from purevision.attribute_2d import ATTRIBUTE_NAMES, Attribute2DSpace
from purevision.dataset import LIDCLesionDataset
from purevision.model import load_trained_model
from purevision.protocol import validate_dataset_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen R39/R30 patch embeddings for text alignment"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--kind", choices=("anatomy", "lesion", "all"), default="all"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=None
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Cache already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_pixels(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
        values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(((values - 0.5) / 0.5).transpose(2, 0, 1).copy())


def read_patch_mask(path: Path, image_size: int, patch_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (image_size, image_size), Image.Resampling.NEAREST
        )
        values = np.asarray(image, dtype=np.uint8) > 0
    grid = image_size // patch_size
    blocks = values.reshape(grid, patch_size, grid, patch_size).transpose(0, 2, 1, 3)
    return blocks.any(axis=(2, 3)).reshape(-1)


def selected_indices(
    labels: np.ndarray,
    *,
    classes: int,
    per_class: int | None,
    excluded: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    eligible = np.ones(len(labels), dtype=bool)
    if excluded is not None:
        eligible[np.asarray(excluded, dtype=np.int64)] = False
    if per_class is None:
        return np.flatnonzero(eligible)
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_index in range(classes):
        local = np.flatnonzero(eligible & (labels == class_index))
        if len(local) > per_class:
            local = np.sort(generator.choice(local, size=per_class, replace=False))
        selected.append(local)
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)


def anatomy_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    required = ("patient_id", "scan_id", "slice_index")
    if any(name not in row or pd.isna(row[name]) for name in required):
        return None
    return str(row["patient_id"]), int(row["scan_id"]), int(row["slice_index"])


def build_lesion_exclusions(
    r34_root: Path, image_size: int, patch_size: int
) -> dict[tuple[str, int, int], np.ndarray]:
    frame = pd.read_csv(r34_root / "metadata.csv")
    unions: dict[tuple[str, int, int], np.ndarray] = {}
    for row in frame.to_dict("records"):
        key = anatomy_key(row)
        if key is None:
            continue
        mask = read_patch_mask(
            r34_root / str(row["lesion_patch_mask_path"]), image_size, patch_size
        )
        unions[key] = mask if key not in unions else unions[key] | mask
    return {key: np.flatnonzero(mask) for key, mask in unions.items()}


@torch.no_grad()
def extract_anatomy_split(
    config: dict[str, Any], split: str, device: torch.device, overwrite: bool
) -> dict[str, Any]:
    paths = config["paths"]
    data = config["data"]
    extraction = config["cache_extraction"]
    image_size = int(data["image_size"])
    patch_size = int(data["patch_size"])
    output = Path(paths["anatomy_cache"]) / split
    prepare_output(output, overwrite)
    dataset = AnatomyPatchDataset(
        paths["r33_anatomy_processed"],
        split,
        image_size=image_size,
        patch_size=patch_size,
    )
    per_class_value = extraction["anatomy_patches_per_class_per_slice"].get(split)
    per_class = None if per_class_value is None else int(per_class_value)
    exclusions = (
        build_lesion_exclusions(
            Path(paths["r34_lesion_processed"]), image_size, patch_size
        )
        if split in {"train", "val"}
        else {}
    )
    selections: list[np.ndarray] = []
    total = 0
    for dataset_index, row in enumerate(dataset.frame.to_dict("records")):
        with Image.open(
            Path(paths["r33_anatomy_processed"]) / str(row["patch_label_path"])
        ) as image:
            labels = np.asarray(image.convert("L"), dtype=np.uint8).reshape(-1)
        indices = selected_indices(
            labels,
            classes=7,
            per_class=per_class,
            excluded=exclusions.get(anatomy_key(row)),
            seed=int(config["seed"]) + dataset_index,
        )
        selections.append(indices)
        total += len(indices)

    dimension = int(data["vision_embedding_dimension"])
    reuse_path_value = paths.get("r41_anatomy_test_embedding_cache")
    reuse_path = Path(reuse_path_value) if reuse_path_value else None
    reuse_embeddings = split == "test" and per_class is None and reuse_path is not None and reuse_path.exists()
    if reuse_embeddings:
        existing = np.load(reuse_path, mmap_mode="r")
        if existing.shape != (total, dimension) or existing.dtype != np.float16:
            raise RuntimeError(
                f"R41 cache contract mismatch: {existing.shape}/{existing.dtype} vs "
                f"{(total, dimension)}/float16"
            )
        os.symlink(reuse_path.resolve(), output / "embeddings.npy")
        embeddings = existing
    else:
        embeddings = np.lib.format.open_memmap(
            output / "embeddings.npy",
            mode="w+",
            dtype=np.float16,
            shape=(total, dimension),
        )
    labels_output = np.lib.format.open_memmap(
        output / "labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    dataset_indices = np.lib.format.open_memmap(
        output / "dataset_indices.npy", mode="w+", dtype=np.int32, shape=(total,)
    )
    patch_indices = np.lib.format.open_memmap(
        output / "patch_indices.npy", mode="w+", dtype=np.uint16, shape=(total,)
    )
    cursor = 0
    if reuse_embeddings:
        anatomy_root = Path(paths["r33_anatomy_processed"])
        for dataset_index, row in enumerate(dataset.frame.to_dict("records")):
            indices = selections[dataset_index]
            count = len(indices)
            if not count:
                continue
            with Image.open(anatomy_root / str(row["patch_label_path"])) as image:
                sample_labels = np.asarray(image.convert("L"), dtype=np.uint8).reshape(-1)[indices]
            labels_output[cursor : cursor + count] = sample_labels.astype(np.uint8)
            dataset_indices[cursor : cursor + count] = dataset_index
            patch_indices[cursor : cursor + count] = indices.astype(np.uint16)
            cursor += count
        model = None
    else:
        model = load_anatomy_patch_encoder(
            paths["model"], paths["r39_checkpoint"], dtype=torch.bfloat16
        ).to(device)
        model.eval()
        batch_size = int(extraction.get("vision_batch_size", 1))
        for start in range(0, len(dataset), batch_size):
            stop = min(len(dataset), start + batch_size)
            samples = [dataset[index] for index in range(start, stop)]
            pixels = torch.stack([sample["pixel_values"] for sample in samples]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = model(pixels)["embedding"]
            for local, sample in enumerate(samples):
                dataset_index = start + local
                indices = selections[dataset_index]
                count = len(indices)
                if not count:
                    continue
                values = (
                    tokens[local, torch.from_numpy(indices).to(device)]
                    .float()
                    .cpu()
                    .numpy()
                )
                embeddings[cursor : cursor + count] = values.astype(np.float16)
                sample_labels = sample["patch_labels"].numpy()[indices]
                labels_output[cursor : cursor + count] = sample_labels.astype(np.uint8)
                dataset_indices[cursor : cursor + count] = dataset_index
                patch_indices[cursor : cursor + count] = indices.astype(np.uint16)
                cursor += count
    if cursor != total:
        raise RuntimeError(f"Anatomy cache count mismatch: wrote {cursor}, expected {total}")
    for array in (labels_output, dataset_indices, patch_indices):
        array.flush()
    if not reuse_embeddings:
        embeddings.flush()
    manifest = {
        "kind": "anatomy_alignment",
        "dataset_id": config["dataset"]["dataset_id"],
        "dataset_release": config["dataset"]["release"],
        "source_root": config["dataset"]["source_root"],
        "label_provenance_zh": config["dataset"]["label_provenance_zh"],
        "split_manifest": config["dataset"]["split_manifest"],
        "split_manifest_sha256": config["dataset"]["split_manifest_sha256"],
        "split": split,
        "count": total,
        "embedding_dimension": dimension,
        "encoder": "R39 All-vessel",
        "checkpoint": str(paths["r39_checkpoint"]),
        "checkpoint_sha256": sha256_file(Path(paths["r39_checkpoint"])),
        "all_patches": per_class is None,
        "patches_per_class_per_slice": per_class,
        "lesion_patches_excluded": split in {"train", "val"},
        "embedding_reused_from_r41": reuse_embeddings,
        "reused_embedding_path": str(reuse_path) if reuse_embeddings else None,
        "reference": "dataset_indices.npy + patch_indices.npy against R33 split metadata",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if model is not None:
        del model
    torch.cuda.empty_cache()
    return manifest


@dataclass(frozen=True)
class LesionPatchReference:
    r30_dataset_index: int
    r34_sample_id: str
    patch_index: int
    anatomy_label: int
    attribute_labels: tuple[int, ...]


@dataclass
class LesionSliceGroup:
    r30_dataset_index: int
    r34_row: dict[str, Any]
    patches: dict[int, LesionPatchReference | None]


def build_lesion_groups(
    dataset: LIDCLesionDataset,
    r30_root: Path,
    r34_root: Path,
    image_size: int,
    patch_size: int,
    anatomy_label_map: dict[int, int] | None = None,
) -> tuple[list[LesionSliceGroup], int, int]:
    if not isinstance(dataset.attribute_space, Attribute2DSpace):
        raise TypeError("R30 alignment requires the formal per-attribute 2D target space")
    r34 = pd.read_csv(r34_root / "metadata.csv")
    label_map = anatomy_label_map or {}
    source_to_row: dict[str, dict[str, Any]] = {}
    for row in r34.to_dict("records"):
        for source in str(row["source_sample_ids"]).split(";"):
            if source in source_to_row:
                raise RuntimeError(f"R30 sample maps to multiple R34 rows: {source}")
            source_to_row[source] = row
    groups: dict[str, LesionSliceGroup] = {}
    missing = 0
    conflicts = 0
    for dataset_index, row in enumerate(dataset.frame.to_dict("records")):
        sample_id = str(row["sample_id"])
        r34_row = source_to_row.get(sample_id)
        if r34_row is None:
            missing += 1
            continue
        points = dataset.attribute_space.encode_points(row)
        labels = tuple(int(point.level - 1) for point in points)
        if tuple(point.attribute for point in points) != tuple(ATTRIBUTE_NAMES):
            raise RuntimeError("R30 attribute order does not match the alignment contract")
        lesion_mask = read_patch_mask(
            r30_root / str(row["nodule_mask_path"]), image_size, patch_size
        )
        group_key = str(r34_row["sample_id"])
        group = groups.setdefault(
            group_key,
            LesionSliceGroup(dataset_index, r34_row, {}),
        )
        for patch_index in np.flatnonzero(lesion_mask):
            reference = LesionPatchReference(
                r30_dataset_index=dataset_index,
                r34_sample_id=group_key,
                patch_index=int(patch_index),
                anatomy_label=label_map.get(
                    int(r34_row["lung_class"]), int(r34_row["lung_class"])
                ),
                attribute_labels=labels,
            )
            existing = group.patches.get(int(patch_index))
            if existing is None and int(patch_index) in group.patches:
                continue
            if existing is not None and existing.attribute_labels != labels:
                group.patches[int(patch_index)] = None
                conflicts += 1
            elif existing is None:
                group.patches[int(patch_index)] = reference
    return list(groups.values()), missing, conflicts


@torch.no_grad()
def extract_lesion_split(
    config: dict[str, Any], split: str, device: torch.device, overwrite: bool
) -> dict[str, Any]:
    paths = config["paths"]
    data = config["data"]
    extraction = config["cache_extraction"]
    image_size = int(data["image_size"])
    patch_size = int(data["patch_size"])
    output = Path(paths["lesion_cache"]) / split
    prepare_output(output, overwrite)
    dataset = LIDCLesionDataset(
        paths["r30_processed"],
        split,
        config["r30_attribute_space"],
        image_size,
        patch_size,
        input_mode="global",
        random_dihedral_augmentation=False,
    )
    groups, missing_samples, conflicts = build_lesion_groups(
        dataset,
        Path(paths["r30_processed"]),
        Path(paths["r34_lesion_processed"]),
        image_size,
        patch_size,
        {
            int(source): int(target)
            for source, target in data.get("lesion_anatomy_label_map", {}).items()
        },
    )
    references = [
        reference
        for group in groups
        for _, reference in sorted(group.patches.items())
        if reference is not None
    ]
    total = len(references)
    dimension = int(data["vision_embedding_dimension"])
    anatomy_embeddings = np.lib.format.open_memmap(
        output / "anatomy_embeddings.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total, dimension),
    )
    pathology_embeddings = np.lib.format.open_memmap(
        output / "pathology_embeddings.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total, dimension),
    )
    anatomy_labels = np.lib.format.open_memmap(
        output / "anatomy_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    attribute_labels = np.lib.format.open_memmap(
        output / "attribute_labels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, len(ATTRIBUTE_NAMES)),
    )
    r30_model = load_trained_model(
        paths["model"],
        paths["r30_checkpoint"],
        target_dimension=2 * len(ATTRIBUTE_NAMES),
        dtype=torch.bfloat16,
    ).to(device)
    r39_model = load_anatomy_patch_encoder(
        paths["model"], paths["r39_checkpoint"], dtype=torch.bfloat16
    ).to(device)
    r30_model.eval()
    r39_model.eval()
    r30_root = Path(paths["r30_processed"])
    r34_root = Path(paths["r34_lesion_processed"])
    batch_size = int(extraction.get("vision_batch_size", 1))
    cursor = 0
    reference_rows: list[dict[str, Any]] = []
    usable_groups = [
        group for group in groups if any(value is not None for value in group.patches.values())
    ]
    for start in range(0, len(usable_groups), batch_size):
        batch_groups = usable_groups[start : start + batch_size]
        pathology_pixels = []
        anatomy_pixels = []
        for group in batch_groups:
            r30_row = dataset.frame.iloc[group.r30_dataset_index]
            pathology_pixels.append(
                read_pixels(r30_root / str(r30_row["image_path"]), image_size)
            )
            anatomy_pixels.append(
                read_pixels(r34_root / str(group.r34_row["image_path"]), image_size)
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pathology_tokens = r30_model.vision_encoder(
                pixel_values=torch.stack(pathology_pixels).to(device), return_dict=True
            ).last_hidden_state
            anatomy_tokens = r39_model(
                torch.stack(anatomy_pixels).to(device)
            )["embedding"]
        for local, group in enumerate(batch_groups):
            local_references = [
                value
                for _, value in sorted(group.patches.items())
                if value is not None
            ]
            indices = torch.tensor(
                [value.patch_index for value in local_references],
                dtype=torch.long,
                device=device,
            )
            count = len(local_references)
            pathology_embeddings[cursor : cursor + count] = (
                pathology_tokens[local, indices].float().cpu().numpy().astype(np.float16)
            )
            anatomy_embeddings[cursor : cursor + count] = (
                anatomy_tokens[local, indices].float().cpu().numpy().astype(np.float16)
            )
            anatomy_labels[cursor : cursor + count] = np.asarray(
                [value.anatomy_label for value in local_references], dtype=np.uint8
            )
            attribute_labels[cursor : cursor + count] = np.asarray(
                [value.attribute_labels for value in local_references], dtype=np.uint8
            )
            for value in local_references:
                reference_rows.append(
                    {
                        "r30_dataset_index": value.r30_dataset_index,
                        "r34_sample_id": value.r34_sample_id,
                        "patch_index": value.patch_index,
                        "anatomy_label": value.anatomy_label,
                    }
                )
            cursor += count
    if cursor != total:
        raise RuntimeError(f"Lesion cache count mismatch: wrote {cursor}, expected {total}")
    for array in (
        anatomy_embeddings,
        pathology_embeddings,
        anatomy_labels,
        attribute_labels,
    ):
        array.flush()
    with (output / "references.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    manifest = {
        "kind": "lesion_alignment",
        "dataset_id": config["dataset"]["dataset_id"],
        "dataset_release": config["dataset"]["release"],
        "source_root": config["dataset"]["source_root"],
        "label_provenance_zh": config["dataset"]["label_provenance_zh"],
        "split_manifest": config["dataset"]["split_manifest"],
        "split_manifest_sha256": config["dataset"]["split_manifest_sha256"],
        "split": split,
        "count": total,
        "embedding_dimension": dimension,
        "anatomy_encoder": "R39 All-vessel",
        "phenotype_encoder": "R30 Global",
        "r39_checkpoint": str(paths["r39_checkpoint"]),
        "r39_checkpoint_sha256": sha256_file(Path(paths["r39_checkpoint"])),
        "r30_checkpoint": str(paths["r30_checkpoint"]),
        "r30_checkpoint_sha256": sha256_file(Path(paths["r30_checkpoint"])),
        "attribute_order": list(ATTRIBUTE_NAMES),
        "anatomy_label_map": data.get("lesion_anatomy_label_map", {}),
        "missing_r30_samples_without_r34_pair": missing_samples,
        "dropped_conflicting_patch_targets": conflicts,
        "deduplication_key": "r34_sample_id + patch_index",
        "reference": "references.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del r30_model, r39_model
    torch.cuda.empty_cache()
    return manifest


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_dataset_protocol(config, verify_files=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Embedding extraction requires CUDA")
    device = torch.device(str(config.get("device", "cuda:0")))
    splits: Iterable[str] = args.splits or ("train", "val", "test")
    summaries: list[dict[str, Any]] = []
    for split in splits:
        if args.kind in {"anatomy", "all"}:
            summaries.append(
                extract_anatomy_split(config, split, device, args.overwrite)
            )
        if args.kind in {"lesion", "all"}:
            summaries.append(
                extract_lesion_split(config, split, device, args.overwrite)
            )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
