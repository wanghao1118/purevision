from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_manifest(root: Path, expected_kind: str) -> dict[str, Any]:
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("kind") != expected_kind:
        raise ValueError(
            f"Expected {expected_kind!r} cache at {root}, got {payload.get('kind')!r}"
        )
    return payload


def _mmap(root: Path, name: str) -> np.ndarray:
    path = root / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


class AnatomyAlignmentCache(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root, "anatomy_alignment")
        self.embeddings = _mmap(self.root, "embeddings")
        self.labels = _mmap(self.root, "labels")
        if self.embeddings.ndim != 2:
            raise ValueError("anatomy embeddings must have shape [patches, dimension]")
        if self.labels.shape != (self.embeddings.shape[0],):
            raise ValueError("anatomy cache labels do not match embeddings")

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": torch.from_numpy(
                np.asarray(self.embeddings[index], dtype=np.float32).copy()
            ),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class LesionAlignmentCache(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root, "lesion_alignment")
        self.anatomy_embeddings = _mmap(self.root, "anatomy_embeddings")
        self.pathology_embeddings = _mmap(self.root, "pathology_embeddings")
        self.anatomy_labels = _mmap(self.root, "anatomy_labels")
        self.attribute_labels = _mmap(self.root, "attribute_labels")
        count = self.anatomy_embeddings.shape[0]
        if self.anatomy_embeddings.ndim != 2:
            raise ValueError("lesion anatomy embeddings must be a matrix")
        if self.pathology_embeddings.shape != self.anatomy_embeddings.shape:
            raise ValueError("paired anatomy/pathology embedding shapes differ")
        if self.anatomy_labels.shape != (count,):
            raise ValueError("lesion anatomy labels do not match embeddings")
        if self.attribute_labels.ndim != 2 or self.attribute_labels.shape[0] != count:
            raise ValueError("lesion attribute labels do not match embeddings")

    def __len__(self) -> int:
        return int(self.anatomy_embeddings.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "anatomy_embedding": torch.from_numpy(
                np.asarray(self.anatomy_embeddings[index], dtype=np.float32).copy()
            ),
            "pathology_embedding": torch.from_numpy(
                np.asarray(self.pathology_embeddings[index], dtype=np.float32).copy()
            ),
            "anatomy_label": torch.tensor(
                int(self.anatomy_labels[index]), dtype=torch.long
            ),
            "attribute_labels": torch.from_numpy(
                np.asarray(self.attribute_labels[index], dtype=np.int64).copy()
            ),
        }
