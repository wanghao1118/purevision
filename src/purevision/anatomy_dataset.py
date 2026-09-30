from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class AnatomyPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 896,
        patch_size: int = 14,
        manifest_name: str = "metadata.csv",
        image_cache_root: str | Path | None = None,
        source_label_values: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.image_cache_root = (
            Path(image_cache_root) if image_cache_root is not None else None
        )
        self.source_label_values = (
            None
            if source_label_values is None
            else tuple(int(value) for value in source_label_values)
        )
        self.label_lookup: np.ndarray | None = None
        if self.source_label_values is not None:
            if (
                not self.source_label_values
                or len(set(self.source_label_values)) != len(self.source_label_values)
                or min(self.source_label_values) < 0
                or max(self.source_label_values) >= 255
            ):
                raise ValueError(
                    "source_label_values must be unique values in the range [0, 254]"
                )
            self.label_lookup = np.full(256, 255, dtype=np.int64)
            for target, source in enumerate(self.source_label_values):
                self.label_lookup[source] = target
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        frame = pd.read_csv(self.root / manifest_name)
        required = {
            "sample_id",
            "patient_id",
            "split",
            "image_path",
            "patch_label_path",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Anatomy manifest is missing columns: {missing}")
        self.frame = frame[frame["split"].eq(split)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No anatomy samples found for split={split}")
        self.split = str(split)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        cache_path = row.get("cache_image_path")
        preferred = (
            self.image_cache_root / Path(str(row["image_path"])).name
            if self.image_cache_root is not None
            else Path(str(cache_path))
            if isinstance(cache_path, str)
            else None
        )
        image_path = (
            preferred
            if preferred is not None and preferred.is_file()
            else self.root / str(row["image_path"])
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            pixels = (pixels - 0.5) / 0.5
        with Image.open(self.root / str(row["patch_label_path"])) as label_image:
            patch_labels = np.asarray(label_image.convert("L"), dtype=np.int64)
        if self.label_lookup is not None:
            patch_labels = self.label_lookup[patch_labels]

        grid = self.image_size // self.patch_size
        if patch_labels.shape != (grid, grid):
            raise ValueError(
                f"Expected patch labels {(grid, grid)}, got {patch_labels.shape}"
            )
        return {
            "pixel_values": torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            "patch_labels": torch.from_numpy(patch_labels.reshape(-1).copy()),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "slice_index": int(row.get("slice_index", -1)),
        }


def collate_anatomy_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "patch_labels": torch.stack([sample["patch_labels"] for sample in samples]),
        "sample_id": [sample["sample_id"] for sample in samples],
        "patient_id": [sample["patient_id"] for sample in samples],
        "slice_index": torch.tensor(
            [sample["slice_index"] for sample in samples], dtype=torch.long
        ),
    }
