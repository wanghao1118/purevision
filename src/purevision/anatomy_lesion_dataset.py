from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class AnatomyLesionPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 896,
        patch_size: int = 14,
        manifest_name: str = "metadata.csv",
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        frame = pd.read_csv(self.root / manifest_name)
        required = {
            "sample_id",
            "patient_id",
            "split",
            "image_path",
            "side_patch_mask_path",
            "lesion_patch_mask_path",
            "lung_class",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"R34 manifest is missing columns: {missing}")
        self.frame = frame[frame["split"].eq(split)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No R34 samples found for split={split}")
        self.split = str(split)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(self.root / str(row["image_path"])) as image:
            image = image.convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            pixels = (pixels - 0.5) / 0.5
        side = self._read_patch_mask(str(row["side_patch_mask_path"]))
        lesion = self._read_patch_mask(str(row["lesion_patch_mask_path"]))
        if np.any(lesion & ~side):
            raise ValueError(f"Lesion mask escapes side mask for {row['sample_id']}")
        if not lesion.any() or not (side & ~lesion).any():
            raise ValueError(f"R34 sample lacks lesion or normal patches: {row['sample_id']}")
        lung_class = int(row["lung_class"])
        if lung_class not in {1, 2}:
            raise ValueError(f"Unknown lung class {lung_class} for {row['sample_id']}")
        return {
            "pixel_values": torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            "side_patch_mask": torch.from_numpy(side.reshape(-1).copy()),
            "lesion_patch_mask": torch.from_numpy(lesion.reshape(-1).copy()),
            "lung_class": lung_class,
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "slice_index": int(row.get("slice_index", -1)),
            "side": str(row.get("side", "")),
        }

    def _read_patch_mask(self, relative_path: str) -> np.ndarray:
        with Image.open(self.root / relative_path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        grid = self.image_size // self.patch_size
        if values.shape != (grid, grid):
            raise ValueError(f"Expected patch mask {(grid, grid)}, got {values.shape}")
        return values


def collate_anatomy_lesion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "side_patch_mask": torch.stack([sample["side_patch_mask"] for sample in samples]),
        "lesion_patch_mask": torch.stack(
            [sample["lesion_patch_mask"] for sample in samples]
        ),
        "lung_class": torch.tensor(
            [sample["lung_class"] for sample in samples], dtype=torch.long
        ),
        "sample_id": [sample["sample_id"] for sample in samples],
        "patient_id": [sample["patient_id"] for sample in samples],
        "slice_index": torch.tensor(
            [sample["slice_index"] for sample in samples], dtype=torch.long
        ),
        "side": [sample["side"] for sample in samples],
    }
