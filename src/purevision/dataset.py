from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .attributes import LOCATION_NAMES, ORDINAL_SOURCE_FIELDS
from .attribute_2d import Attribute2DSpace
from .target_space import build_target_space


def _to_pixel_values(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def _to_mask_pixels(mask: Image.Image, image_size: int) -> torch.Tensor:
    mask = mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    return torch.from_numpy(array)


def _to_patch_mask(mask_pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    tensor = mask_pixels[None, None]

    weights = F.max_pool2d(tensor, kernel_size=patch_size, stride=patch_size)
    return weights[0, 0]


def dihedral_transform(
    pixel_values: torch.Tensor,
    mask_pixels: torch.Tensor,
    rotations: int,
    reflect: bool,
) -> tuple[torch.Tensor, torch.Tensor]:

    rotations = int(rotations) % 4
    pixels = torch.rot90(pixel_values, rotations, dims=(-2, -1))
    mask = torch.rot90(mask_pixels, rotations, dims=(-2, -1))
    if reflect:
        pixels = torch.flip(pixels, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    return pixels.contiguous(), mask.contiguous()


def patch_aligned_lesion_crop(
    pixel_values: torch.Tensor,
    mask_pixels: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if pixel_values.ndim != 3 or mask_pixels.ndim != 2:
        raise ValueError("Expected image [C,H,W] and mask [H,W]")
    if pixel_values.shape[-2:] != mask_pixels.shape:
        raise ValueError("Image and mask must share the canonical global grid")
    height, width = mask_pixels.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Canonical image dimensions must be divisible by patch_size")

    global_patch_mask = _to_patch_mask(mask_pixels, patch_size)
    lesion_rows, lesion_cols = torch.where(global_patch_mask > 0)
    if lesion_rows.numel() == 0:
        raise ValueError("No lesion patch survived")
    row_min = int(lesion_rows.min())
    row_max = int(lesion_rows.max())
    col_min = int(lesion_cols.min())
    col_max = int(lesion_cols.max())

    cropped_pixels = pixel_values[
        :,
        row_min * patch_size : (row_max + 1) * patch_size,
        col_min * patch_size : (col_max + 1) * patch_size,
    ]
    cropped_patch_mask = global_patch_mask[
        row_min : row_max + 1, col_min : col_max + 1
    ].flatten()

    global_grid_width = width // patch_size
    rows = torch.arange(row_min, row_max + 1, dtype=torch.long)
    cols = torch.arange(col_min, col_max + 1, dtype=torch.long)
    global_position_ids = (
        rows[:, None] * global_grid_width + cols[None, :]
    ).flatten()
    crop_bounds = torch.tensor(
        [row_min, row_max + 1, col_min, col_max + 1], dtype=torch.long
    )
    return cropped_pixels, cropped_patch_mask, global_position_ids, crop_bounds


def collate_lesion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:

    variable_keys = {"pixel_values", "patch_mask", "global_position_ids"}
    result: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key in variable_keys:
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


class LIDCLesionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        attribute_config: dict[str, Any],
        image_size: int,
        patch_size: int,
        input_mode: str = "global",
        random_dihedral_augmentation: bool = False,
    ):
        self.root = Path(processed_dir)
        frame = pd.read_csv(self.root / "metadata.csv")
        split_payload = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))
        patients = set(split_payload["splits"][split])
        self.frame = frame[frame["patient_id"].isin(patients)].reset_index(drop=True)
        self.split = split
        self.attribute_space = build_target_space(attribute_config)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.input_mode = str(input_mode)
        self.random_dihedral_augmentation = bool(random_dihedral_augmentation)
        if self.input_mode not in {"global", "lesion_patch_rectangle"}:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index].to_dict()
        with Image.open(self.root / str(row["image_path"])) as image:
            pixel_values = _to_pixel_values(image, self.image_size)
        with Image.open(self.root / str(row["nodule_mask_path"])) as mask:
            mask_pixels = _to_mask_pixels(mask, self.image_size)

        if self.random_dihedral_augmentation:
            pixel_values, mask_pixels = dihedral_transform(
                pixel_values,
                mask_pixels,
                rotations=int(torch.randint(0, 4, ()).item()),
                reflect=bool(torch.randint(0, 2, ()).item()),
            )

        if self.input_mode == "lesion_patch_rectangle":
            try:
                pixel_values, patch_mask, global_position_ids, crop_bounds = (
                    patch_aligned_lesion_crop(
                        pixel_values, mask_pixels, self.patch_size
                    )
                )
            except ValueError as error:
                raise ValueError(f"{error} for {row['sample_id']}") from error
        else:
            patch_mask = _to_patch_mask(mask_pixels, self.patch_size).flatten()
            if patch_mask.sum() == 0:
                raise ValueError(f"No lesion patch survived for {row['sample_id']}")
            global_position_ids = torch.arange(patch_mask.numel(), dtype=torch.long)
            grid_size = self.image_size // self.patch_size
            crop_bounds = torch.tensor([0, grid_size, 0, grid_size], dtype=torch.long)

        ordinal_source_fields = (
            self.attribute_space.ordinal_source_fields
            if isinstance(self.attribute_space, Attribute2DSpace)
            else ORDINAL_SOURCE_FIELDS
        )
        ordinal_scores = torch.tensor(
            [float(row[source]) for source in ordinal_source_fields.values()],
            dtype=torch.float32,
        )
        if isinstance(self.attribute_space, Attribute2DSpace):
            ordinal_levels = torch.tensor(
                [
                    self.attribute_space.ordinal_level(name, float(row[source]))
                    for name, source in ordinal_source_fields.items()
                ],
                dtype=torch.long,
            )
            calcification = self.attribute_space.calcification_index(
                int(row[self.attribute_space.calcification_column])
            )
        else:
            ordinal_levels = torch.tensor(
                [int(row[f"{name}_level"]) for name in ORDINAL_SOURCE_FIELDS],
                dtype=torch.long,
            )
            calcification = int(row["calcification_group"])
        return {
            "pixel_values": pixel_values,
            "patch_mask": patch_mask,
            "global_position_ids": global_position_ids,
            "crop_bounds": crop_bounds,
            "ideal_target": torch.from_numpy(self.attribute_space.encode(row)),
            "location": torch.tensor(LOCATION_NAMES.index(str(row["position"])), dtype=torch.long),
            "ordinal_scores": ordinal_scores,
            "ordinal_levels": ordinal_levels,
            "calcification": torch.tensor(calcification, dtype=torch.long),
            "diameter": torch.tensor(float(row["diameter_mm_mean"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "position_name": str(row["position"]),
        }
