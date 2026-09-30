from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ANATOMY_NAMES = (
    "outside_body",
    "left_lung",
    "right_lung",
    "vessel",
    "heart",
    "bone",
    "peripheral_soft_tissue",
)
PRIMARY_ANATOMY_NAMES = ANATOMY_NAMES[1:]

OUTSIDE_BODY = 0
LEFT_LUNG = 1
RIGHT_LUNG = 2
VESSEL = 3
HEART = 4
BONE = 5
PERIPHERAL_SOFT_TISSUE = 6

LEFT_LUNG_TOTAL_IDS = (10, 11)
RIGHT_LUNG_TOTAL_IDS = (12, 13, 14)
HEART_TOTAL_IDS = (51,)
VESSEL_TOTAL_IDS = tuple(range(52, 61)) + tuple(range(62, 69))
LUNG_VESSEL_IDS = (3, 4)
BONE_TOTAL_IDS = (
    tuple(range(25, 51))
    + tuple(range(69, 79))
    + (91,)
    + tuple(range(92, 117))
)


@dataclass(frozen=True)
class PatchLabelResult:
    labels: np.ndarray
    confidence: np.ndarray
    fractions: np.ndarray


def compose_anatomy_volume(
    total_labels: np.ndarray,
    lung_vessel_labels: np.ndarray,
    body_labels: np.ndarray,
) -> np.ndarray:

    if not (
        total_labels.shape == lung_vessel_labels.shape == body_labels.shape
    ):
        raise ValueError(
            "Total, lung-vessel, and body label volumes must have identical shapes"
        )

    total = np.asarray(total_labels)
    lung_vessels = np.asarray(lung_vessel_labels)
    body = np.asarray(body_labels)
    result = np.full(total.shape, OUTSIDE_BODY, dtype=np.uint8)

    body_mask = body > 0
    left_lung = np.isin(total, LEFT_LUNG_TOTAL_IDS)
    right_lung = np.isin(total, RIGHT_LUNG_TOTAL_IDS)
    heart = np.isin(total, HEART_TOTAL_IDS)
    bone = np.isin(total, BONE_TOTAL_IDS)
    vessel = np.isin(total, VESSEL_TOTAL_IDS) | np.isin(
        lung_vessels, LUNG_VESSEL_IDS
    )



    result[body_mask] = PERIPHERAL_SOFT_TISSUE
    result[left_lung] = LEFT_LUNG
    result[right_lung] = RIGHT_LUNG
    result[heart] = HEART
    result[bone] = BONE
    result[vessel] = VESSEL
    return result


def patchify_anatomy_labels(
    pixel_labels: np.ndarray,
    *,
    patch_size: int,
    vessel_min_fraction: float = 0.01,
    bone_min_fraction: float = 0.03,
) -> PatchLabelResult:

    labels = np.asarray(pixel_labels)
    if labels.ndim != 2:
        raise ValueError("pixel_labels must be a 2D array")
    if labels.shape[0] % patch_size or labels.shape[1] % patch_size:
        raise ValueError("pixel label dimensions must be divisible by patch_size")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= len(ANATOMY_NAMES):
        raise ValueError("pixel_labels contains an unknown anatomy class")
    if not 0.0 <= vessel_min_fraction <= 1.0:
        raise ValueError("vessel_min_fraction must be in [0, 1]")
    if not 0.0 <= bone_min_fraction <= 1.0:
        raise ValueError("bone_min_fraction must be in [0, 1]")

    rows = labels.shape[0] // patch_size
    cols = labels.shape[1] // patch_size
    blocks = labels.reshape(rows, patch_size, cols, patch_size).transpose(0, 2, 1, 3)
    counts = np.stack(
        [(blocks == index).sum(axis=(2, 3)) for index in range(len(ANATOMY_NAMES))],
        axis=-1,
    )
    fractions = counts.astype(np.float32) / float(patch_size * patch_size)
    patch_labels = fractions.argmax(axis=-1).astype(np.uint8)

    bone_priority = fractions[..., BONE] >= float(bone_min_fraction)
    vessel_priority = fractions[..., VESSEL] >= float(vessel_min_fraction)
    patch_labels[bone_priority] = BONE
    patch_labels[vessel_priority] = VESSEL
    confidence = np.take_along_axis(
        fractions, patch_labels[..., None], axis=-1
    )[..., 0]
    return PatchLabelResult(
        labels=patch_labels,
        confidence=confidence.astype(np.float32),
        fractions=fractions,
    )


def regular_simplex_targets(classes: int) -> np.ndarray:

    if classes < 2:
        raise ValueError("classes must be at least two")
    targets = np.eye(classes, dtype=np.float32)
    targets -= targets.mean(axis=0, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    return targets


def mult_window_rgb(
    hu_slice: np.ndarray,
    windows: tuple[tuple[float, float], ...] = (
        (-600.0, 1500.0),
        (40.0, 400.0),
        (400.0, 1800.0),
    ),
) -> np.ndarray:

    if len(windows) != 3:
        raise ValueError("Exactly three CT windows are required")
    hu = np.asarray(hu_slice, dtype=np.float32)
    channels: list[np.ndarray] = []
    for center, width in windows:
        if width <= 0:
            raise ValueError("Window width must be positive")
        lower = float(center) - float(width) / 2.0
        upper = float(center) + float(width) / 2.0
        clipped = np.clip(hu, lower, upper)
        channels.append(
            np.round((clipped - lower) * 255.0 / (upper - lower)).astype(np.uint8)
        )
    return np.stack(channels, axis=-1)
