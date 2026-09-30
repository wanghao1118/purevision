from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .attributes import CALCIFICATION_NAMES, ORDINAL_SOURCE_FIELDS, plan_ordinal_level


ATTRIBUTE_NAMES = (*ORDINAL_SOURCE_FIELDS, "calcification", "size")


@dataclass(frozen=True)
class AttributePoint:
    attribute: str
    source_value: float
    level: int
    label: str
    target: np.ndarray
    display: np.ndarray
    prototype: np.ndarray
    within_level_position: float
    measurement_std: float | None = None


def semicircle_prototypes(levels: int, radius: float) -> np.ndarray:

    if levels < 2:
        raise ValueError("At least two prototype levels are required")
    angles = np.linspace(math.pi, 0.0, levels, dtype=np.float64)
    result = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    result[np.abs(result) < 1e-12] = 0.0
    return result


def _grade_tangent(level: int, levels: int) -> np.ndarray:

    angle = math.pi * (1.0 - (level - 1.0) / (levels - 1.0))
    return np.asarray([math.sin(angle), -math.cos(angle)], dtype=np.float64)


def _ordinal_within_level(score: float, level: int, levels: int) -> float:
    width = 4.0 / levels
    lower = 1.0 + (level - 1.0) * width
    upper = 1.0 + level * width
    midpoint = (lower + upper) / 2.0
    return float(np.clip((score - midpoint) / (width / 2.0), -1.0, 1.0))


def _ordinal_within_bounds(score: float, lower: float, upper: float) -> float:
    midpoint = (lower + upper) / 2.0
    return float(np.clip((score - midpoint) / ((upper - lower) / 2.0), -1.0, 1.0))


def _log_within_bin(value: float, lower: float, upper: float) -> float:
    clipped = float(np.clip(value, lower, upper))
    log_value = math.log(clipped)
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    return float(2.0 * (log_value - log_lower) / (log_upper - log_lower) - 1.0)


def _display_offset(sample_id: str, attribute: str, radius: float) -> np.ndarray:

    if radius <= 0.0:
        return np.zeros(2, dtype=np.float64)
    digest = hashlib.blake2b(
        f"{sample_id}:{attribute}".encode("utf-8"), digest_size=16
    ).digest()
    angle_unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    radius_unit = int.from_bytes(digest[8:], "big") / float(2**64 - 1)
    angle = 2.0 * math.pi * angle_unit
    distance = radius * math.sqrt(radius_unit)
    return distance * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)


class Attribute2DSpace:


    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.radius = float(config["prototype_radius"])
        self.residual_scale = float(config["within_level_scale"])
        self.display_jitter = float(config["display_jitter"])
        self.ordinal_target_mode = str(
            config.get("ordinal_target_mode", "continuous_within_class")
        )
        if self.radius <= 0.0:
            raise ValueError("prototype_radius must be positive")
        if self.residual_scale < 0.0 or self.display_jitter < 0.0:
            raise ValueError("within_level_scale and display_jitter must be non-negative")
        if self.ordinal_target_mode not in {
            "continuous_within_class",
            "class_center",
        }:
            raise ValueError(
                "ordinal_target_mode must be 'continuous_within_class' or 'class_center'"
            )
        self.ordinal_source_fields = {
            str(name): str(source)
            for name, source in config.get(
                "ordinal_source_fields", ORDINAL_SOURCE_FIELDS
            ).items()
        }
        if set(self.ordinal_source_fields) != set(ORDINAL_SOURCE_FIELDS):
            raise ValueError(
                "ordinal_source_fields must define exactly the five phenotype attributes"
            )
        configured_cut_points = config.get("ordinal_cut_points")
        self.ordinal_cut_points: dict[str, tuple[float, ...]] = {}
        if configured_cut_points is not None:
            if set(configured_cut_points) != set(ORDINAL_SOURCE_FIELDS):
                raise ValueError(
                    "ordinal_cut_points must define exactly the five phenotype attributes"
                )
            for name, values in configured_cut_points.items():
                cuts = tuple(float(value) for value in values)
                if not 1 <= len(cuts) <= 4:
                    raise ValueError("Every ordinal phenotype must have 1-4 cut points")
                if any(not 1.0 < value < 5.0 for value in cuts) or any(
                    right <= left for left, right in zip(cuts, cuts[1:])
                ):
                    raise ValueError(
                        "Ordinal cut points must be strictly increasing and inside (1, 5)"
                    )
                self.ordinal_cut_points[name] = cuts
            self.ordinal_levels = {
                name: len(values) + 1
                for name, values in self.ordinal_cut_points.items()
            }
        else:
            self.ordinal_levels = {
                name: int(levels) for name, levels in config["ordinal_levels"].items()
            }
            if set(self.ordinal_levels) != set(ORDINAL_SOURCE_FIELDS):
                raise ValueError(
                    "ordinal_levels must define exactly the five phenotype attributes"
                )
            if any(not 2 <= levels <= 5 for levels in self.ordinal_levels.values()):
                raise ValueError("Every ordinal phenotype must use 2-5 levels")

        configured_labels = config.get("ordinal_labels", {})
        self.ordinal_labels: dict[str, tuple[str, ...]] = {}
        for name, levels in self.ordinal_levels.items():
            labels = tuple(
                str(value)
                for value in configured_labels.get(
                    name, [f"Level {level}" for level in range(1, levels + 1)]
                )
            )
            if len(labels) != levels:
                raise ValueError(f"ordinal_labels[{name}] must contain {levels} labels")
            self.ordinal_labels[name] = labels
        self.size_edges = tuple(float(value) for value in config["size_bins_mm"])
        if len(self.size_edges) != 4 or any(
            right <= left for left, right in zip(self.size_edges, self.size_edges[1:])
        ):
            raise ValueError("size_bins_mm must contain four strictly increasing edges")
        if self.size_edges[0] <= 0.0:
            raise ValueError("size_bins_mm values must be positive")
        self.size_labels = tuple(str(value) for value in config["size_labels"])
        if len(self.size_labels) != 3:
            raise ValueError("size_labels must contain three labels")

        self.calcification_column = str(
            config.get("calcification_column", "calcification_group")
        )
        configured_value_groups = config.get("calcification_value_groups")
        self.calcification_value_groups: tuple[tuple[int, ...], ...] | None = None
        if configured_value_groups is not None:
            groups = tuple(
                tuple(int(value) for value in group)
                for group in configured_value_groups
            )
            if len(groups) < 2 or any(not group for group in groups):
                raise ValueError(
                    "calcification_value_groups must contain at least two non-empty groups"
                )
            flattened = [value for group in groups for value in group]
            if len(set(flattened)) != len(flattened):
                raise ValueError(
                    "calcification_value_groups must not contain duplicate source values"
                )
            self.calcification_value_groups = groups
            self.calcification_values = tuple(range(len(groups)))
            calcification_indices = {
                value: index for index, group in enumerate(groups) for value in group
            }
        else:
            self.calcification_values = tuple(
                int(value)
                for value in config.get(
                    "calcification_values", range(len(CALCIFICATION_NAMES))
                )
            )
            calcification_indices = {
                value: index for index, value in enumerate(self.calcification_values)
            }
        self.calcification_labels = tuple(
            str(value)
            for value in config.get(
                "calcification_labels",
                [name.title() for name in CALCIFICATION_NAMES],
            )
        )
        if len(self.calcification_values) < 2:
            raise ValueError("calcification_values must contain at least two classes")
        if len(set(self.calcification_values)) != len(self.calcification_values):
            raise ValueError("calcification_values must not contain duplicates")
        if len(self.calcification_labels) != len(self.calcification_values):
            raise ValueError(
                "calcification_labels must match the number of calcification_values"
            )
        self._calcification_indices = calcification_indices

        self.level_counts = {
            **self.ordinal_levels,
            "calcification": len(self.calcification_values),
            "size": len(self.size_labels),
        }
        self.prototypes = {
            name: semicircle_prototypes(levels, self.radius)
            for name, levels in self.level_counts.items()
        }
        minimum_spacing = min(
            np.linalg.norm(values[1:] - values[:-1], axis=1).min()
            for values in self.prototypes.values()
        )
        if 2.0 * (self.residual_scale + self.display_jitter) >= minimum_spacing:
            raise ValueError("Configured cluster radii overlap adjacent prototypes")
        self.dimension = 2 * len(ATTRIBUTE_NAMES)

    def ordinal_level(self, attribute: str, score: float) -> int:
        if attribute in self.ordinal_cut_points:
            return int(np.searchsorted(self.ordinal_cut_points[attribute], score, side="left")) + 1
        return plan_ordinal_level(score, self.ordinal_levels[attribute])

    def ordinal_within_level(self, attribute: str, score: float, level: int) -> float:
        if self.ordinal_target_mode == "class_center":
            return 0.0
        if attribute in self.ordinal_cut_points:
            bounds = (1.0, *self.ordinal_cut_points[attribute], 5.0)
            return _ordinal_within_bounds(score, bounds[level - 1], bounds[level])
        return _ordinal_within_level(score, level, self.ordinal_levels[attribute])

    def calcification_index(self, value: int) -> int:
        try:
            return self._calcification_indices[int(value)]
        except KeyError as error:
            raise ValueError(
                f"Unsupported calcification value {value}; expected one of "
                f"{self.calcification_values}"
            ) from error

    def _point(
        self,
        *,
        sample_id: str,
        attribute: str,
        source_value: float,
        level: int,
        label: str,
        within_level_position: float,
        measurement_std: float | None = None,
    ) -> AttributePoint:
        prototype = self.prototypes[attribute][level - 1]
        tangent = _grade_tangent(level, self.level_counts[attribute])
        target = prototype + self.residual_scale * within_level_position * tangent
        display = target + _display_offset(sample_id, attribute, self.display_jitter)
        return AttributePoint(
            attribute=attribute,
            source_value=float(source_value),
            level=level,
            label=label,
            target=target.astype(np.float32),
            display=display.astype(np.float32),
            prototype=prototype.astype(np.float32),
            within_level_position=float(within_level_position),
            measurement_std=measurement_std,
        )

    def encode_points(self, row: dict[str, Any]) -> tuple[AttributePoint, ...]:
        sample_id = str(row["sample_id"])
        points: list[AttributePoint] = []
        for attribute, source in self.ordinal_source_fields.items():
            score = float(row[source])
            levels = self.ordinal_levels[attribute]
            level_key = f"{attribute}_level"
            if attribute in self.ordinal_cut_points:
                level = self.ordinal_level(attribute, score)
            else:
                level = (
                    int(row[level_key])
                    if level_key in row and row[level_key] is not None
                    else self.ordinal_level(attribute, score)
                )
            points.append(
                self._point(
                    sample_id=sample_id,
                    attribute=attribute,
                    source_value=score,
                    level=level,
                    label=self.ordinal_labels[attribute][level - 1],
                    within_level_position=self.ordinal_within_level(
                        attribute, score, level
                    ),
                )
            )

        calcification_value = int(row[self.calcification_column])
        calcification = self.calcification_index(calcification_value)
        points.append(
            self._point(
                sample_id=sample_id,
                attribute="calcification",
                source_value=float(calcification_value),
                level=calcification + 1,
                label=self.calcification_labels[calcification],
                within_level_position=0.0,
            )
        )

        diameter = float(row["diameter_mm_mean"])
        if diameter <= self.size_edges[1]:
            size_level = 1
        elif diameter <= self.size_edges[2]:
            size_level = 2
        else:
            size_level = 3
        lower = self.size_edges[size_level - 1]
        upper = self.size_edges[size_level]
        points.append(
            self._point(
                sample_id=sample_id,
                attribute="size",
                source_value=diameter,
                level=size_level,
                label=self.size_labels[size_level - 1],
                within_level_position=_log_within_bin(diameter, lower, upper),
                measurement_std=float(row.get("diameter_mm_std", 0.0)),
            )
        )
        return tuple(points)

    def encode(self, row: dict[str, Any]) -> np.ndarray:
        return np.concatenate([point.target for point in self.encode_points(row)])

    def metadata(self) -> dict[str, Any]:
        blocks = []
        cursor = 0
        for name in ATTRIBUTE_NAMES:
            blocks.append(
                {
                    "name": name,
                    "start": cursor,
                    "end": cursor + 2,
                    "levels": self.level_counts[name],
                }
            )
            cursor += 2
        return {
            "dimension": self.dimension,
            "attribute_names": list(ATTRIBUTE_NAMES),
            "blocks": blocks,
            "prototypes": {
                name: values.tolist() for name, values in self.prototypes.items()
            },
            "ordinal_cut_points": {
                name: list(values) for name, values in self.ordinal_cut_points.items()
            },
            "ordinal_labels": {
                name: list(values) for name, values in self.ordinal_labels.items()
            },
            "ordinal_source_fields": self.ordinal_source_fields,
            "ordinal_target_mode": self.ordinal_target_mode,
            "calcification_column": self.calcification_column,
            "calcification_values": list(self.calcification_values),
            "calcification_value_groups": (
                [list(group) for group in self.calcification_value_groups]
                if self.calcification_value_groups is not None
                else None
            ),
            "calcification_labels": list(self.calcification_labels),
            "target_contract": {
                "prototype_layout": "ordered upper semicircle",
                "within_level_offset": "attribute-specific continuous residual along grade tangent",
                "display_jitter_in_target": False,
                "position_in_target": False,
                "pca_or_umap_applied": False,
            },
            "config": self.config,
        }
