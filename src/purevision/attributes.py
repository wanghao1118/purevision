from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOCATION_NAMES = (
    "left_upper",
    "left_lower",
    "right_upper",
    "right_middle",
    "right_lower",
)

ORDINAL_SOURCE_FIELDS = {
    "density": "texture_mean",
    "sphericity": "sphericity_mean",
    "margin": "margin_mean",
    "lobulation": "lobulation_mean",
    "spiculation": "spiculation_mean",
}

CALCIFICATION_NAMES = (
    "absent",
    "present",
)


def plan_ordinal_level(score: float, levels: int) -> int:

    if not 2 <= levels <= 5:
        raise ValueError(f"Ordinal level count must be in [2, 5], got {levels}")
    score = float(np.clip(score, 1.0, 5.0))
    if score >= 5.0:
        return levels
    return min(levels, math.floor((score - 1.0) * levels / 4.0) + 1)


def calcification_group(code: int) -> int:

    code = int(code)
    if code == 6:
        return 0
    if code in (1, 2, 3, 4, 5):
        return 1
    raise ValueError(f"Unexpected LIDC calcification code: {code}")


def majority_calcification_group(codes: Iterable[int]) -> int:
    grouped = [calcification_group(int(code)) for code in codes]
    if not grouped:
        raise ValueError("At least one calcification score is required")
    counts = np.bincount(grouped, minlength=len(CALCIFICATION_NAMES))
    winners = np.flatnonzero(counts == counts.max())

    return int(0 if 0 in winners else winners[0])


def _simplex_one_hot(index: int, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=np.float32)
    result[index] = 1.0 / math.sqrt(2.0)
    return result


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int
    weight: float
    kind: str


class AttributeSpace:


    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.include_location = bool(config.get("include_location", True))
        self.ordinal_levels = {
            name: int(levels)
            for name, levels in config["ordinal_levels"].items()
        }
        self.ordinal_weights = {
            name: float(weight)
            for name, weight in config["ordinal_weights"].items()
        }
        if set(self.ordinal_levels) != set(ORDINAL_SOURCE_FIELDS):
            raise ValueError("ordinal_levels must define exactly the five phenotype axes")
        for levels in self.ordinal_levels.values():
            if not 2 <= levels <= 5:
                raise ValueError("Every ordinal phenotype must use 2-5 levels")
        if int(config["calcification_levels"]) != 2:
            raise ValueError("The v1 calcification mapping has exactly two groups")

        blocks: list[Block] = []
        cursor = 0
        if self.include_location:
            blocks.append(
                Block(
                    "location",
                    cursor,
                    cursor + len(LOCATION_NAMES),
                    float(config["location_weight"]),
                    "categorical",
                )
            )
            cursor += len(LOCATION_NAMES)
        for name in ORDINAL_SOURCE_FIELDS:
            blocks.append(Block(name, cursor, cursor + 1, self.ordinal_weights[name], "ordinal"))
            cursor += 1
        blocks.append(
            Block("calcification", cursor, cursor + len(CALCIFICATION_NAMES), float(config["calcification_weight"]), "categorical")
        )
        cursor += len(CALCIFICATION_NAMES)
        blocks.append(Block("size", cursor, cursor + 1, float(config["size_weight"]), "continuous"))
        cursor += 1
        self.blocks = tuple(blocks)
        self.dimension = cursor

    def encode(self, row: dict[str, Any]) -> np.ndarray:
        pieces: list[np.ndarray] = []
        if self.include_location:
            location = str(row["position"])
            pieces.append(
                _simplex_one_hot(LOCATION_NAMES.index(location), len(LOCATION_NAMES))
                * float(self.config["location_weight"])
            )

        for name, source in ORDINAL_SOURCE_FIELDS.items():
            levels = self.ordinal_levels[name]
            level_key = f"{name}_level"
            level = int(row[level_key]) if level_key in row else plan_ordinal_level(float(row[source]), levels)
            coordinate = 0.0 if levels == 1 else (level - 1.0) / (levels - 1.0)
            pieces.append(np.asarray([coordinate * self.ordinal_weights[name]], dtype=np.float32))

        calcification = int(row["calcification_group"])
        pieces.append(
            _simplex_one_hot(calcification, len(CALCIFICATION_NAMES))
            * float(self.config["calcification_weight"])
        )

        diameter = float(row["diameter_mm_mean"])
        lower = float(self.config["size_min_mm"])
        upper = float(self.config["size_max_mm"])
        diameter = max(diameter, lower)
        size_coordinate = (math.log(diameter) - math.log(lower)) / (math.log(upper) - math.log(lower))
        pieces.append(np.asarray([size_coordinate * float(self.config["size_weight"])], dtype=np.float32))
        return np.concatenate(pieces)

    def enrich_row(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        for name, source in ORDINAL_SOURCE_FIELDS.items():
            result[f"{name}_level"] = plan_ordinal_level(
                float(result[source]), self.ordinal_levels[name]
            )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "include_location": self.include_location,
            "locations": list(LOCATION_NAMES),
            "calcification_groups": list(CALCIFICATION_NAMES),
            "ordinal_levels": self.ordinal_levels,
            "blocks": [block.__dict__ for block in self.blocks],
            "config": self.config,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
