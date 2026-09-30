from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return config


def validate_input_contract(config: dict[str, Any]) -> None:
    data = config["data"]
    if int(data["image_size"]) % int(data["patch_size"]):
        raise ValueError("data.image_size must be divisible by data.patch_size")
    if data.get("input_mode", "global") != "lesion_patch_rectangle":
        return
    if data.get("crop_resize") is not False:
        raise ValueError("Patch-aligned lesion crops require data.crop_resize=false")
    if data.get("position_encoding") != "global_patch_ids":
        raise ValueError(
            "Patch-aligned lesion crops must retain position_encoding=global_patch_ids"
        )


def with_path_overrides(
    config: dict[str, Any],
    *,
    processed: str | None = None,
    output: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.setdefault("paths", {})
    for key, value in (("processed", processed), ("output", output), ("model", model)):
        if value is not None:
            result["paths"][key] = value
    return result
