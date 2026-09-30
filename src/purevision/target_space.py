from __future__ import annotations

from typing import Any

from .attribute_2d import Attribute2DSpace
from .attributes import AttributeSpace


def build_target_space(config: dict[str, Any]) -> AttributeSpace | Attribute2DSpace:
    kind = str(config.get("kind", "legacy"))
    if kind == "legacy":
        return AttributeSpace(config)
    if kind == "per_attribute_2d":
        return Attribute2DSpace(config)
    raise ValueError(f"Unsupported attribute-space kind: {kind}")
