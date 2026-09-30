from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TextTargetGroup:
    name: str
    labels: tuple[str, ...]
    texts: tuple[str, ...]
    start: int
    stop: int

    @property
    def classes(self) -> int:
        return self.stop - self.start


class MedGemmaPatchAligner(nn.Module):








    def __init__(self, vision_dimension: int, text_dimension: int, eps: float):
        super().__init__()
        if vision_dimension <= 0 or text_dimension <= 0:
            raise ValueError("alignment dimensions must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.vision_dimension = int(vision_dimension)
        self.text_dimension = int(text_dimension)
        self.eps = float(eps)
        self.norm_weight = nn.Parameter(torch.zeros(self.vision_dimension))
        self.projection_weight = nn.Parameter(
            torch.empty(self.vision_dimension, self.text_dimension)
        )
        nn.init.normal_(self.projection_weight, mean=0.0, std=0.02)

    def forward(self, vision_embedding: torch.Tensor) -> torch.Tensor:
        if vision_embedding.shape[-1] != self.vision_dimension:
            raise ValueError(
                f"Expected {self.vision_dimension}D vision embeddings, got "
                f"{vision_embedding.shape[-1]}D"
            )
        values = vision_embedding.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        normalized = normalized * (1.0 + self.norm_weight.float())
        projected = normalized @ self.projection_weight.float()
        return projected.to(dtype=vision_embedding.dtype)

    def load_native_state(
        self, *, norm_weight: torch.Tensor, projection_weight: torch.Tensor
    ) -> None:
        if tuple(norm_weight.shape) != tuple(self.norm_weight.shape):
            raise ValueError(
                f"Native norm shape mismatch: {tuple(norm_weight.shape)} vs "
                f"{tuple(self.norm_weight.shape)}"
            )
        if tuple(projection_weight.shape) != tuple(self.projection_weight.shape):
            raise ValueError(
                "Native projection shape mismatch: "
                f"{tuple(projection_weight.shape)} vs "
                f"{tuple(self.projection_weight.shape)}"
            )
        with torch.no_grad():
            self.norm_weight.copy_(norm_weight)
            self.projection_weight.copy_(projection_weight)


class AlignmentTargetBank(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        groups: Sequence[TextTargetGroup],
    ) -> None:
        super().__init__()
        if embeddings.ndim != 2:
            raise ValueError("text target embeddings must have shape [targets, dimension]")
        groups = tuple(groups)
        if not groups:
            raise ValueError("at least one text target group is required")
        cursor = 0
        for group in groups:
            if group.start != cursor or group.stop <= group.start:
                raise ValueError("text target groups must be contiguous and non-empty")
            if len(group.labels) != group.classes or len(group.texts) != group.classes:
                raise ValueError(f"Malformed text target group: {group.name}")
            cursor = group.stop
        if cursor != embeddings.shape[0]:
            raise ValueError(
                f"Target group size {cursor} does not match embeddings {embeddings.shape[0]}"
            )
        self.register_buffer("embeddings", F.normalize(embeddings.float(), dim=-1))
        self.groups = groups
        self._by_name = {group.name: group for group in groups}
        if len(self._by_name) != len(groups):
            raise ValueError("text target group names must be unique")

    def group(self, name: str) -> tuple[torch.Tensor, TextTargetGroup]:
        try:
            group = self._by_name[name]
        except KeyError as error:
            raise KeyError(f"Unknown text target group: {name}") from error
        return self.embeddings[group.start : group.stop], group

    def metadata(self) -> dict[str, Any]:
        return {
            "dimension": int(self.embeddings.shape[1]),
            "pooling": "last_non_padding_token",
            "groups": [
                {
                    "name": group.name,
                    "labels": list(group.labels),
                    "texts": list(group.texts),
                    "start": group.start,
                    "stop": group.stop,
                }
                for group in self.groups
            ],
        }


def build_target_groups(
    target_config: Mapping[str, Sequence[Mapping[str, str]]],
) -> tuple[tuple[TextTargetGroup, ...], tuple[str, ...]]:
    groups: list[TextTargetGroup] = []
    all_texts: list[str] = []
    cursor = 0
    for name, entries in target_config.items():
        labels: list[str] = []
        texts: list[str] = []
        for entry in entries:
            label = str(entry["label"])
            text = str(entry["text"])
            if not label or not text:
                raise ValueError(f"Empty label/text in target group {name}")
            labels.append(label)
            texts.append(text)
        if len(labels) < 2:
            raise ValueError(f"Target group {name} must contain at least two classes")
        if len(set(labels)) != len(labels):
            raise ValueError(f"Target group {name} contains duplicate labels")
        stop = cursor + len(labels)
        groups.append(
            TextTargetGroup(
                name=str(name),
                labels=tuple(labels),
                texts=tuple(texts),
                start=cursor,
                stop=stop,
            )
        )
        all_texts.extend(texts)
        cursor = stop
    return tuple(groups), tuple(all_texts)


def cosine_logits(
    aligned_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if aligned_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("cosine logits expect two matrices")
    if aligned_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("vision/text alignment dimensions do not match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    vision = F.normalize(aligned_embeddings.float(), dim=-1)
    text = F.normalize(text_embeddings.float(), dim=-1)
    return (vision @ text.transpose(0, 1)) / float(temperature)


def group_alignment_loss(
    aligned_embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_bank: AlignmentTargetBank,
    group_name: str,
    *,
    temperature: float,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets, group = target_bank.group(group_name)
    if labels.ndim != 1 or labels.shape[0] != aligned_embeddings.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= group.classes):
        raise ValueError(f"Labels outside target group {group_name}")
    logits = cosine_logits(aligned_embeddings, targets, temperature=temperature)
    loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
    return loss, logits


def group_matching_cosine_loss(
    aligned_embeddings: torch.Tensor,
    labels: torch.Tensor,
    target_bank: AlignmentTargetBank,
    group_name: str,
    *,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets, group = target_bank.group(group_name)
    if labels.ndim != 1 or labels.shape[0] != aligned_embeddings.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= group.classes):
        raise ValueError(f"Labels outside target group {group_name}")
    similarities = F.normalize(aligned_embeddings.float(), dim=-1) @ targets.T
    matching = similarities.gather(1, labels.long().unsqueeze(1)).squeeze(1)
    distances = 1.0 - matching
    if class_weights is None:
        loss = distances.mean()
    else:
        if class_weights.ndim != 1 or class_weights.shape[0] != group.classes:
            raise ValueError(f"Class weights do not match target group {group_name}")
        sample_weights = class_weights[labels.long()]
        loss = (distances * sample_weights).sum() / sample_weights.sum()
    return loss, matching


def pathology_alignment_loss(
    aligned_embeddings: torch.Tensor,
    attribute_labels: torch.Tensor,
    target_bank: AlignmentTargetBank,
    attribute_names: Sequence[str],
    *,
    temperature: float,
    attribute_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if attribute_labels.ndim != 2:
        raise ValueError("attribute_labels must have shape [batch, attributes]")
    names = tuple(str(name) for name in attribute_names)
    if attribute_labels.shape != (aligned_embeddings.shape[0], len(names)):
        raise ValueError("attribute label matrix does not match embeddings/attribute names")
    weights = attribute_weights or {}
    per_class_weights = class_weights or {}
    losses: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    denominator = 0.0
    for index, name in enumerate(names):
        labels = attribute_labels[:, index]
        valid = (
            torch.ones_like(labels, dtype=torch.bool)
            if ignore_index is None
            else labels.ne(ignore_index)
        )
        if not bool(valid.any()):
            continue
        loss, _ = group_alignment_loss(
            aligned_embeddings[valid],
            labels[valid],
            target_bank,
            name,
            temperature=temperature,
            class_weights=per_class_weights.get(name),
        )
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError(f"Negative attribute weight for {name}")
        losses[name] = loss
        if weight:
            weighted.append(weight * loss)
            denominator += weight
    if not weighted:
        raise ValueError("At least one pathology attribute weight must be positive")
    return torch.stack(weighted).sum() / denominator, losses


def pathology_matching_cosine_loss(
    aligned_embeddings: torch.Tensor,
    attribute_labels: torch.Tensor,
    target_bank: AlignmentTargetBank,
    attribute_names: Sequence[str],
    *,
    attribute_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if attribute_labels.ndim != 2:
        raise ValueError("attribute_labels must have shape [batch, attributes]")
    names = tuple(str(name) for name in attribute_names)
    if attribute_labels.shape != (aligned_embeddings.shape[0], len(names)):
        raise ValueError("attribute label matrix does not match embeddings/attribute names")
    weights = attribute_weights or {}
    per_class_weights = class_weights or {}
    losses: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    denominator = 0.0
    for index, name in enumerate(names):
        labels = attribute_labels[:, index]
        valid = (
            torch.ones_like(labels, dtype=torch.bool)
            if ignore_index is None
            else labels.ne(ignore_index)
        )
        if not bool(valid.any()):
            continue
        loss, _ = group_matching_cosine_loss(
            aligned_embeddings[valid],
            labels[valid],
            target_bank,
            name,
            class_weights=per_class_weights.get(name),
        )
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError(f"Negative attribute weight for {name}")
        losses[name] = loss
        if weight:
            weighted.append(weight * loss)
            denominator += weight
    if not weighted:
        raise ValueError("At least one pathology attribute weight must be positive")
    return torch.stack(weighted).sum() / denominator, losses


def similarity_matrix(
    aligned_embeddings: torch.Tensor, target_bank: AlignmentTargetBank
) -> torch.Tensor:
    return F.normalize(aligned_embeddings.float(), dim=-1) @ target_bank.embeddings.T


def _checkpoint_tensor_names(model_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model_path / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"], payload


def _find_checkpoint_key(weight_map: Mapping[str, str], suffix: str) -> str:
    matches = [key for key in weight_map if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint tensor ending in {suffix!r}, got {matches}"
        )
    return matches[0]


def load_native_patch_aligner(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> MedGemmaPatchAligner:

    from safetensors import safe_open
    from transformers import AutoConfig

    root = Path(model_path)
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    weight_map, _ = _checkpoint_tensor_names(root)
    norm_key = _find_checkpoint_key(
        weight_map, "multi_modal_projector.mm_soft_emb_norm.weight"
    )
    projection_key = _find_checkpoint_key(
        weight_map, "multi_modal_projector.mm_input_projection_weight"
    )
    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in (norm_key, projection_key):
        by_shard[weight_map[key]].append(key)
    for shard, keys in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)

    aligner = MedGemmaPatchAligner(
        vision_dimension=int(config.vision_config.hidden_size),
        text_dimension=int(config.text_config.hidden_size),
        eps=float(config.vision_config.layer_norm_eps),
    )
    aligner.load_native_state(
        norm_weight=tensors[norm_key],
        projection_weight=tensors[projection_key],
    )
    return aligner.to(dtype=dtype)


def load_trained_patch_aligner(
    model_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MedGemmaPatchAligner:

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint.get("aligner") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Alignment checkpoint does not contain an 'aligner' state")
    aligner = load_native_patch_aligner(model_path, dtype=dtype)
    aligner.load_state_dict(state, strict=True)
    return aligner.to(device=device, dtype=dtype).eval()
