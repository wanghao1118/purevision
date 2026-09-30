from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from collections.abc import Sequence

import torch
from safetensors import safe_open
from torch import nn


def instantiate_medgemma_vision_tower(model_path: str | Path) -> nn.Module:

    from transformers import AutoConfig, SiglipVisionModel

    model_path = Path(model_path)
    full_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return SiglipVisionModel(full_config.vision_config)


def load_medgemma_vision_tower(model_path: str | Path) -> nn.Module:

    model_path = Path(model_path)
    vision = instantiate_medgemma_vision_tower(model_path)

    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in index["weight_map"].items():
        if key.startswith("vision_tower."):
            by_shard[shard].append(key)
    if not by_shard:
        raise ValueError(f"No vision_tower tensors found under {model_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key.removeprefix("vision_tower.")] = handle.get_tensor(key)
    incompatible = vision.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Vision checkpoint mismatch: {incompatible}")
    return vision


def mask_average_pool(tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or patch_mask.ndim != 2:
        raise ValueError("Expected tokens [B,N,D] and patch_mask [B,N]")
    if tokens.shape[:2] != patch_mask.shape:
        raise ValueError(f"Token/mask mismatch: {tokens.shape} vs {patch_mask.shape}")
    weights = patch_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


class MedGemmaLesionEncoder(nn.Module):
    def __init__(self, vision_encoder: nn.Module, target_dimension: int):
        super().__init__()
        self.vision_encoder = vision_encoder
        hidden_size = int(vision_encoder.config.hidden_size)
        self.target_projection = nn.Linear(hidden_size, target_dimension)

    def forward(
        self,
        pixel_values: torch.Tensor | Sequence[torch.Tensor],
        patch_mask: torch.Tensor | Sequence[torch.Tensor],
        global_position_ids: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(pixel_values, torch.Tensor):
            output = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
            tokens = output.last_hidden_state
            if not isinstance(patch_mask, torch.Tensor):
                raise TypeError("Fixed-size images require a tensor patch_mask")
        else:
            if global_position_ids is None or isinstance(global_position_ids, torch.Tensor):
                raise TypeError("Variable crops require a sequence of global position IDs")
            if isinstance(patch_mask, torch.Tensor):
                raise TypeError("Variable crops require a sequence of patch masks")
            tokens, patch_mask = self._encode_variable_crops(
                pixel_values, patch_mask, global_position_ids
            )
        lesion_embedding = mask_average_pool(tokens, patch_mask)
        return {
            "embedding": lesion_embedding,
            "normalized_embedding": torch.nn.functional.normalize(lesion_embedding.float(), dim=-1),
            "coordinate": self.target_projection(lesion_embedding),
        }

    def _encode_variable_crops(
        self,
        pixel_values: Sequence[torch.Tensor],
        patch_masks: Sequence[torch.Tensor],
        global_position_ids: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (len(pixel_values) == len(patch_masks) == len(global_position_ids)):
            raise ValueError("Variable crop batch fields must have equal lengths")
        vision_model = self.vision_encoder.vision_model
        embeddings_module = vision_model.embeddings
        token_sequences: list[torch.Tensor] = []
        for pixels, mask, position_ids in zip(
            pixel_values, patch_masks, global_position_ids
        ):
            patch_tokens = embeddings_module.patch_embedding(
                pixels.unsqueeze(0).to(dtype=embeddings_module.patch_embedding.weight.dtype)
            ).flatten(2).transpose(1, 2)[0]
            if patch_tokens.shape[0] != mask.numel():
                raise ValueError(
                    f"Patch projection/mask mismatch: {patch_tokens.shape[0]} vs {mask.numel()}"
                )
            if position_ids.numel() != patch_tokens.shape[0]:
                raise ValueError("Each crop token must retain one global position ID")
            token_sequences.append(
                patch_tokens + embeddings_module.position_embedding(position_ids)
            )

        lengths = torch.tensor(
            [tokens.shape[0] for tokens in token_sequences],
            device=token_sequences[0].device,
        )
        padded_tokens = nn.utils.rnn.pad_sequence(token_sequences, batch_first=True)
        padded_masks = nn.utils.rnn.pad_sequence(
            [mask.to(dtype=padded_tokens.dtype) for mask in patch_masks],
            batch_first=True,
        )
        valid_keys = (
            torch.arange(padded_tokens.shape[1], device=padded_tokens.device)[None, :]
            < lengths[:, None]
        )
        attention_mask = torch.zeros(
            (len(token_sequences), 1, 1, padded_tokens.shape[1]),
            dtype=padded_tokens.dtype,
            device=padded_tokens.device,
        )
        attention_mask.masked_fill_(
            ~valid_keys[:, None, None, :], torch.finfo(padded_tokens.dtype).min
        )
        encoded = vision_model.encoder(
            inputs_embeds=padded_tokens, attention_mask=attention_mask
        ).last_hidden_state
        return vision_model.post_layernorm(encoded), padded_masks


def set_trainable_vision_layers(
    model: MedGemmaLesionEncoder, trainable_layers: int | None
) -> dict[str, int]:

    if trainable_layers is not None:
        layers = model.vision_encoder.vision_model.encoder.layers
        trainable_layers = int(trainable_layers)
        if not 0 <= trainable_layers <= len(layers):
            raise ValueError(
                f"trainable_vision_layers must be in [0, {len(layers)}], "
                f"got {trainable_layers}"
            )
        model.vision_encoder.requires_grad_(False)
        for layer in layers[len(layers) - trainable_layers :]:
            layer.requires_grad_(True)
        model.vision_encoder.vision_model.post_layernorm.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def build_model(
    model_path: str | Path,
    target_dimension: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> MedGemmaLesionEncoder:
    vision = load_medgemma_vision_tower(model_path)
    if gradient_checkpointing:
        vision.gradient_checkpointing_enable()
    model = MedGemmaLesionEncoder(vision, target_dimension)
    return model.to(dtype=dtype)


def load_trained_model(
    model_path: str | Path,
    checkpoint_path: str | Path,
    target_dimension: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> MedGemmaLesionEncoder:
    model = build_model(
        model_path, target_dimension, dtype=dtype, gradient_checkpointing=False
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
