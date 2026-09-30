from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model import load_medgemma_vision_tower

try:
    from .model import instantiate_medgemma_vision_tower
except ImportError:
    def instantiate_medgemma_vision_tower(model_path: str | Path) -> nn.Module:
        from transformers import AutoConfig, SiglipVisionModel

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        return SiglipVisionModel(config.vision_config)


class MedGemmaAnatomyPatchEncoder(nn.Module):


    def __init__(self, vision_encoder: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        tokens = output.last_hidden_state
        return {
            "embedding": tokens,
            "normalized_embedding": torch.nn.functional.normalize(
                tokens.float(), dim=-1
            ),
        }


def build_anatomy_patch_encoder(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> MedGemmaAnatomyPatchEncoder:
    vision = load_medgemma_vision_tower(model_path)
    if gradient_checkpointing:
        vision.gradient_checkpointing_enable()
    return MedGemmaAnatomyPatchEncoder(vision).to(dtype=dtype)


def set_trainable_anatomy_layers(
    model: MedGemmaAnatomyPatchEncoder,
    trainable_layers: int | None,
) -> dict[str, int]:
    vision = model.vision_encoder
    if trainable_layers is None:
        vision.requires_grad_(True)
    else:
        layers = vision.vision_model.encoder.layers
        count = int(trainable_layers)
        if not 1 <= count <= len(layers):
            raise ValueError(f"trainable_layers must be in [1, {len(layers)}]")
        vision.requires_grad_(False)
        for layer in layers[-count:]:
            layer.requires_grad_(True)
        vision.vision_model.post_layernorm.requires_grad_(True)
    return {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_anatomy_patch_encoder(
    model_path: str | Path,
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> MedGemmaAnatomyPatchEncoder:
    model = MedGemmaAnatomyPatchEncoder(
        instantiate_medgemma_vision_tower(model_path)
    ).to(dtype=dtype)
    if Path(checkpoint_path).suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state = checkpoint["model"]
    model.load_state_dict(state, strict=True)
    return model
