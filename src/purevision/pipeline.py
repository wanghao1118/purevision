from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .alignment import (
    AlignmentTargetBank,
    TextTargetGroup,
    load_trained_patch_aligner,
)
from .anatomy_model import load_anatomy_patch_encoder
from .decoder import FrozenMedGemmaDecoder
from .model import load_trained_model
from .semantic_fusion import (
    FusionResult,
    cosine_similarity_matrix,
    fuse_semantic_groups,
    lesion_margin,
    select_local_lesion_region,
)


@dataclass(frozen=True)
class PureVisionOutput:
    text: str
    fusion: FusionResult
    phenotype_patch_embeddings: torch.Tensor
    anatomy_patch_embeddings: torch.Tensor

    def semantic_probabilities(self) -> dict[str, dict[str, float]]:
        return {
            group.name: {
                label: float(probability)
                for label, probability in zip(
                    group.labels, group.probabilities.detach().cpu()
                )
            }
            for group in self.fusion.groups
        }


def load_alignment_target_bank(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> AlignmentTargetBank:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    embeddings = checkpoint.get("text_target_embeddings")
    metadata = checkpoint.get("text_target_metadata")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(metadata, dict):
        raise ValueError("alignment checkpoint is missing its frozen text target bank")
    groups = tuple(
        TextTargetGroup(
            name=str(group["name"]),
            labels=tuple(str(value) for value in group["labels"]),
            texts=tuple(str(value) for value in group["texts"]),
            start=int(group["start"]),
            stop=int(group["stop"]),
        )
        for group in metadata["groups"]
    )
    return AlignmentTargetBank(embeddings, groups).to(device)


class PureVisionPipeline:


    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        paths = config["paths"]
        runtime = config.get("runtime", {})
        inference = config["inference"]
        if not bool(inference.get("native_visual_tokens_retained", False)):
            raise ValueError("the paper requires retention of native visual tokens")
        if bool(inference.get("inference_masks", False)):
            raise ValueError("training masks must never be supplied during inference")
        if bool(inference.get("lora", False)):
            raise ValueError("the paper uses a frozen decoder without LoRA")

        self.device = torch.device(runtime.get("device", "cuda:0"))
        dtype_name = str(runtime.get("dtype", "bfloat16"))
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported runtime dtype: {dtype_name}")
        self.dtype = getattr(torch, dtype_name)
        model_path = Path(paths["model"])
        self.phenotype_encoder = load_trained_model(
            model_path,
            paths["phenotype_checkpoint"],
            int(inference.get("phenotype_target_dimension", 14)),
            dtype=self.dtype,
        ).to(self.device)
        self.anatomy_encoder = load_anatomy_patch_encoder(
            model_path,
            paths["anatomy_checkpoint"],
            dtype=self.dtype,
        ).to(self.device)
        self.aligner = load_trained_patch_aligner(
            model_path,
            paths["alignment_checkpoint"],
            device=self.device,
            dtype=self.dtype,
        )
        self.target_bank = load_alignment_target_bank(
            paths["alignment_checkpoint"], device=self.device
        )
        self.decoder = FrozenMedGemmaDecoder(
            model_path,
            device=self.device,
            dtype=self.dtype,
            attention_implementation=str(
                runtime.get("attention_implementation", "sdpa")
            ),
        )
        for module in (
            self.phenotype_encoder,
            self.anatomy_encoder,
            self.aligner,
            self.target_bank,
        ):
            module.requires_grad_(False).eval()

        self.grid_size = int(inference.get("grid_size", 64))
        self.window_size = int(inference.get("window_size", 5))
        self.anchor_top_k = int(inference.get("anchor_top_k", 2))
        self.patch_count = int(inference.get("selected_patches", 8))
        self.semantic_temperature = float(
            inference.get("semantic_temperature", 0.125)
        )
        self.anatomy_group = str(inference.get("anatomy_group", "anatomy"))
        self.phenotype_groups = tuple(
            str(value) for value in inference["phenotype_groups"]
        )
        self.lesion_labels = tuple(
            str(value) for value in inference["lesion_anatomy_labels"]
        )

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype in {
            torch.bfloat16,
            torch.float16,
        }:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.inference_mode()
    def _aligned_patch_embeddings(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast():
            phenotype = self.phenotype_encoder.vision_encoder(
                pixel_values=pixel_values, return_dict=True
            ).last_hidden_state[0]
            anatomy = self.anatomy_encoder(pixel_values)["embedding"][0]
            aligned_phenotype = self.aligner(phenotype)
            aligned_anatomy = self.aligner(anatomy)
        expected_patches = self.grid_size * self.grid_size
        if phenotype.shape[0] != expected_patches or anatomy.shape[0] != expected_patches:
            raise RuntimeError(
                "visual encoders did not produce the configured spatial patch grid"
            )
        return phenotype, anatomy, aligned_phenotype, aligned_anatomy

    def _group_similarity(
        self, aligned: torch.Tensor, group_name: str
    ) -> tuple[torch.Tensor, TextTargetGroup]:
        targets, group = self.target_bank.group(group_name)
        return cosine_similarity_matrix(aligned, targets), group

    @torch.inference_mode()
    def run(
        self,
        image: Any,
        instruction: str,
        *,
        max_new_tokens: int = 384,
    ) -> PureVisionOutput:
        pixel_values = self.decoder.preprocess_image(image)
        (
            phenotype,
            anatomy,
            aligned_phenotype,
            aligned_anatomy,
        ) = self._aligned_patch_embeddings(pixel_values)
        anatomy_similarity, anatomy_targets = self._group_similarity(
            aligned_anatomy, self.anatomy_group
        )
        label_to_index = {
            label: index for index, label in enumerate(anatomy_targets.labels)
        }
        missing = [label for label in self.lesion_labels if label not in label_to_index]
        if missing:
            raise ValueError(f"lesion labels absent from anatomy target bank: {missing}")
        lesion_indices = [label_to_index[label] for label in self.lesion_labels]
        other_indices = [
            index
            for index in range(anatomy_targets.classes)
            if index not in lesion_indices
        ]
        margins = lesion_margin(
            anatomy_similarity, lesion_indices, other_indices
        )
        selection = select_local_lesion_region(
            margins,
            grid_size=self.grid_size,
            window_size=self.window_size,
            anchor_top_k=self.anchor_top_k,
            patch_count=self.patch_count,
        )

        similarities: OrderedDict[str, torch.Tensor] = OrderedDict()
        labels: OrderedDict[str, Sequence[str]] = OrderedDict()
        sequences: OrderedDict[str, Sequence[torch.Tensor]] = OrderedDict()
        similarities[self.anatomy_group] = anatomy_similarity
        labels[self.anatomy_group] = anatomy_targets.labels
        sequences[self.anatomy_group] = self.decoder.candidate_token_sequences(
            anatomy_targets.texts
        )
        for group_name in self.phenotype_groups:
            group_similarity, group = self._group_similarity(
                aligned_phenotype, group_name
            )
            similarities[group_name] = group_similarity
            labels[group_name] = group.labels
            sequences[group_name] = self.decoder.candidate_token_sequences(group.texts)

        fusion = fuse_semantic_groups(
            selection,
            similarities,
            labels,
            sequences,
            semantic_temperature=self.semantic_temperature,
        )
        text = self.decoder.generate(
            pixel_values,
            fusion.soft_tokens(),
            center_row=selection.center_row,
            center_column=selection.center_column,
            grid_size=self.grid_size,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
        )
        return PureVisionOutput(
            text=text,
            fusion=fusion,
            phenotype_patch_embeddings=phenotype.detach().cpu(),
            anatomy_patch_embeddings=anatomy.detach().cpu(),
        )
