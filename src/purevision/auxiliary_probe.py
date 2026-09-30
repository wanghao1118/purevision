from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attribute_2d import ATTRIBUTE_NAMES


class AuxiliaryAttributeProbe(nn.Module):


    def __init__(
        self,
        input_dimension: int,
        classes: int,
        *,
        dropout: float = 0.0,
        first_layer_initialization: str = "default",
    ):
        super().__init__()
        self.first_layer = nn.Linear(input_dimension, input_dimension)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(input_dimension, classes)
        if first_layer_initialization == "identity":
            nn.init.eye_(self.first_layer.weight)
            nn.init.zeros_(self.first_layer.bias)
        elif first_layer_initialization != "default":
            raise ValueError(
                "first_layer_initialization must be 'default' or 'identity'"
            )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.first_layer(embedding))
        return hidden, self.classifier(self.dropout(hidden))

    def geometry_hidden(
        self, embedding: torch.Tensor, *, update_probe: bool
    ) -> torch.Tensor:
        if update_probe:
            return self.activation(self.first_layer(embedding))
        return self.activation(
            F.linear(
                embedding,
                self.first_layer.weight.detach(),
                None
                if self.first_layer.bias is None
                else self.first_layer.bias.detach(),
            )
        )


class AuxiliaryProbeBank(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        class_counts: dict[str, int],
        *,
        dropout: float = 0.0,
        first_layer_initialization: str = "default",
    ):
        super().__init__()
        if tuple(class_counts) != ATTRIBUTE_NAMES:
            raise ValueError("class_counts must follow ATTRIBUTE_NAMES order")
        self.probes = nn.ModuleDict(
            {
                name: AuxiliaryAttributeProbe(
                    input_dimension,
                    class_counts[name],
                    dropout=dropout,
                    first_layer_initialization=first_layer_initialization,
                )
                for name in ATTRIBUTE_NAMES
            }
        )

    def forward(
        self, embedding: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {name: self.probes[name](embedding) for name in ATTRIBUTE_NAMES}

    def geometry_hidden(
        self, embedding: torch.Tensor, *, update_probe: bool
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.probes[name].geometry_hidden(
                embedding, update_probe=update_probe
            )
            for name in ATTRIBUTE_NAMES
        }


class EncoderWithAuxiliaryProbes(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        probes: AuxiliaryProbeBank,
        *,
        geometry_updates_probe: bool = True,
        classification_updates_encoder: bool = True,
        geometry_uses_embedding: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.probes = probes
        self.geometry_updates_probe = bool(geometry_updates_probe)
        self.classification_updates_encoder = bool(classification_updates_encoder)
        self.geometry_uses_embedding = bool(geometry_uses_embedding)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        outputs = self.encoder(*args, **kwargs)
        classification_embedding = (
            outputs["embedding"]
            if self.classification_updates_encoder
            else outputs["embedding"].detach()
        )
        outputs["auxiliary_probes"] = self.probes(classification_embedding)
        if self.geometry_uses_embedding:
            outputs["auxiliary_geometry_hidden"] = {
                name: outputs["embedding"] for name in ATTRIBUTE_NAMES
            }
        elif self.geometry_updates_probe:
            outputs["auxiliary_geometry_hidden"] = {
                name: hidden
                for name, (hidden, _) in outputs["auxiliary_probes"].items()
            }
        else:
            outputs["auxiliary_geometry_hidden"] = self.probes.geometry_hidden(
                outputs["embedding"], update_probe=False
            )
        return outputs


@dataclass(frozen=True)
class PCAProjection:
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: torch.Tensor
    alignment_rotation: torch.Tensor
    alignment_scale: torch.Tensor
    alignment_offset: torch.Tensor

    def project(self, values: torch.Tensor) -> torch.Tensor:
        coordinates = (values.float() - self.mean) @ self.components.T
        return (
            self.alignment_scale * (coordinates @ self.alignment_rotation)
            + self.alignment_offset
        )

    def to(self, device: torch.device) -> PCAProjection:
        return PCAProjection(
            mean=self.mean.to(device),
            components=self.components.to(device),
            explained_variance_ratio=self.explained_variance_ratio.to(device),
            alignment_rotation=self.alignment_rotation.to(device),
            alignment_scale=self.alignment_scale.to(device),
            alignment_offset=self.alignment_offset.to(device),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "explained_variance_ratio": self.explained_variance_ratio.detach().cpu(),
            "alignment_rotation": self.alignment_rotation.detach().cpu(),
            "alignment_scale": self.alignment_scale.detach().cpu(),
            "alignment_offset": self.alignment_offset.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> PCAProjection:
        device = state["mean"].device
        return cls(
            mean=state["mean"].float(),
            components=state["components"].float(),
            explained_variance_ratio=state["explained_variance_ratio"].float(),
            alignment_rotation=state.get(
                "alignment_rotation", torch.eye(2, device=device)
            ).float(),
            alignment_scale=state.get(
                "alignment_scale", torch.ones((), device=device)
            ).float(),
            alignment_offset=state.get(
                "alignment_offset", torch.zeros(2, device=device)
            ).float(),
        )


def _fit_similarity_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = sample_weights.float().clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-12)
    source_mean = (source * weights[:, None]).sum(dim=0)
    target_mean = (target * weights[:, None]).sum(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered * weights[:, None]).T @ target_centered
    left, singular_values, right_transpose = torch.linalg.svd(covariance)
    rotation = left @ right_transpose
    source_energy = (
        weights * source_centered.square().sum(dim=1)
    ).sum().clamp_min(1e-12)
    scale = singular_values.sum() / source_energy
    offset = target_mean - scale * (source_mean @ rotation)
    return rotation.detach(), scale.detach(), offset.detach()


def fit_pca_projection(
    values: torch.Tensor,
    *,
    seed: int,
    alignment_targets: torch.Tensor | None = None,
    alignment_weights: torch.Tensor | None = None,
    pca_weights: torch.Tensor | None = None,
    approximation_rank: int = 2,
    niter: int = 4,
) -> PCAProjection:
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("PCA requires a [samples, features] tensor with both dimensions >= 2")
    values = values.float()
    if int(approximation_rank) < 2:
        raise ValueError("PCA approximation_rank must be at least 2")
    if int(niter) < 1:
        raise ValueError("PCA niter must be at least 1")
    decomposition_values: torch.Tensor
    if pca_weights is None:
        mean = values.mean(dim=0)
        centered = values - mean
        decomposition_values = centered
    else:
        if pca_weights.shape != (len(values),):
            raise ValueError("PCA weights must have shape [samples]")
        normalized_weights = pca_weights.to(values.device, torch.float32).clamp_min(0.0)
        normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(
            1e-12
        )
        mean = (values * normalized_weights[:, None]).sum(dim=0)
        centered = values - mean
        decomposition_values = centered * torch.sqrt(
            normalized_weights * len(values)
        )[:, None]
    devices = [values.device.index] if values.is_cuda and values.device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        rank = min(int(approximation_rank), *decomposition_values.shape)
        _, singular_values, vectors = torch.pca_lowrank(
            decomposition_values, q=rank, center=False, niter=int(niter)
        )
    components = vectors[:, :2].T.contiguous()
    for index in range(2):
        loading = components[index]
        pivot = loading[loading.abs().argmax()]
        components[index] = loading * torch.where(pivot < 0, -1.0, 1.0)
    total_variance = decomposition_values.square().sum().clamp_min(1e-12)
    explained = singular_values[:2].square() / total_variance
    rotation = torch.eye(2, device=values.device)
    scale = torch.ones((), device=values.device)
    offset = torch.zeros(2, device=values.device)
    if alignment_targets is not None:
        if alignment_targets.shape != (len(values), 2):
            raise ValueError("PCA alignment targets must have shape [samples, 2]")
        if alignment_weights is None:
            alignment_weights = torch.ones(len(values), device=values.device)
        if alignment_weights.shape != (len(values),):
            raise ValueError("PCA alignment weights must have shape [samples]")
        source = centered @ components.T
        rotation, scale, offset = _fit_similarity_alignment(
            source,
            alignment_targets.to(values.device, torch.float32),
            alignment_weights.to(values.device, torch.float32),
        )
    return PCAProjection(
        mean=mean.detach(),
        components=components.detach(),
        explained_variance_ratio=explained.detach(),
        alignment_rotation=rotation,
        alignment_scale=scale,
        alignment_offset=offset,
    )


def serialize_pca_projections(
    projections: dict[str, PCAProjection],
) -> dict[str, dict[str, torch.Tensor]]:
    return {name: projections[name].state_dict() for name in ATTRIBUTE_NAMES}


def load_pca_projections(
    payload: dict[str, dict[str, torch.Tensor]], device: torch.device
) -> dict[str, PCAProjection]:
    return {
        name: PCAProjection.from_state_dict(payload[name]).to(device)
        for name in ATTRIBUTE_NAMES
    }


def _class_balanced_mean(
    values: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    sample_weights = weights.to(values.device, values.dtype)[labels]
    return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def _supervised_contrastive_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if len(hidden) < 2:
        return hidden.float().sum() * 0.0
    normalized = F.normalize(hidden.float(), dim=1)
    similarities = (normalized @ normalized.T) / float(temperature)
    other_mask = ~torch.eye(len(hidden), device=hidden.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & other_mask
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return hidden.float().sum() * 0.0
    log_probabilities = similarities - torch.logsumexp(
        similarities.masked_fill(~other_mask, -torch.inf), dim=1, keepdim=True
    )
    per_sample = -torch.where(
        positive_mask, log_probabilities, torch.zeros_like(log_probabilities)
    ).sum(dim=1) / positive_counts.clamp_min(1)
    return _class_balanced_mean(
        per_sample[valid], labels[valid], class_weights
    )


@dataclass(frozen=True)
class AuxiliaryProbeLoss:
    total: torch.Tensor
    classification: torch.Tensor
    pca_distance: torch.Tensor
    target_position: torch.Tensor
    contrastive: torch.Tensor
    per_attribute: dict[str, dict[str, torch.Tensor]]


def compute_auxiliary_probe_loss(
    probe_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    ideal_targets: torch.Tensor,
    prototypes: tuple[torch.Tensor, ...],
    class_weights: tuple[torch.Tensor, ...],
    pca_projections: dict[str, PCAProjection],
    *,
    classification_weight: float,
    pca_distance_weight: float,
    distance_margin: float,
    target_position_weight: float = 0.0,
    label_smoothing: float = 0.0,
    attribute_weights: dict[str, float] | None = None,
    geometry_hiddens: dict[str, torch.Tensor] | None = None,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    negative_aggregation: str = "mean",
    attribute_distance_margins: dict[str, float] | None = None,
    classification_reduction: str = "mean",
) -> AuxiliaryProbeLoss:
    if labels.shape[1] != len(ATTRIBUTE_NAMES):
        raise ValueError("One label column is required per pathology attribute")
    target_blocks = ideal_targets.float().reshape(len(labels), len(ATTRIBUTE_NAMES), 2)
    classification_losses: list[torch.Tensor] = []
    distance_losses: list[torch.Tensor] = []
    position_losses: list[torch.Tensor] = []
    contrastive_losses: list[torch.Tensor] = []
    per_attribute: dict[str, dict[str, torch.Tensor]] = {}

    for index, name in enumerate(ATTRIBUTE_NAMES):
        hidden, logits = probe_outputs[name]
        geometry_hidden = (
            hidden if geometry_hiddens is None else geometry_hiddens[name]
        )
        attribute_labels = labels[:, index].long()
        weights = class_weights[index].to(logits.device, logits.dtype)
        classification = F.cross_entropy(
            logits.float(),
            attribute_labels,
            weight=weights.float(),
            label_smoothing=float(label_smoothing),
        )

        pca = pca_projections[name]
        coordinates = pca.project(geometry_hidden)
        positive_distance = torch.linalg.vector_norm(
            coordinates - target_blocks[:, index], dim=1
        )
        centers = prototypes[index].to(coordinates.device, coordinates.dtype)
        center_distances = torch.cdist(coordinates, centers, p=2)
        negative_mask = torch.ones_like(center_distances, dtype=torch.bool)
        negative_mask.scatter_(1, attribute_labels[:, None], False)
        margin = float(
            (attribute_distance_margins or {}).get(name, distance_margin)
        )
        violations = F.relu(positive_distance[:, None] - center_distances + margin)
        if negative_aggregation == "mean":
            per_sample_distance = (
                (violations * negative_mask).sum(dim=1)
                / negative_mask.sum(dim=1).clamp_min(1)
            )
        elif negative_aggregation == "hardest":
            per_sample_distance = violations.masked_fill(
                ~negative_mask, -torch.inf
            ).max(dim=1).values
        else:
            raise ValueError(
                "negative_aggregation must be 'mean' or 'hardest'"
            )
        pca_distance = _class_balanced_mean(
            per_sample_distance, attribute_labels, class_weights[index]
        )
        position_per_sample = F.smooth_l1_loss(
            coordinates, target_blocks[:, index], reduction="none"
        ).mean(dim=1)
        target_position = _class_balanced_mean(
            position_per_sample, attribute_labels, class_weights[index]
        )
        contrastive = (
            _supervised_contrastive_loss(
                geometry_hidden,
                attribute_labels,
                class_weights[index],
                contrastive_temperature,
            )
            if float(contrastive_weight) != 0.0
            else geometry_hidden.float().sum() * 0.0
        )

        classification_losses.append(classification)
        distance_losses.append(pca_distance)
        position_losses.append(target_position)
        contrastive_losses.append(contrastive)
        per_attribute[name] = {
            "classification": classification,
            "pca_distance": pca_distance,
            "target_position": target_position,
            "contrastive": contrastive,
        }

    configured_weights = attribute_weights or {}
    raw_weights = torch.tensor(
        [float(configured_weights.get(name, 1.0)) for name in ATTRIBUTE_NAMES],
        device=classification_losses[0].device,
    )
    normalized_weights = raw_weights / raw_weights.sum().clamp_min(1e-12)
    if classification_reduction == "mean":
        classification_weights = normalized_weights
    elif classification_reduction == "sum":
        classification_weights = raw_weights
    else:
        raise ValueError("classification_reduction must be 'mean' or 'sum'")
    classification = (
        torch.stack(classification_losses) * classification_weights
    ).sum()
    pca_distance = (torch.stack(distance_losses) * normalized_weights).sum()
    target_position = (torch.stack(position_losses) * normalized_weights).sum()
    contrastive = (torch.stack(contrastive_losses) * normalized_weights).sum()
    total = (
        float(classification_weight) * classification
        + float(pca_distance_weight) * pca_distance
        + float(target_position_weight) * target_position
        + float(contrastive_weight) * contrastive
    )
    return AuxiliaryProbeLoss(
        total,
        classification,
        pca_distance,
        target_position,
        contrastive,
        per_attribute,
    )
