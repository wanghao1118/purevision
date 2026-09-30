from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attribute_2d import ATTRIBUTE_NAMES
from .attributes import LOCATION_NAMES


class GeometryMemoryBank:


    def __init__(self, capacity: int = 1024):
        self.capacity = int(capacity)
        self._embeddings: deque[torch.Tensor] = deque()
        self._targets: deque[torch.Tensor] = deque()
        self._size = 0

    def add(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        targets = targets.detach()
        for embedding, target in zip(embeddings, targets):
            self._embeddings.append(embedding)
            self._targets.append(target)
            self._size += 1
        while self._size > self.capacity:
            self._embeddings.popleft()
            self._targets.popleft()
            self._size -= 1

    def tensors(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._embeddings:
            return None, None
        return torch.stack(tuple(self._embeddings)), torch.stack(tuple(self._targets))

    def clear(self) -> None:
        self._embeddings.clear()
        self._targets.clear()
        self._size = 0


class LearnableAttributeGeometry(nn.Module):


    def __init__(
        self,
        input_dimension: int,
        classes: int,
        *,
        ordinal: bool,
        seed: int,
        initial_spacing: float = 0.5,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dimension, 2)
        generator = torch.Generator().manual_seed(int(seed))
        if ordinal:
            x = torch.arange(classes, dtype=torch.float32)
            x = (x - x.mean()) * float(initial_spacing)
            y = 0.03 * torch.randn(classes, generator=generator)
            initial_centers = torch.stack((x, y), dim=1)
        else:
            initial_centers = 0.1 * torch.randn(classes, 2, generator=generator)
        self.centers = nn.Parameter(initial_centers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)


class EncoderWithLearnableGeometry(nn.Module):
    def __init__(self, encoder: nn.Module, geometry: LearnableAttributeGeometry):
        super().__init__()
        self.encoder = encoder
        self.geometry = geometry

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        outputs = self.encoder(*args, **kwargs)
        outputs["learnable_geometry_coordinates"] = self.geometry(outputs["embedding"])
        outputs["learnable_geometry_centers"] = self.geometry.centers
        return outputs


def _normalized_vector(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def dynamic_geometry_losses(
    embeddings: torch.Tensor,
    ideal_targets: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_targets: torch.Tensor | None,
    triplet_margin: float,
    ideal_gap_min: float,
    triplets_per_anchor: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    ideal_targets = ideal_targets.float()
    candidates = embeddings
    candidate_targets = ideal_targets
    current_count = embeddings.shape[0]
    if memory_embeddings is not None and memory_targets is not None:
        candidates = torch.cat(
            [candidates, F.normalize(memory_embeddings.float(), dim=-1)], dim=0
        )
        candidate_targets = torch.cat(
            [candidate_targets, memory_targets.float()], dim=0
        )

    learned_distances = torch.cdist(embeddings, candidates, p=2)
    ideal_distances = torch.cdist(ideal_targets, candidate_targets, p=2)
    valid_pair_mask = torch.ones_like(ideal_distances, dtype=torch.bool)
    diagonal = torch.arange(current_count, device=embeddings.device)
    valid_pair_mask[diagonal, diagonal] = False

    pairwise_learned = learned_distances[valid_pair_mask]
    pairwise_ideal = ideal_distances[valid_pair_mask]
    if pairwise_learned.numel() >= 2:
        pairwise_loss = F.mse_loss(
            _normalized_vector(pairwise_learned), _normalized_vector(pairwise_ideal)
        )
    else:
        pairwise_loss = embeddings.sum() * 0.0

    triplet_terms: list[torch.Tensor] = []
    for anchor in range(current_count):
        candidate_indices = torch.nonzero(
            valid_pair_mask[anchor], as_tuple=False
        ).flatten()
        if candidate_indices.numel() < 2:
            continue
        ideal = ideal_distances[anchor, candidate_indices]
        sorted_ideal, sorted_local = ideal.sort()
        negative_starts = torch.searchsorted(
            sorted_ideal,
            ideal + float(ideal_gap_min),
            right=True,
        )
        pair_counts = ideal.numel() - negative_starts
        total_pairs = int(pair_counts.sum().item())
        if total_pairs == 0:
            continue
        selected_count = min(total_pairs, int(triplets_per_anchor))
        if selected_count == total_pairs:
            selected_pairs = torch.arange(total_pairs, device=embeddings.device)
        elif generator is None:
            selected_pairs = (
                torch.linspace(
                    0,
                    total_pairs - 1,
                    selected_count,
                    device=embeddings.device,
                )
                .round()
                .long()
            )
        else:
            selected_values: set[int] = set()
            while len(selected_values) < selected_count:
                needed = selected_count - len(selected_values)
                draws = torch.randint(
                    total_pairs,
                    (max(needed * 2, 4),),
                    device=embeddings.device,
                    generator=generator,
                )
                selected_values.update(int(value) for value in draws.tolist())
            selected_pairs = torch.tensor(
                tuple(selected_values)[:selected_count],
                device=embeddings.device,
                dtype=torch.long,
            )
        cumulative = pair_counts.cumsum(dim=0)
        positive_local = torch.searchsorted(cumulative, selected_pairs, right=True)
        previous = torch.where(
            positive_local > 0,
            cumulative[(positive_local - 1).clamp_min(0)],
            torch.zeros_like(positive_local),
        )
        negative_offsets = selected_pairs - previous
        negative_local = sorted_local[
            negative_starts[positive_local] + negative_offsets
        ]
        positive = candidate_indices[positive_local]
        negative = candidate_indices[negative_local]
        positive_distance = learned_distances[anchor, positive]
        negative_distance = learned_distances[anchor, negative]
        triplet_terms.append(
            F.relu(positive_distance - negative_distance + triplet_margin)
        )

    if triplet_terms:
        triplet_loss = torch.cat(triplet_terms).mean()
        triplet_count = sum(term.numel() for term in triplet_terms)
    else:
        triplet_loss = embeddings.sum() * 0.0
        triplet_count = 0
    return triplet_loss, pairwise_loss, triplet_count


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_labels: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    labels = labels.long().reshape(-1)
    candidates = embeddings
    candidate_labels = labels
    current_count = embeddings.shape[0]
    if memory_embeddings is not None and memory_labels is not None:
        candidates = torch.cat(
            [candidates, F.normalize(memory_embeddings.float(), dim=-1)], dim=0
        )
        candidate_labels = torch.cat(
            [candidate_labels, memory_labels.long().reshape(-1)], dim=0
        )

    logits = embeddings @ candidates.T / float(temperature)
    valid = torch.ones_like(logits, dtype=torch.bool)
    diagonal = torch.arange(current_count, device=embeddings.device)
    valid[diagonal, diagonal] = False
    positive = valid & labels[:, None].eq(candidate_labels[None, :])
    valid_anchors = positive.any(dim=1)
    if not bool(valid_anchors.any()):
        return embeddings.sum() * 0.0

    denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    return (denominator[valid_anchors] - numerator[valid_anchors]).mean()


def attribute_relation_losses(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_labels: torch.Tensor | None,
    ordinal: bool,
    maximum_distance: float,
    order_margin: float,
    class_distance_matrix: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:

    embeddings = F.normalize(embeddings.float(), dim=-1)
    labels = labels.long()
    candidates = embeddings
    candidate_labels = labels
    current_count = len(embeddings)
    if memory_embeddings is not None and memory_labels is not None:
        candidates = torch.cat(
            [candidates, F.normalize(memory_embeddings.float(), dim=-1)], dim=0
        )
        candidate_labels = torch.cat(
            [candidate_labels, memory_labels.long().reshape(-1)], dim=0
        )

    learned = torch.cdist(embeddings, candidates, p=2)
    grade_gaps = (labels[:, None] - candidate_labels[None, :]).abs().float()
    classes = len(class_weights)
    if class_distance_matrix is not None:
        distance_matrix = torch.as_tensor(
            class_distance_matrix, device=embeddings.device, dtype=torch.float32
        )
        expected_shape = (classes, classes)
        if tuple(distance_matrix.shape) != expected_shape:
            raise ValueError(
                "class_distance_matrix must have shape "
                f"{expected_shape}, got {tuple(distance_matrix.shape)}"
            )
        desired = distance_matrix[labels][:, candidate_labels]
    elif ordinal:
        desired = float(maximum_distance) * grade_gaps / max(1, classes - 1)
    else:
        desired = float(maximum_distance) * (grade_gaps > 0).float()

    valid = torch.ones_like(learned, dtype=torch.bool)
    diagonal = torch.arange(current_count, device=embeddings.device)
    valid[diagonal, diagonal] = False
    weights = class_weights.to(embeddings.device, torch.float32)
    if ordinal:
        pair_weights = valid.float()
    else:
        pair_weights = torch.sqrt(
            weights[labels][:, None] * weights[candidate_labels][None, :]
        )
        pair_weights = pair_weights * valid
    pair_errors = F.smooth_l1_loss(learned, desired, reduction="none")
    distance_loss = (pair_errors * pair_weights).sum() / pair_weights.sum().clamp_min(
        1e-6
    )

    if not ordinal:
        return embeddings.sum() * 0.0, distance_loss, 0

    order_per_anchor: list[torch.Tensor] = []
    for anchor in range(current_count):
        means: list[torch.Tensor | None] = []
        gaps: list[int] = []
        for class_index in range(classes):
            class_mask = candidate_labels.eq(class_index) & valid[anchor]
            means.append(
                learned[anchor, class_mask].mean() if bool(class_mask.any()) else None
            )
            if class_distance_matrix is not None:
                gaps.append(
                    float(
                        distance_matrix[int(labels[anchor].item()), class_index].item()
                    )
                )
            elif ordinal:
                gaps.append(abs(int(labels[anchor].item()) - class_index))
            else:
                gaps.append(int(class_index != int(labels[anchor].item())))
        terms: list[torch.Tensor] = []
        for close_index in range(classes):
            if means[close_index] is None:
                continue
            for far_index in range(classes):
                if means[far_index] is None or gaps[close_index] >= gaps[far_index]:
                    continue
                gap_difference = gaps[far_index] - gaps[close_index]
                terms.append(
                    F.relu(
                        means[close_index]
                        - means[far_index]
                        + float(order_margin) * gap_difference
                    )
                )
        if terms:
            order_per_anchor.append(torch.stack(terms).mean())

    if order_per_anchor:
        order_values = torch.stack(order_per_anchor)
        order_loss = order_values.mean()
        order_count = sum(value.numel() for value in order_per_anchor)
    else:
        order_loss = embeddings.sum() * 0.0
        order_count = 0
    return order_loss, distance_loss, order_count


def continuous_attribute_relation_loss(
    embeddings: torch.Tensor,
    measurements: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_measurements: torch.Tensor | None,
    distance_scale: float = 1.5,
) -> torch.Tensor:

    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    measurements = measurements.float().reshape(-1)
    if measurements.shape != (len(embeddings),) or not torch.isfinite(
        measurements
    ).all():
        raise ValueError("measurements must be one finite value per embedding")

    candidates = embeddings
    candidate_measurements = measurements
    current_count = len(embeddings)
    if memory_embeddings is not None and memory_measurements is not None:
        candidates = torch.cat(
            [candidates, F.normalize(memory_embeddings.float(), dim=-1)], dim=0
        )
        candidate_measurements = torch.cat(
            [candidate_measurements, memory_measurements.float().reshape(-1)], dim=0
        )

    learned = torch.cdist(embeddings, candidates, p=2)
    desired = float(distance_scale) * (
        measurements[:, None] - candidate_measurements[None, :]
    ).abs()
    valid = torch.ones_like(learned, dtype=torch.bool)
    diagonal = torch.arange(current_count, device=embeddings.device)
    valid[diagonal, diagonal] = False
    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    return F.smooth_l1_loss(learned[valid], desired[valid], reduction="mean")


def learnable_center_losses(
    coordinates: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    ordinal: bool,
    order_margin: float,
    minimum_separation: float,
    maximum_adjacent_distance: float,
    separation_weight: float,
    adjacent_weight: float,
    radius_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    coordinates = coordinates.float()
    centers = centers.float()
    labels = labels.long()
    positive = centers[labels]
    compact_values = F.smooth_l1_loss(coordinates, positive, reduction="none").mean(
        dim=1
    )
    sample_weights = class_weights.to(coordinates.device, torch.float32)[labels]
    compact = (compact_values * sample_weights).sum() / sample_weights.sum().clamp_min(
        1e-6
    )

    distances = torch.cdist(centers, centers, p=2)
    classes = len(centers)
    off_diagonal = ~torch.eye(classes, device=centers.device, dtype=torch.bool)
    separation = (
        F.relu(float(minimum_separation) - distances[off_diagonal]).square().mean()
    )

    order_terms: list[torch.Tensor] = []
    adjacent_terms: list[torch.Tensor] = []
    if ordinal:
        for anchor in range(classes):
            for close in range(classes):
                close_gap = abs(anchor - close)
                if close_gap == 1:
                    adjacent_terms.append(
                        F.relu(
                            distances[anchor, close] - float(maximum_adjacent_distance)
                        ).square()
                    )
                for far in range(classes):
                    far_gap = abs(anchor - far)
                    if close_gap == 0 or close_gap >= far_gap:
                        continue
                    order_terms.append(
                        F.relu(
                            distances[anchor, close]
                            - distances[anchor, far]
                            + float(order_margin) * (far_gap - close_gap)
                        )
                    )
    order = torch.stack(order_terms).mean() if order_terms else coordinates.sum() * 0.0
    adjacent = (
        torch.stack(adjacent_terms).mean()
        if adjacent_terms
        else coordinates.sum() * 0.0
    )
    centered = centers - centers.mean(dim=0, keepdim=True)
    radius = centered.square().sum(dim=1).mean()
    relation = (
        order
        + float(separation_weight) * separation
        + float(adjacent_weight) * adjacent
        + float(radius_weight) * radius
    )
    return compact, relation


def _weighted_attribute_mean(
    values: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Sequence[torch.Tensor],
) -> torch.Tensor:

    attribute_losses: list[torch.Tensor] = []
    for attribute_index, weights in enumerate(class_weights):
        sample_weights = weights.to(values.device, values.dtype)[
            labels[:, attribute_index]
        ]
        attribute_losses.append(
            (values[:, attribute_index] * sample_weights).sum()
            / sample_weights.sum().clamp_min(1e-6)
        )
    return torch.stack(attribute_losses).mean()


def attribute_prototype_losses(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Sequence[torch.Tensor],
    class_weights: Sequence[torch.Tensor],
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("Attribute predictions must have shape [batch, attributes, 2]")
    if labels.shape != prediction.shape[:2]:
        raise ValueError(
            f"Attribute label mismatch: {labels.shape} vs {prediction.shape[:2]}"
        )
    if (
        len(prototypes) != prediction.shape[1]
        or len(class_weights) != prediction.shape[1]
    ):
        raise ValueError(
            "One prototype and class-weight tensor is required per attribute"
        )

    compact_per_sample: list[torch.Tensor] = []
    margin_per_sample: list[torch.Tensor] = []
    for attribute_index, prototype_values in enumerate(prototypes):
        points = prediction[:, attribute_index].float()
        attribute_labels = labels[:, attribute_index].long()
        centers = prototype_values.to(points.device, points.dtype)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("Each prototype tensor must have shape [classes, 2]")
        if attribute_labels.min() < 0 or attribute_labels.max() >= centers.shape[0]:
            raise ValueError("Attribute labels index outside the configured prototypes")

        positive_centers = centers[attribute_labels]
        compact_per_sample.append(
            F.smooth_l1_loss(points, positive_centers, reduction="none").mean(dim=1)
        )

        distances = torch.cdist(points, centers, p=2)
        positive_distances = distances.gather(1, attribute_labels[:, None])
        class_indices = torch.arange(centers.shape[0], device=points.device)
        grade_gaps = (class_indices[None, :] - attribute_labels[:, None]).abs()
        negative_mask = grade_gaps > 0
        violations = F.relu(positive_distances - distances + float(margin) * grade_gaps)
        margin_per_sample.append(
            (violations * negative_mask).sum(dim=1)
            / negative_mask.sum(dim=1).clamp_min(1)
        )

    compact_values = torch.stack(compact_per_sample, dim=1)
    margin_values = torch.stack(margin_per_sample, dim=1)
    return (
        _weighted_attribute_mean(compact_values, labels, class_weights),
        _weighted_attribute_mean(margin_values, labels, class_weights),
    )


@dataclass
class LossOutput:
    total: torch.Tensor
    coordinate: torch.Tensor
    triplet: torch.Tensor
    pairwise: torch.Tensor
    prototype_compact: torch.Tensor
    prototype_margin: torch.Tensor
    triplets: int


def compute_loss(
    outputs: dict[str, torch.Tensor],
    ideal_targets: torch.Tensor,
    memory_bank: GeometryMemoryBank,
    config: dict[str, Any],
    *,
    attribute_labels: torch.Tensor | None = None,
    attribute_measurements: torch.Tensor | None = None,
    attribute_prototypes: Sequence[torch.Tensor] | None = None,
    attribute_class_weights: Sequence[torch.Tensor] | None = None,
) -> LossOutput:
    prediction = outputs["coordinate"].float()
    target = ideal_targets.float()
    block_size = int(config.get("coordinate_block_size", target.shape[-1]))
    if prediction.shape != target.shape:
        raise ValueError(
            f"Coordinate prediction/target mismatch: {prediction.shape} vs {target.shape}"
        )
    if target.shape[-1] % block_size:
        raise ValueError("Target dimension must be divisible by coordinate_block_size")
    prediction_blocks = prediction.reshape(target.shape[0], -1, block_size)
    target_blocks = target.reshape(target.shape[0], -1, block_size)
    coordinate_kind = str(config.get("coordinate_loss_kind", "mse"))
    if coordinate_kind == "mse":
        coordinate = (
            (prediction_blocks - target_blocks).square().mean(dim=(0, 2)).mean()
        )
    elif coordinate_kind == "class_balanced_smooth_l1":
        if attribute_labels is None or attribute_class_weights is None:
            raise ValueError(
                "class_balanced_smooth_l1 requires labels and class weights"
            )
        coordinate_values = F.smooth_l1_loss(
            prediction_blocks, target_blocks, reduction="none"
        ).mean(dim=2)
        coordinate = _weighted_attribute_mean(
            coordinate_values, attribute_labels, attribute_class_weights
        )
    else:
        raise ValueError(f"Unsupported coordinate_loss_kind: {coordinate_kind}")

    zero = prediction.sum() * 0.0
    prototype_compact = zero
    prototype_margin = zero
    center_config = config.get("learnable_center_supervision", {})
    center_enabled = bool(center_config.get("enabled", False))
    prototype_context = (
        attribute_labels is not None
        and attribute_prototypes is not None
        and attribute_class_weights is not None
    )
    if center_enabled:
        if attribute_labels is None or attribute_class_weights is None:
            raise ValueError(
                "Learnable center supervision requires labels and class weights"
            )
        target_attribute = str(center_config["target_attribute"])
        if target_attribute not in ATTRIBUTE_NAMES:
            raise ValueError(f"Unknown target attribute: {target_attribute}")
        target_index = ATTRIBUTE_NAMES.index(target_attribute)
        coordinates = outputs.get("learnable_geometry_coordinates")
        centers = outputs.get("learnable_geometry_centers")
        if coordinates is None or centers is None:
            raise ValueError(
                "Learnable center supervision requires training geometry outputs"
            )
        prototype_compact, prototype_margin = learnable_center_losses(
            coordinates,
            attribute_labels[:, target_index],
            centers,
            attribute_class_weights[target_index],
            ordinal=target_attribute != "calcification",
            order_margin=float(center_config.get("order_margin", 0.15)),
            minimum_separation=float(center_config.get("minimum_separation", 0.5)),
            maximum_adjacent_distance=float(
                center_config.get("maximum_adjacent_distance", 1.25)
            ),
            separation_weight=float(center_config.get("separation_weight", 1.0)),
            adjacent_weight=float(center_config.get("adjacent_weight", 0.25)),
            radius_weight=float(center_config.get("radius_weight", 0.01)),
        )
    elif prototype_context:
        if block_size != 2:
            raise ValueError("Prototype losses require coordinate_block_size=2")
        prototype_compact, prototype_margin = attribute_prototype_losses(
            prediction_blocks,
            attribute_labels[:, : len(attribute_prototypes)],
            attribute_prototypes,
            attribute_class_weights,
            margin=float(config.get("prototype_margin", 0.2)),
        )
    elif float(config.get("prototype_compact_loss_weight", 0.0)) or float(
        config.get("prototype_margin_loss_weight", 0.0)
    ):
        raise ValueError("Prototype loss weights require complete attribute context")
    memory_embeddings, memory_targets = memory_bank.tensors()
    relation_config = config.get("attribute_relation_supervision", {})
    relation_enabled = bool(relation_config.get("enabled", False))
    if relation_enabled:
        if attribute_labels is None or attribute_class_weights is None:
            raise ValueError(
                "Attribute relation supervision requires labels and class weights"
            )
        tasks = relation_config.get("tasks")
        task_configs = list(tasks) if tasks is not None else [relation_config]
        if not task_configs:
            raise ValueError(
                "Attribute relation supervision requires at least one task"
            )
        task_weights = torch.as_tensor(
            [float(task.get("weight", 1.0)) for task in task_configs],
            dtype=torch.float32,
            device=outputs["normalized_embedding"].device,
        )
        if bool((task_weights <= 0).any()):
            raise ValueError("Attribute relation task weights must be positive")
        task_weights = task_weights / task_weights.sum()
        embedding_partition = str(relation_config.get("embedding_partition", "shared"))
        if embedding_partition not in {"shared", "equal_tasks", "task_slices"}:
            raise ValueError(
                "attribute_relation_supervision.embedding_partition must be "
                "'shared', 'equal_tasks', or 'task_slices'"
            )
        embedding_blocks: tuple[torch.Tensor, ...] | None = None
        memory_embedding_blocks: tuple[torch.Tensor, ...] | None = None
        if embedding_partition == "equal_tasks":
            embedding_dimension = outputs["normalized_embedding"].shape[-1]
            if embedding_dimension < len(task_configs):
                raise ValueError(
                    "Embedding dimension must be at least the number of relation tasks"
                )
            embedding_blocks = tuple(
                torch.tensor_split(
                    outputs["normalized_embedding"], len(task_configs), dim=-1
                )
            )
            if memory_embeddings is not None:
                memory_embedding_blocks = tuple(
                    torch.tensor_split(memory_embeddings, len(task_configs), dim=-1)
                )
        triplet_terms: list[torch.Tensor] = []
        pairwise_terms: list[torch.Tensor] = []
        relation_target_columns: list[torch.Tensor] = []
        count = 0
        for task_index, task in enumerate(task_configs):
            target_attribute = str(task["target_attribute"])
            task_kind = str(task.get("kind", "ordinal_or_categorical"))
            if task_kind not in {"continuous", "ordinal_or_categorical"}:
                raise ValueError(f"Unknown relation task kind: {task_kind}")
            if target_attribute in ATTRIBUTE_NAMES:
                target_index = ATTRIBUTE_NAMES.index(target_attribute)
                task_class_weights = attribute_class_weights[target_index]
                task_ordinal = target_attribute != "calcification"
            elif target_attribute == "location":
                target_index = len(ATTRIBUTE_NAMES)
                configured_weights = task.get("class_weights")
                if configured_weights is None:
                    raise ValueError(
                        "Location relation supervision requires class_weights"
                    )
                task_class_weights = torch.as_tensor(
                    configured_weights,
                    dtype=torch.float32,
                    device=outputs["normalized_embedding"].device,
                )
                if task_class_weights.shape != (len(LOCATION_NAMES),):
                    raise ValueError(
                        "Location relation class_weights must contain "
                        f"{len(LOCATION_NAMES)} values"
                    )
                if bool((task_class_weights <= 0).any()):
                    raise ValueError("Location relation class_weights must be positive")
                task_ordinal = False
            else:
                raise ValueError(f"Unknown target attribute: {target_attribute}")
            if attribute_labels.shape[1] <= target_index:
                raise ValueError(
                    f"Missing label column for relation target: {target_attribute}"
                )
            relation_labels = attribute_labels[:, target_index]
            task_memory_labels = None
            if memory_targets is not None:
                task_memory_labels = (
                    memory_targets[:, task_index]
                    if memory_targets.ndim == 2
                    else memory_targets
                )
            if embedding_partition == "task_slices":
                embedding_slice = task.get("embedding_slice", "shared")
                if embedding_slice == "shared":
                    task_embeddings = outputs["normalized_embedding"]
                    task_memory_embeddings = memory_embeddings
                else:
                    if not (
                        isinstance(embedding_slice, (list, tuple))
                        and len(embedding_slice) == 2
                    ):
                        raise ValueError(
                            "task_slices tasks require embedding_slice='shared' "
                            "or [start, end]"
                        )
                    start, end = (int(value) for value in embedding_slice)
                    embedding_dimension = outputs["normalized_embedding"].shape[-1]
                    if not 0 <= start < end <= embedding_dimension:
                        raise ValueError(
                            "embedding_slice must satisfy "
                            f"0 <= start < end <= {embedding_dimension}, got "
                            f"[{start}, {end}]"
                        )
                    task_embeddings = outputs["normalized_embedding"][..., start:end]
                    task_memory_embeddings = (
                        None
                        if memory_embeddings is None
                        else memory_embeddings[..., start:end]
                    )
            else:
                task_embeddings = (
                    embedding_blocks[task_index]
                    if embedding_blocks is not None
                    else outputs["normalized_embedding"]
                )
                task_memory_embeddings = (
                    memory_embedding_blocks[task_index]
                    if memory_embedding_blocks is not None
                    else memory_embeddings
                )
            if task_kind == "continuous":
                if attribute_measurements is None:
                    raise ValueError(
                        "Continuous relation supervision requires measurements"
                    )
                minimum = float(task["measurement_min"])
                maximum = float(task["measurement_max"])
                if not maximum > minimum:
                    raise ValueError("measurement_max must exceed measurement_min")
                relation_values = (
                    attribute_measurements[:, target_index].float() - minimum
                ) / (maximum - minimum)
                task_triplet = task_embeddings.sum() * 0.0
                task_pairwise = continuous_attribute_relation_loss(
                    task_embeddings,
                    relation_values,
                    memory_embeddings=task_memory_embeddings,
                    memory_measurements=task_memory_labels,
                    distance_scale=float(task.get("distance_scale", 1.5)),
                )
                task_count = 0
                relation_target = relation_values
            else:
                task_triplet, task_pairwise, task_count = attribute_relation_losses(
                    task_embeddings,
                    relation_labels,
                    task_class_weights,
                    memory_embeddings=task_memory_embeddings,
                    memory_labels=task_memory_labels,
                    ordinal=task_ordinal,
                    maximum_distance=float(task.get("maximum_distance", 1.5)),
                    order_margin=float(task.get("order_margin", 0.1)),
                    class_distance_matrix=task.get("class_distance_matrix"),
                )
                relation_target = relation_labels.float()
            triplet_terms.append(task_weights[task_index] * task_triplet)
            pairwise_terms.append(task_weights[task_index] * task_pairwise)
            relation_target_columns.append(relation_target)
            count += task_count
        triplet = torch.stack(triplet_terms).sum()
        pairwise = torch.stack(pairwise_terms).sum()
        memory_labels = torch.stack(relation_target_columns, dim=1)
        if len(task_configs) == 1:
            memory_labels = memory_labels[:, 0]
        memory_bank.add(outputs["normalized_embedding"], memory_labels)
    else:
        triplet, pairwise, count = dynamic_geometry_losses(
            outputs["normalized_embedding"],
            ideal_targets,
            memory_embeddings=memory_embeddings,
            memory_targets=memory_targets,
            triplet_margin=float(config["triplet_margin"]),
            ideal_gap_min=float(config["ideal_gap_min"]),
            triplets_per_anchor=int(config["triplets_per_anchor"]),
        )
        memory_bank.add(outputs["normalized_embedding"], ideal_targets)
    total = (
        float(config["coordinate_loss_weight"]) * coordinate
        + float(config["triplet_loss_weight"]) * triplet
        + float(config["pairwise_loss_weight"]) * pairwise
        + float(config.get("prototype_compact_loss_weight", 0.0)) * prototype_compact
        + float(config.get("prototype_margin_loss_weight", 0.0)) * prototype_margin
    )
    return LossOutput(
        total,
        coordinate,
        triplet,
        pairwise,
        prototype_compact,
        prototype_margin,
        count,
    )
