from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialSelection:
    anchor_index: int
    window_bounds: tuple[int, int, int, int]
    patch_indices: tuple[int, ...]
    lesion_scores: tuple[float, ...]
    patch_weights: tuple[float, ...]
    center_row: float
    center_column: float


@dataclass(frozen=True)
class FusedSemanticGroup:
    name: str
    labels: tuple[str, ...]
    scores: torch.Tensor
    probabilities: torch.Tensor
    soft_tokens: torch.Tensor


@dataclass(frozen=True)
class FusionResult:
    selection: SpatialSelection
    groups: tuple[FusedSemanticGroup, ...]

    def soft_tokens(self) -> dict[str, torch.Tensor]:
        return {group.name: group.soft_tokens for group in self.groups}


def cosine_similarity_matrix(
    patch_embeddings: torch.Tensor, text_embeddings: torch.Tensor
) -> torch.Tensor:
    if patch_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("patch and text embeddings must both be matrices")
    if patch_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("patch and text embedding dimensions do not match")
    return F.normalize(patch_embeddings.float(), dim=-1) @ F.normalize(
        text_embeddings.float(), dim=-1
    ).T


def lesion_margin(
    anatomy_similarities: torch.Tensor,
    lesion_class_indices: Sequence[int],
    other_class_indices: Sequence[int],
) -> torch.Tensor:
    if anatomy_similarities.ndim != 2:
        raise ValueError("anatomy similarities must have shape [patches, classes]")
    lesion = torch.as_tensor(
        lesion_class_indices, dtype=torch.long, device=anatomy_similarities.device
    )
    other = torch.as_tensor(
        other_class_indices, dtype=torch.long, device=anatomy_similarities.device
    )
    classes = anatomy_similarities.shape[1]
    if not len(lesion) or not len(other):
        raise ValueError("lesion and other anatomy class sets must be non-empty")
    if int(lesion.min()) < 0 or int(other.min()) < 0:
        raise ValueError("class indices must be non-negative")
    if int(lesion.max()) >= classes or int(other.max()) >= classes:
        raise ValueError("class index exceeds the anatomy target bank")
    if set(lesion.tolist()) & set(other.tolist()):
        raise ValueError("lesion and other anatomy class sets must be disjoint")
    return anatomy_similarities[:, lesion].amax(dim=1) - anatomy_similarities[
        :, other
    ].amax(dim=1)


def select_local_lesion_region(
    scores: torch.Tensor,
    *,
    grid_size: int = 64,
    window_size: int = 5,
    anchor_top_k: int = 2,
    patch_count: int = 8,
) -> SpatialSelection:

    values = scores.detach().float().reshape(-1)
    if values.shape != (grid_size * grid_size,) or not torch.isfinite(values).all():
        raise ValueError("scores must be one finite flattened square patch grid")
    if grid_size <= 0 or window_size <= 0 or window_size % 2 != 1:
        raise ValueError("grid_size must be positive and window_size must be odd")
    if not 0 < anchor_top_k <= window_size * window_size:
        raise ValueError("anchor_top_k is outside the candidate window")
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")

    radius = window_size // 2
    image = values.reshape(1, 1, grid_size, grid_size)
    padded = F.pad(image, (radius, radius, radius, radius), value=-torch.inf)
    windows = padded.unfold(2, window_size, 1).unfold(3, window_size, 1)
    windows = windows.reshape(grid_size, grid_size, -1)
    concentration = windows.topk(anchor_top_k, dim=-1).values.mean(dim=-1)



    padding_floor = values.min() - values.abs().amax() - 1.0
    complete_windows = F.pad(
        image, (radius, radius, radius, radius), value=float(padding_floor)
    ).unfold(2, window_size, 1).unfold(3, window_size, 1)
    complete_score = complete_windows.reshape(grid_size, grid_size, -1).sum(dim=-1)
    primary = concentration.reshape(-1)
    candidates = primary.eq(primary.max())
    secondary = complete_score.reshape(-1).masked_fill(~candidates, -torch.inf)
    anchor_index = int(secondary.argmax().item())
    anchor_row, anchor_column = divmod(anchor_index, grid_size)

    row_start = max(0, anchor_row - radius)
    row_stop = min(grid_size, anchor_row + radius + 1)
    column_start = max(0, anchor_column - radius)
    column_stop = min(grid_size, anchor_column + radius + 1)
    local_indices = [
        row * grid_size + column
        for row in range(row_start, row_stop)
        for column in range(column_start, column_stop)
    ]
    selected = sorted(
        local_indices, key=lambda index: (-float(values[index]), index)
    )[: min(patch_count, len(local_indices))]
    selected_tensor = torch.as_tensor(
        selected, dtype=torch.long, device=values.device
    )
    selected_scores = values[selected_tensor]

    weights = torch.softmax(selected_scores, dim=0)
    rows = selected_tensor.div(grid_size, rounding_mode="floor").float() + 0.5
    columns = selected_tensor.remainder(grid_size).float() + 0.5
    center_row = float((rows * weights).sum().item())
    center_column = float((columns * weights).sum().item())
    return SpatialSelection(
        anchor_index=anchor_index,
        window_bounds=(row_start, row_stop, column_start, column_stop),
        patch_indices=tuple(int(value) for value in selected),
        lesion_scores=tuple(float(value) for value in selected_scores.cpu()),
        patch_weights=tuple(float(value) for value in weights.cpu()),
        center_row=center_row,
        center_column=center_column,
    )


def aggregate_candidate_scores(
    similarities: torch.Tensor,
    patch_indices: Sequence[int],
    patch_weights: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [patches, candidates]")
    indices = torch.as_tensor(
        patch_indices, dtype=torch.long, device=similarities.device
    )
    weights = torch.as_tensor(
        patch_weights, dtype=torch.float32, device=similarities.device
    )
    if not len(indices) or weights.shape != indices.shape:
        raise ValueError("patch indices and weights must be aligned non-empty vectors")
    if int(indices.min()) < 0 or int(indices.max()) >= len(similarities):
        raise ValueError("patch index exceeds the similarity matrix")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("patch weights must be finite and non-negative")
    weights = weights / weights.sum().clamp_min(1e-12)
    return torch.einsum("k,kc->c", weights, similarities[indices].float())


def standardized_softmax(
    candidate_scores: torch.Tensor, *, temperature: float = 0.125
) -> torch.Tensor:
    scores = candidate_scores.float().reshape(-1)
    if not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("candidate scores must be one finite non-empty vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centered = scores - scores.mean()
    scale = centered.std(unbiased=False)
    standardized = centered / scale if float(scale) > 1e-8 else centered
    return torch.softmax(standardized / float(temperature), dim=0)


def mix_token_sequences(
    sequences: Sequence[torch.Tensor], weights: Sequence[float] | torch.Tensor
) -> torch.Tensor:

    if not sequences:
        raise ValueError("at least one candidate token sequence is required")
    values = [sequence.float() for sequence in sequences]
    dimension = values[0].shape[-1]
    if any(
        value.ndim != 2 or not len(value) or value.shape[-1] != dimension
        for value in values
    ):
        raise ValueError("candidate sequences must be non-empty [tokens, dimension]")
    mixture_weights = torch.as_tensor(
        weights, dtype=torch.float32, device=values[0].device
    )
    if mixture_weights.shape != (len(values),):
        raise ValueError("mixture weights do not match candidate sequences")
    if not torch.isfinite(mixture_weights).all() or bool(
        (mixture_weights < 0).any()
    ):
        raise ValueError("mixture weights must be finite and non-negative")
    mixture_weights = mixture_weights / mixture_weights.sum().clamp_min(1e-12)
    maximum_length = max(len(value) for value in values)
    padded = []
    for value in values:
        if value.device != values[0].device:
            value = value.to(values[0].device)
        if len(value) < maximum_length:
            value = torch.cat(
                (value, value[-1:].expand(maximum_length - len(value), -1)), dim=0
            )
        padded.append(value)
    return torch.einsum("c,cld->ld", mixture_weights, torch.stack(padded))


def fuse_semantic_group(
    name: str,
    labels: Sequence[str],
    similarities: torch.Tensor,
    candidate_token_sequences: Sequence[torch.Tensor],
    selection: SpatialSelection,
    *,
    semantic_temperature: float = 0.125,
) -> FusedSemanticGroup:
    labels = tuple(str(label) for label in labels)
    if similarities.shape[1] != len(labels):
        raise ValueError(f"candidate labels do not match similarities for {name}")
    if len(candidate_token_sequences) != len(labels):
        raise ValueError(f"candidate token sequences do not match labels for {name}")
    scores = aggregate_candidate_scores(
        similarities, selection.patch_indices, selection.patch_weights
    )
    probabilities = standardized_softmax(
        scores, temperature=semantic_temperature
    )
    soft_tokens = mix_token_sequences(candidate_token_sequences, probabilities)
    return FusedSemanticGroup(
        name=str(name),
        labels=labels,
        scores=scores,
        probabilities=probabilities,
        soft_tokens=soft_tokens,
    )


def fuse_semantic_groups(
    selection: SpatialSelection,
    group_similarities: Mapping[str, torch.Tensor],
    group_labels: Mapping[str, Sequence[str]],
    group_token_sequences: Mapping[str, Sequence[torch.Tensor]],
    *,
    semantic_temperature: float = 0.125,
) -> FusionResult:
    if tuple(group_similarities) != tuple(group_labels) or tuple(
        group_similarities
    ) != tuple(group_token_sequences):
        raise ValueError("semantic group mappings must have identical ordered keys")
    groups = tuple(
        fuse_semantic_group(
            name,
            group_labels[name],
            group_similarities[name],
            group_token_sequences[name],
            selection,
            semantic_temperature=semantic_temperature,
        )
        for name in group_similarities
    )
    return FusionResult(selection=selection, groups=groups)
