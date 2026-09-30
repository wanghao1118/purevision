from __future__ import annotations

import torch
import torch.nn.functional as F


NORMAL_PATCH = 0
LESION_PATCH = 1


def lesion_parent_patch_labels(
    side_masks: torch.Tensor, lung_classes: torch.Tensor
) -> torch.Tensor:

    if side_masks.ndim != 2:
        raise ValueError("side_masks must have shape [batch, patches]")
    if lung_classes.shape != (side_masks.shape[0],):
        raise ValueError("lung_classes must have one value per image")
    if not bool(((lung_classes == 1) | (lung_classes == 2)).all()):
        raise ValueError("lung_classes must contain left=1 or right=2")
    labels = torch.full_like(side_masks, -1, dtype=torch.long)
    expanded_classes = lung_classes.long().unsqueeze(1).expand_as(labels)
    labels[side_masks.bool()] = expanded_classes[side_masks.bool()]
    return labels


def teacher_relative_parent_pair_separation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    memory_student_embeddings: torch.Tensor | None,
    memory_teacher_embeddings: torch.Tensor | None,
    memory_labels: torch.Tensor | None,
    class_pairs: tuple[tuple[int, int], ...],
    improvement_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student and teacher embeddings must have equal shape")
    if student_embeddings.ndim != 2:
        raise ValueError("student and teacher embeddings must be matrices")
    labels = labels.long().reshape(-1)
    if len(labels) != len(student_embeddings):
        raise ValueError("labels must match the embedding rows")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")
    memory_items = (
        memory_student_embeddings,
        memory_teacher_embeddings,
        memory_labels,
    )
    if any(item is None for item in memory_items) and not all(
        item is None for item in memory_items
    ):
        raise ValueError("student, teacher, and label memories must be provided together")

    student = F.normalize(student_embeddings.float(), dim=-1)
    teacher = F.normalize(teacher_embeddings.detach().float(), dim=-1)
    student_candidates = student
    teacher_candidates = teacher
    candidate_labels = labels
    if memory_student_embeddings is not None:
        assert memory_teacher_embeddings is not None
        assert memory_labels is not None
        if memory_student_embeddings.shape != memory_teacher_embeddings.shape:
            raise ValueError("student and teacher memories must have equal shape")
        if memory_student_embeddings.ndim != 2:
            raise ValueError("student and teacher memories must be matrices")
        memory_labels = memory_labels.long().reshape(-1)
        if len(memory_labels) != len(memory_student_embeddings):
            raise ValueError("memory labels must match memory embedding rows")
        student_candidates = torch.cat(
            (
                student_candidates,
                F.normalize(memory_student_embeddings.detach().float(), dim=-1),
            ),
            dim=0,
        )
        teacher_candidates = torch.cat(
            (
                teacher_candidates,
                F.normalize(memory_teacher_embeddings.detach().float(), dim=-1),
            ),
            dim=0,
        )
        candidate_labels = torch.cat((candidate_labels, memory_labels), dim=0)

    terms: list[torch.Tensor] = []
    student_distances: list[torch.Tensor] = []
    teacher_distances: list[torch.Tensor] = []
    comparisons = 0
    for first, second in class_pairs:
        if first == second:
            raise ValueError("class pairs must contain different labels")
        for anchor_class, candidate_class in ((first, second), (second, first)):
            anchor_mask = labels.eq(anchor_class)
            candidate_mask = candidate_labels.eq(candidate_class)
            if not bool(anchor_mask.any()) or not bool(candidate_mask.any()):
                continue
            student_similarity = (
                student[anchor_mask] @ student_candidates[candidate_mask].T
            ).mean()
            teacher_similarity = (
                teacher[anchor_mask] @ teacher_candidates[candidate_mask].T
            ).mean()
            terms.append(
                F.relu(
                    student_similarity
                    - teacher_similarity
                    + float(improvement_margin)
                )
            )
            student_distances.append(1.0 - student_similarity)
            teacher_distances.append(1.0 - teacher_similarity)
            comparisons += int(anchor_mask.sum()) * int(candidate_mask.sum())

    if not terms:
        zero = student.sum() * 0.0
        return zero, zero.detach(), zero.detach(), 0
    return (
        torch.stack(terms).mean(),
        torch.stack(student_distances).mean(),
        torch.stack(teacher_distances).mean(),
        comparisons,
    )


def focused_anchor_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_labels: torch.Tensor | None,
    anchor_class: int,
    negative_classes: tuple[int, ...],
    temperature: float,
) -> torch.Tensor:

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    labels = labels.long().reshape(-1)
    if len(labels) != len(embeddings):
        raise ValueError("labels must match the embedding rows")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if anchor_class in negative_classes:
        raise ValueError("anchor_class cannot also be a negative class")
    if (memory_embeddings is None) != (memory_labels is None):
        raise ValueError("memory embeddings and labels must be provided together")

    current = F.normalize(embeddings.float(), dim=-1)


    candidates = current.detach()
    candidate_labels = labels
    if memory_embeddings is not None:
        assert memory_labels is not None
        memory_labels = memory_labels.long().reshape(-1)
        if memory_embeddings.ndim != 2 or len(memory_embeddings) != len(memory_labels):
            raise ValueError("memory embeddings and labels must have equal rows")
        candidates = torch.cat(
            (candidates, F.normalize(memory_embeddings.detach().float(), dim=-1)),
            dim=0,
        )
        candidate_labels = torch.cat((candidate_labels, memory_labels), dim=0)

    anchor_mask = labels.eq(int(anchor_class))
    if not bool(anchor_mask.any()):
        return current.sum() * 0.0
    allowed = candidate_labels.eq(int(anchor_class))
    for class_index in negative_classes:
        allowed |= candidate_labels.eq(int(class_index))
    valid = allowed.unsqueeze(0).expand(int(anchor_mask.sum()), -1).clone()
    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    valid[
        torch.arange(len(anchor_indices), device=embeddings.device), anchor_indices
    ] = False
    positive = valid & candidate_labels.eq(int(anchor_class)).unsqueeze(0)
    negative = valid & ~positive
    usable = positive.any(dim=1) & negative.any(dim=1)
    if not bool(usable.any()):
        return current.sum() * 0.0

    logits = current[anchor_mask] @ candidates.T / float(temperature)
    denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    return (denominator[usable] - numerator[usable]).mean()


def teacher_relative_anchor_hard_negative_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    memory_student_embeddings: torch.Tensor | None,
    memory_teacher_embeddings: torch.Tensor | None,
    memory_labels: torch.Tensor | None,
    anchor_class: int,
    negative_classes: tuple[int, ...],
    improvement_margin: float,
    hard_negatives: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student and teacher embeddings must have equal shape")
    if student_embeddings.ndim != 2:
        raise ValueError("student and teacher embeddings must be matrices")
    labels = labels.long().reshape(-1)
    if len(labels) != len(student_embeddings):
        raise ValueError("labels must match the embedding rows")
    if anchor_class in negative_classes:
        raise ValueError("anchor_class cannot also be a negative class")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")
    if hard_negatives <= 0:
        raise ValueError("hard_negatives must be positive")
    memories = (
        memory_student_embeddings,
        memory_teacher_embeddings,
        memory_labels,
    )
    if any(item is None for item in memories) and not all(
        item is None for item in memories
    ):
        raise ValueError("student, teacher, and label memories must be provided together")

    student = F.normalize(student_embeddings.float(), dim=-1)
    teacher = F.normalize(teacher_embeddings.detach().float(), dim=-1)


    student_candidates = student.detach()
    teacher_candidates = teacher
    candidate_labels = labels
    if memory_student_embeddings is not None:
        assert memory_teacher_embeddings is not None
        assert memory_labels is not None
        if memory_student_embeddings.shape != memory_teacher_embeddings.shape:
            raise ValueError("student and teacher memories must have equal shape")
        memory_labels = memory_labels.long().reshape(-1)
        if len(memory_labels) != len(memory_student_embeddings):
            raise ValueError("memory labels must match memory embedding rows")
        student_candidates = torch.cat(
            (
                student_candidates,
                F.normalize(memory_student_embeddings.detach().float(), dim=-1),
            ),
            dim=0,
        )
        teacher_candidates = torch.cat(
            (
                teacher_candidates,
                F.normalize(memory_teacher_embeddings.detach().float(), dim=-1),
            ),
            dim=0,
        )
        candidate_labels = torch.cat((candidate_labels, memory_labels), dim=0)

    anchor_mask = labels.eq(int(anchor_class))
    if not bool(anchor_mask.any()):
        zero = student.sum() * 0.0
        return zero, zero.detach(), zero.detach(), 0

    terms: list[torch.Tensor] = []
    student_similarities: list[torch.Tensor] = []
    teacher_similarities: list[torch.Tensor] = []
    comparisons = 0
    anchors = student[anchor_mask]
    teacher_anchors = teacher[anchor_mask]
    for negative_class in negative_classes:
        negative_mask = candidate_labels.eq(int(negative_class))
        if not bool(negative_mask.any()):
            continue
        student_matrix = anchors @ student_candidates[negative_mask].T
        teacher_matrix = teacher_anchors @ teacher_candidates[negative_mask].T
        count = min(int(hard_negatives), student_matrix.shape[1])
        student_hard, hard_indices = torch.topk(
            student_matrix, k=count, dim=1, largest=True, sorted=False
        )
        teacher_hard = teacher_matrix.gather(1, hard_indices)
        student_mean = student_hard.mean(dim=1)
        teacher_mean = teacher_hard.mean(dim=1)
        terms.append(
            F.relu(student_mean - teacher_mean + float(improvement_margin)).mean()
        )
        student_similarities.append(student_mean.mean())
        teacher_similarities.append(teacher_mean.mean())
        comparisons += int(anchor_mask.sum()) * count

    if not terms:
        zero = student.sum() * 0.0
        return zero, zero.detach(), zero.detach(), 0
    return (
        torch.stack(terms).mean(),
        torch.stack(student_similarities).mean(),
        torch.stack(teacher_similarities).mean(),
        comparisons,
    )


def hierarchical_lung_targets(
    parent_prototypes: torch.Tensor,
    lung_labels: torch.Tensor,
    status_labels: torch.Tensor,
    *,
    status_offset: float,
) -> torch.Tensor:

    if parent_prototypes.ndim != 2:
        raise ValueError("parent_prototypes must have shape [classes, dimensions]")
    lung_labels = lung_labels.long().reshape(-1)
    status_labels = status_labels.long().reshape(-1)
    if lung_labels.shape != status_labels.shape:
        raise ValueError("lung_labels and status_labels must have equal shape")
    if not bool(((lung_labels == 1) | (lung_labels == 2)).all()):
        raise ValueError("Hierarchical lesion targets only support left/right lung labels")
    if not bool(((status_labels == NORMAL_PATCH) | (status_labels == LESION_PATCH)).all()):
        raise ValueError("status_labels must be normal=0 or lesion=1")
    if status_offset <= 0:
        raise ValueError("status_offset must be positive")

    dimensions = int(parent_prototypes.shape[1])
    targets = parent_prototypes.new_zeros((len(lung_labels), dimensions + 2))
    targets[:, :dimensions] = parent_prototypes[lung_labels]
    row = torch.arange(len(lung_labels), device=lung_labels.device)
    axis = dimensions + lung_labels - 1
    sign = status_labels.to(dtype=targets.dtype).mul(2.0).sub(1.0)
    targets[row, axis] = sign * float(status_offset)
    return targets


def hierarchical_group_labels_from_targets(ideal_targets: torch.Tensor) -> torch.Tensor:

    if ideal_targets.ndim != 2 or ideal_targets.shape[1] < 2:
        raise ValueError("ideal_targets must be a matrix with two status axes")
    status_axes = ideal_targets[:, -2:]
    active = status_axes.abs() > 0
    if not bool(active.sum(dim=1).eq(1).all()):
        raise ValueError("Every hierarchy target must activate exactly one status axis")
    lungs = status_axes.abs().argmax(dim=1)
    signs = status_axes.gather(1, lungs[:, None]).squeeze(1)
    statuses = signs.gt(0).long()
    return lungs.long() * 2 + statuses


def hierarchical_status_centroid_loss(
    embeddings: torch.Tensor,
    ideal_targets: torch.Tensor,
    lung_labels: torch.Tensor,
    status_labels: torch.Tensor,
    *,
    memory_embeddings: torch.Tensor | None,
    memory_targets: torch.Tensor | None,
    minimum_cosine_distance: float,
    compact_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if embeddings.ndim != 2 or ideal_targets.ndim != 2:
        raise ValueError("embeddings and ideal_targets must be matrices")
    if embeddings.shape[0] != ideal_targets.shape[0]:
        raise ValueError("embeddings and ideal_targets must have equal rows")
    lung_labels = lung_labels.long().reshape(-1)
    status_labels = status_labels.long().reshape(-1)
    if lung_labels.shape != status_labels.shape or len(lung_labels) != len(embeddings):
        raise ValueError("labels must match the embedding rows")
    if not 0.0 < minimum_cosine_distance <= 2.0:
        raise ValueError("minimum_cosine_distance must be in (0, 2]")
    if compact_weight < 0.0:
        raise ValueError("compact_weight must be nonnegative")

    current = F.normalize(embeddings.float(), dim=-1)
    candidates = current
    candidate_targets = ideal_targets.float()
    if memory_embeddings is not None and memory_targets is not None:
        candidates = torch.cat(
            (candidates, F.normalize(memory_embeddings.float(), dim=-1)), dim=0
        )
        candidate_targets = torch.cat(
            (candidate_targets, memory_targets.float()), dim=0
        )

    separation_terms: list[torch.Tensor] = []
    compact_terms: list[torch.Tensor] = []
    for lung in lung_labels.unique(sorted=True):
        centers: list[torch.Tensor] = []
        for status in (NORMAL_PATCH, LESION_PATCH):
            current_mask = (lung_labels == lung) & (status_labels == status)
            if not bool(current_mask.any()):
                raise ValueError(f"Missing status={status} for lung={int(lung)}")
            group_target = ideal_targets[current_mask][0].float()
            candidate_mask = torch.isclose(
                candidate_targets,
                group_target.unsqueeze(0),
                rtol=0.0,
                atol=1e-6,
            ).all(dim=1)
            center = F.normalize(
                candidates[candidate_mask].mean(dim=0), dim=0
            )
            centers.append(center)
            compact_terms.append(
                (1.0 - current[current_mask] @ center).mean()
            )
        cosine_distance = 1.0 - torch.dot(centers[0], centers[1])
        separation_terms.append(
            F.relu(cosine_distance.new_tensor(minimum_cosine_distance) - cosine_distance)
        )
    separation = torch.stack(separation_terms).mean()
    compact = torch.stack(compact_terms).mean()
    return separation + float(compact_weight) * compact, separation, compact


def teacher_relative_status_gap_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    lung_labels: torch.Tensor,
    status_labels: torch.Tensor,
    *,
    improvement_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student_embeddings and teacher_embeddings must share shape")
    if student_embeddings.ndim != 2:
        raise ValueError("student_embeddings and teacher_embeddings must be matrices")
    lung_labels = lung_labels.long().reshape(-1)
    status_labels = status_labels.long().reshape(-1)
    if lung_labels.shape != status_labels.shape or len(lung_labels) != len(student_embeddings):
        raise ValueError("labels must match the embedding rows")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")

    student = F.normalize(student_embeddings.float(), dim=-1)
    teacher = F.normalize(teacher_embeddings.detach().float(), dim=-1)
    losses: list[torch.Tensor] = []
    student_gaps: list[torch.Tensor] = []
    teacher_gaps: list[torch.Tensor] = []
    for lung in lung_labels.unique(sorted=True):
        lung_mask = lung_labels == lung
        masks = [lung_mask & (status_labels == status) for status in (NORMAL_PATCH, LESION_PATCH)]
        if not all(bool(mask.any()) for mask in masks):
            raise ValueError(f"Both statuses are required for lung={int(lung)}")
        student_centers = [F.normalize(student[mask].mean(dim=0), dim=0) for mask in masks]
        teacher_centers = [F.normalize(teacher[mask].mean(dim=0), dim=0) for mask in masks]
        for status, mask in enumerate(masks):
            other = 1 - status
            student_gap = (
                student[mask] @ student_centers[status]
                - student[mask] @ student_centers[other]
            )
            teacher_gap = (
                teacher[mask] @ teacher_centers[status]
                - teacher[mask] @ teacher_centers[other]
            )
            losses.append(
                F.relu(teacher_gap + float(improvement_margin) - student_gap).mean()
            )
            student_gaps.append(student_gap.mean())
            teacher_gaps.append(teacher_gap.mean())
    return (
        torch.stack(losses).mean(),
        torch.stack(student_gaps).mean(),
        torch.stack(teacher_gaps).mean(),
    )


def memory_teacher_relative_status_gap_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    ideal_targets: torch.Tensor,
    lung_labels: torch.Tensor,
    status_labels: torch.Tensor,
    *,
    memory_teacher_embeddings: torch.Tensor | None,
    memory_targets: torch.Tensor | None,
    improvement_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student_embeddings and teacher_embeddings must share shape")
    if student_embeddings.ndim != 2 or ideal_targets.ndim != 2:
        raise ValueError("embeddings and ideal_targets must be matrices")
    if len(student_embeddings) != len(ideal_targets):
        raise ValueError("embeddings and ideal_targets must have equal rows")
    lung_labels = lung_labels.long().reshape(-1)
    status_labels = status_labels.long().reshape(-1)
    if lung_labels.shape != status_labels.shape or len(lung_labels) != len(student_embeddings):
        raise ValueError("labels must match the embedding rows")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")
    if (memory_teacher_embeddings is None) != (memory_targets is None):
        raise ValueError("memory teacher embeddings and targets must be paired")

    student = F.normalize(student_embeddings.float(), dim=-1)
    teacher = F.normalize(teacher_embeddings.detach().float(), dim=-1)
    teacher_candidates = teacher
    target_candidates = ideal_targets.float()
    if memory_teacher_embeddings is not None and memory_targets is not None:
        teacher_candidates = torch.cat(
            (
                teacher_candidates,
                F.normalize(memory_teacher_embeddings.detach().float(), dim=-1),
            ),
            dim=0,
        )
        target_candidates = torch.cat(
            (target_candidates, memory_targets.float()), dim=0
        )

    losses: list[torch.Tensor] = []
    student_gaps: list[torch.Tensor] = []
    teacher_gaps: list[torch.Tensor] = []
    for lung in lung_labels.unique(sorted=True):
        lung_mask = lung_labels == lung
        masks = [lung_mask & (status_labels == status) for status in (NORMAL_PATCH, LESION_PATCH)]
        if not all(bool(mask.any()) for mask in masks):
            raise ValueError(f"Both statuses are required for lung={int(lung)}")
        centers: list[torch.Tensor] = []
        for mask in masks:
            group_target = ideal_targets[mask][0].float()
            candidate_mask = torch.isclose(
                target_candidates,
                group_target.unsqueeze(0),
                rtol=0.0,
                atol=1e-6,
            ).all(dim=1)
            centers.append(
                F.normalize(teacher_candidates[candidate_mask].mean(dim=0), dim=0)
            )
        for status, mask in enumerate(masks):
            other = 1 - status
            student_gap = student[mask] @ centers[status] - student[mask] @ centers[other]
            teacher_gap = teacher[mask] @ centers[status] - teacher[mask] @ centers[other]
            losses.append(
                F.relu(teacher_gap + float(improvement_margin) - student_gap).mean()
            )
            student_gaps.append(student_gap.mean())
            teacher_gaps.append(teacher_gap.mean())
    return (
        torch.stack(losses).mean(),
        torch.stack(student_gaps).mean(),
        torch.stack(teacher_gaps).mean(),
    )


def teacher_relative_parent_containment_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    lung_labels: torch.Tensor,
    *,
    maximum_similarity_drop: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError("student_embeddings and teacher_embeddings must share shape")
    if student_embeddings.ndim != 2:
        raise ValueError("student_embeddings and teacher_embeddings must be matrices")
    lung_labels = lung_labels.long().reshape(-1)
    if len(lung_labels) != len(student_embeddings):
        raise ValueError("lung_labels must match the embedding rows")
    if maximum_similarity_drop < 0.0:
        raise ValueError("maximum_similarity_drop must be nonnegative")

    student = F.normalize(student_embeddings.float(), dim=-1)
    teacher = F.normalize(teacher_embeddings.detach().float(), dim=-1)
    losses: list[torch.Tensor] = []
    student_similarities: list[torch.Tensor] = []
    teacher_similarities: list[torch.Tensor] = []
    for lung in lung_labels.unique(sorted=True):
        mask = lung_labels == lung
        teacher_parent = F.normalize(teacher[mask].mean(dim=0), dim=0)
        student_similarity = student[mask] @ teacher_parent
        teacher_similarity = teacher[mask] @ teacher_parent
        losses.append(
            F.relu(
                teacher_similarity
                - float(maximum_similarity_drop)
                - student_similarity
            ).mean()
        )
        student_similarities.append(student_similarity.mean())
        teacher_similarities.append(teacher_similarity.mean())
    return (
        torch.stack(losses).mean(),
        torch.stack(student_similarities).mean(),
        torch.stack(teacher_similarities).mean(),
    )


def cosine_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:

    if student_tokens.shape != teacher_tokens.shape or student_tokens.ndim != 3:
        raise ValueError("student and teacher tokens must share shape [batch, patches, dim]")
    selected_indices = selected_indices.long().reshape(-1)
    if selected_indices.numel() == 0:
        return student_tokens.sum() * 0.0
    student = student_tokens.reshape(-1, student_tokens.shape[-1])[selected_indices]
    teacher = teacher_tokens.reshape(-1, teacher_tokens.shape[-1])[selected_indices]
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def sampled_nonlesion_indices(
    lesion_masks: torch.Tensor,
    *,
    per_slice: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if lesion_masks.ndim != 2:
        raise ValueError("lesion_masks must have shape [batch, patches]")
    if per_slice <= 0:
        raise ValueError("per_slice must be positive")
    batch, patches = lesion_masks.shape
    selected: list[torch.Tensor] = []
    for batch_index in range(batch):
        local = torch.nonzero(~lesion_masks[batch_index].bool(), as_tuple=False).flatten()
        if local.numel() > per_slice:
            if generator is None:
                positions = torch.linspace(
                    0, local.numel() - 1, per_slice, device=local.device
                ).round().long()
                local = local[positions]
            else:
                order = torch.randperm(
                    local.numel(), device=local.device, generator=generator
                )[:per_slice]
                local = local[order]
        selected.append(batch_index * patches + local)
    return torch.cat(selected) if selected else lesion_masks.new_empty(0, dtype=torch.long)


def balanced_lung_status_indices(
    side_masks: torch.Tensor,
    lesion_masks: torch.Tensor,
    lung_classes: torch.Tensor,
    *,
    per_status: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if side_masks.shape != lesion_masks.shape or side_masks.ndim != 2:
        raise ValueError("side_masks and lesion_masks must share shape [batch, patches]")
    if lung_classes.shape != (side_masks.shape[0],):
        raise ValueError("lung_classes must have one value per image")
    if per_status <= 0:
        raise ValueError("per_status must be positive")
    if bool((lesion_masks.bool() & ~side_masks.bool()).any()):
        raise ValueError("Every lesion patch must belong to its side-lung mask")

    batch, patches = side_masks.shape
    indices: list[torch.Tensor] = []
    statuses: list[torch.Tensor] = []
    parents: list[torch.Tensor] = []
    for batch_index in range(batch):
        side = side_masks[batch_index].bool()
        lesion = lesion_masks[batch_index].bool()
        groups = (side & ~lesion, lesion)
        for status, local_mask in enumerate(groups):
            local = torch.nonzero(local_mask, as_tuple=False).flatten()
            if local.numel() == 0:
                raise ValueError(f"Image {batch_index} has no patches for status={status}")
            if local.numel() > per_status:
                if generator is None:
                    positions = torch.linspace(
                        0, local.numel() - 1, per_status, device=local.device
                    ).round().long()
                    local = local[positions]
                else:
                    order = torch.randperm(
                        local.numel(), device=local.device, generator=generator
                    )[:per_status]
                    local = local[order]
            indices.append(batch_index * patches + local)
            statuses.append(torch.full_like(local, status))
            parents.append(torch.full_like(local, int(lung_classes[batch_index].item())))
    return torch.cat(indices), torch.cat(parents), torch.cat(statuses)
