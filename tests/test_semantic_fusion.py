from __future__ import annotations

import torch

from purevision.decoder import replace_visual_embeddings
from purevision.semantic_fusion import (
    aggregate_candidate_scores,
    lesion_margin,
    mix_token_sequences,
    select_local_lesion_region,
    standardized_softmax,
)


def test_lesion_margin_uses_best_lesion_minus_best_other() -> None:
    similarities = torch.tensor(
        [[0.1, 0.7, 0.2, 0.5], [0.8, 0.3, 0.6, 0.4]]
    )
    result = lesion_margin(similarities, (1, 2), (0, 3))
    assert torch.allclose(result, torch.tensor([0.2, -0.2]))


def test_spatial_selection_uses_raw_softmax_and_stays_local() -> None:
    scores = torch.full((8, 8), -5.0)
    scores[3, 4] = 3.0
    scores[3, 5] = 2.0
    scores[4, 4] = 1.0
    result = select_local_lesion_region(
        scores.reshape(-1),
        grid_size=8,
        window_size=5,
        anchor_top_k=2,
        patch_count=3,
    )
    assert result.patch_indices == (3 * 8 + 4, 3 * 8 + 5, 4 * 8 + 4)
    expected = torch.softmax(torch.tensor([3.0, 2.0, 1.0]), dim=0)
    assert torch.allclose(torch.tensor(result.patch_weights), expected)
    row_start, row_stop, column_start, column_stop = result.window_bounds
    for index in result.patch_indices:
        row, column = divmod(index, 8)
        assert row_start <= row < row_stop
        assert column_start <= column < column_stop


def test_group_score_aggregation_and_standardized_softmax() -> None:
    similarities = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]], dtype=torch.float32
    )
    scores = aggregate_candidate_scores(similarities, (0, 1), (0.25, 0.75))
    assert torch.allclose(scores, torch.tensor([0.25, 1.5]))
    probabilities = standardized_softmax(scores, temperature=0.125)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[1] > probabilities[0]


def test_token_mixture_repeats_final_token_without_norm_rescaling() -> None:
    short = torch.tensor([[1.0, 0.0]])
    long = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    mixed = mix_token_sequences((short, long), (0.25, 0.75))
    expected = torch.tensor([[0.25, 1.5], [0.25, 3.0]])
    assert torch.allclose(mixed, expected)


def test_decoder_embedding_replacement_retains_native_then_semantic() -> None:
    prompt = torch.zeros((1, 7, 3))
    native = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    semantic = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    result = replace_visual_embeddings(
        prompt,
        native_positions=torch.tensor([1, 2]),
        native_tokens=native,
        semantic_positions=torch.tensor([3, 4]),
        semantic_tokens=semantic,
    )
    assert torch.equal(result[0, 1:3], native)
    assert torch.equal(result[0, 3:5], semantic)
    assert torch.equal(result[0, 0], torch.zeros(3))
