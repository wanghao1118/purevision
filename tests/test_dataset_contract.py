from __future__ import annotations

import pytest
from PIL import Image

from purevision.benchmark import build_questions, build_rrg_cases, grid_cell, mapped_label, score_rrg, score_vqa
from purevision.dataset_contract import label_groups, validate_contract, validate_target_bank


def contract():
    return {
        "schema_version": 1,
        "dataset": {
            "dataset_id": "fixture", "release": "v1", "source_root": "/source",
            "disease_display_zh_en": "病灶 / lesion",
            "label_provenance_zh": "测试标签", "split_manifest": "/splits.json",
            "split_manifest_sha256": "a" * 64,
        },
        "anatomy": {
            "mask_encoding": "independent_channel_bitset",
            "lesion_labels": ["lesion"],
            "choices": [
                {"id": "normal", "text": "Normal", "display_zh_en": "正常 / normal"},
                {"id": "lesion", "text": "Lesion", "display_zh_en": "病灶 / lesion"},
            ],
        },
        "phenotypes": [
            {
                "id": "grade", "display_zh_en": "等级 / grade", "kind": "ordinal",
                "question_zh_en": "等级是什么？ / What is the grade?",
                "source_field": "dimensions.grade.label",
                "choices": [
                    {"id": f"g{i}", "text": f"Grade {i}", "display_zh_en": f"{i} 级 / grade {i}"}
                    for i in range(4)
                ],
            },
        ],
        "grounding": {"grid_size": 4, "benchmark_grounding_eligible": False},
    }


def test_contract_drives_group_order_and_checkpoint_guard():
    data = contract()
    validate_contract(data)
    groups = label_groups(data)
    assert groups["grade"] == ("g0", "g1", "g2", "g3")
    validate_target_bank(data, groups)
    with pytest.raises(ValueError, match="不一致"):
        validate_target_bank(data, {"anatomy": groups["anatomy"], "grade": tuple(reversed(groups["grade"]))})


def test_label_mapping_uses_dataset_dimensions():
    data = contract()
    assert mapped_label({"dimensions": {"grade": {"label": "g2"}}}, data["phenotypes"][0]) == "g2"
    with pytest.raises(ValueError, match="不在候选词表"):
        mapped_label({"dimensions": {"grade": {"label": "g9"}}}, data["phenotypes"][0])


def test_single_cell_gate_rejects_crossing_mask(tmp_path):
    path = tmp_path / "mask.png"
    image = Image.new("L", (8, 8), 0)
    image.putpixel((1, 1), 255)
    image.save(path)
    assert grid_cell(path, 4) == "r1c1"
    image.putpixel((4, 1), 255)
    image.save(path)
    assert grid_cell(path, 4) is None
    assert grid_cell(path, 4, require_single_cell=False) == "r1c1"
    image.putpixel((5, 1), 255)
    image.save(path)
    assert grid_cell(path, 4, require_single_cell=False) == "r1c3"


def test_disabled_grounding_and_four_choice_grade_runs():
    data = contract()
    records = [
        {"sample_id": str(i), "grounding_cell": None, "phenotypes": {"grade": f"g{i % 4}"}}
        for i in range(8)
    ]
    with pytest.raises(ValueError, match="未启用定位参考掩码"):
        build_questions(data, records, phenotype_target=4, grounding_target=1, seed=1)
    questions, counts = build_questions(data, records, phenotype_target=4, grounding_target=0, seed=1)
    assert counts == {"grounding": 0, "grade": 4}
    assert all(len(set(item["options"])) == 4 and item["answer"] in item["options"] for item in questions)
    assert all(" / " in item["question_zh_en"] for item in questions)
    assert len(build_rrg_cases(data, records, 6, 1)) == 6


def test_proxy_grounding_and_native_two_choice_dimension():
    data = contract()
    data["source"] = {"lesion_mask_field": "lesion_roi_proxy_path"}
    data["grounding"].update({"benchmark_grounding_eligible": True, "require_single_cell": False})
    data["phenotypes"][0]["choices"] = data["phenotypes"][0]["choices"][:2]
    validate_contract(data)
    records = [
        {"sample_id": str(i), "grounding_cell": f"r1c{i % 2 + 1}", "phenotypes": {"grade": f"g{i % 2}"}}
        for i in range(8)
    ]
    questions, counts = build_questions(data, records, phenotype_target=4, grounding_target=4, seed=1)
    assert counts == {"grounding": 4, "grade": 4}
    assert all(len(item["options"]) == 2 for item in questions if item["task"] == "grade")
    assert all(len(item["options"]) == 4 for item in questions if item["task"] == "grounding")
    assert len(build_rrg_cases(data, records, 6, 1)) == 6


def test_grounding_contract_requires_reference_mask_source():
    data = contract()
    data["grounding"]["benchmark_grounding_eligible"] = True
    with pytest.raises(ValueError, match="参考掩码字段"):
        validate_contract(data)


def test_missing_predictions_count_as_wrong():
    questions = [
        {"question_id": "1:grounding", "task": "grounding", "answer": "r1c1"},
        {"question_id": "1:grade", "task": "grade", "answer": "g2"},
    ]
    vqa = score_vqa(questions, {"1:grounding": "r1c1"}, ["grade"])
    assert vqa["grounding_accuracy"] == 1.0
    assert vqa["phenotype_macro_accuracy"] == 0.0
    rrg = score_rrg(
        [{"sample_id": "1", "grade": "g0"}, {"sample_id": "2", "grade": "g1"}],
        {"1": {"grade": "g0"}}, ["grade"],
    )
    assert rrg["phenotype_macro_accuracy"] == 0.5
    assert rrg["grounding_accuracy"] is None


def test_rrg_grounding_is_exact_accuracy_not_class_balanced():
    references = [
        {"sample_id": "1", "grid_cell": "r1c1", "grade": "g0"},
        {"sample_id": "2", "grid_cell": "r1c1", "grade": "g0"},
        {"sample_id": "3", "grid_cell": "r2c2", "grade": "g1"},
    ]
    predictions = {
        "1": {"grid_cell": "r1c1", "grade": "g0"},
        "2": {"grid_cell": "r1c1", "grade": "g0"},
    }
    score = score_rrg(references, predictions, ["grid_cell", "grade"])
    assert score["grounding_accuracy"] == 2 / 3
    assert score["phenotype_macro_accuracy"] == 0.5
