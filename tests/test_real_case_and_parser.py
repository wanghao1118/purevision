from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from purevision.rrg_parser import normalize_parsed_labels, parse_rrg_report, parser_schema
from run_real_case import verify_case


ROOT = Path(__file__).resolve().parents[1]


def test_real_lidc_case_matches_frozen_image_mask_and_grid() -> None:
    case_path = ROOT / "examples" / "lidc_case_0079" / "case_zh.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "07_medgemma15_reproduction_smoke.yaml").read_text(encoding="utf-8"))
    fixture = verify_case(case, config, case_path.parent)
    assert fixture["grid_cell"] == "r4c3"
    assert fixture["lesion_pixels"] > 0
    assert fixture["single_cell"] is False
    assert fixture["occupied_cells"] == [
        {"grid_cell": "r3c3", "pixels": 66},
        {"grid_cell": "r4c3", "pixels": 145},
    ]


def test_parser_restricts_grid_and_labels_without_guessing() -> None:
    groups = {"density": ("solid", "mixed_or_mostly_solid")}
    schema = parser_schema(groups)
    assert "r4c3" in schema["properties"]["grid_cell"]["enum"]
    assert "r5c1" not in schema["properties"]["grid_cell"]["enum"]
    labels, invalid = normalize_parsed_labels({"grid_cell": "r5c1", "density": "unknown"}, groups)
    assert labels == {"grid_cell": None, "density": None}
    assert invalid == ["grid_cell", "density"]


def test_parser_uses_structured_responses_and_preserves_missing_fields() -> None:
    class FakeResponses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(id="resp_test", output_text='{"grid_cell":null,"density":"solid"}')

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    parsed = parse_rrg_report("The nodule is solid.", {"density": ("solid", "mixed")}, client=client)
    assert responses.kwargs["model"] == "gpt-6-astra"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert parsed["parsed_labels"] == {"grid_cell": None, "density": "solid"}
    assert parsed["response_id"] == "resp_test"
