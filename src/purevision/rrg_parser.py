from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence


PARSER_VERSION = "dataset_contract_rrg_gpt6_astra_v2"
SYSTEM_PROMPT = (
    "你是放射学报告的结构化解析器。只从报告明确陈述的内容提取 4x4 网格位置和表型。"
    "只能使用给定机器可读 ID。缺失、含糊、冲突或无法映射的字段填写 null。"
    "不要依据医学常识、病例标签、图像或其他模型输出来补全。"
    "grid_cell 只有在报告明确给出 4x4 单元时才能填写。"
)
GRID_CELLS = tuple(f"r{row}c{column}" for row in range(1, 5) for column in range(1, 5))


def label_groups_from_config(config: Mapping[str, Any], groups: Sequence[str]) -> dict[str, tuple[str, ...]]:
    targets = config["text_targets"]
    result: dict[str, tuple[str, ...]] = {}
    for name in groups:
        labels = tuple(str(item["label"]) for item in targets[name])
        if not labels or len(set(labels)) != len(labels):
            raise ValueError(f"候选类别为空或重复：{name}")
        result[name] = labels
    return result


def parser_schema(groups: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    properties = {"grid_cell": {"type": ["string", "null"], "enum": [*GRID_CELLS, None]}}
    for name, labels in groups.items():
        properties[name] = {"type": ["string", "null"], "enum": [*labels, None]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def normalize_parsed_labels(
    payload: Any, groups: Mapping[str, Sequence[str]]
) -> tuple[dict[str, str | None], list[str]]:
    schema = parser_schema(groups)
    source = payload if isinstance(payload, dict) else {}
    output: dict[str, str | None] = {}
    invalid: list[str] = []
    for name, property_schema in schema["properties"].items():
        value = source.get(name)
        if value is None:
            output[name] = None
        elif isinstance(value, str) and value in property_schema["enum"]:
            output[name] = value
        else:
            output[name] = None
            invalid.append(name)
    invalid.extend(sorted(set(source) - set(schema["properties"])))
    return output, invalid


def parse_rrg_report(
    report: str,
    groups: Mapping[str, Sequence[str]],
    *,
    client: Any = None,
    model: str = "gpt-6-astra",
) -> dict[str, Any]:
    if not report.strip():
        raise ValueError("报告正文不能为空")
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("请先设置 OPENAI_API_KEY")
        from openai import OpenAI

        client = OpenAI()
    schema = parser_schema(groups)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "purevision_rrg_labels",
                "schema": schema,
                "strict": True,
            }
        },
        store=False,
    )
    raw_text = str(getattr(response, "output_text", "") or "")
    try:
        payload = json.loads(raw_text)
        status = "已解析"
    except json.JSONDecodeError:
        payload = None
        status = "无效输出"
    labels, invalid = normalize_parsed_labels(payload, groups)
    if status == "无效输出":
        invalid = list(labels)
    schema_bytes = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "解析状态": status,
        "parser_version": PARSER_VERSION,
        "model": model,
        "response_id": getattr(response, "id", None),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "raw_parser_output": raw_text,
        "parsed_labels": labels,
        "invalid_fields": invalid,
    }
