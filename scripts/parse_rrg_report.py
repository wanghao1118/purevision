from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from purevision.dataset_contract import display_catalog, label_groups, load_contract
from purevision.protocol import sha256_file, validate_dataset_protocol
from purevision.rrg_parser import label_groups_from_config, parse_rrg_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 GPT6-Astra 将自由文本放射学报告解析为固定标签")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--label-config", type=Path)
    parser.add_argument("--dataset-contract", type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else None
    if not args.dataset_contract and not args.label_config:
        raise ValueError("必须提供 --dataset-contract 或 --label-config")
    if config is None and not args.dataset_contract:
        raise ValueError("旧标签配置模式需要 --config")
    contract = load_contract(args.dataset_contract) if args.dataset_contract else None
    dataset = validate_dataset_protocol(config) if config else contract["dataset"]
    source = json.loads(args.result_json.read_text(encoding="utf-8"))
    for field, expected in (
        ("dataset_id", dataset["dataset_id"]),
        ("dataset_release", dataset["release"]),
        ("split_manifest_sha256", dataset["split_manifest_sha256"]),
    ):
        if source.get(field) != expected:
            raise ValueError(f"输入结果的 {field} 与固定数据协议不一致")
    if contract:
        if contract["dataset"]["dataset_id"] != dataset["dataset_id"]:
            raise ValueError("数据集规范与结果的数据集 ID 不一致")
        if contract["dataset"]["split_manifest_sha256"] != dataset["split_manifest_sha256"]:
            raise ValueError("数据集规范与结果的划分哈希不一致")
        groups = {name: values for name, values in label_groups(contract).items() if name != "anatomy"}
    else:
        label_config = yaml.safe_load(args.label_config.read_text(encoding="utf-8"))
        names = tuple(config["inference"]["phenotype_groups"])
        groups = label_groups_from_config(label_config, names)
    report = str(source["generated_text"])
    parsed = parse_rrg_report(report, groups, model=args.model)
    output = {
        "schema_version": 1,
        "记录类型": "GPT6-Astra 自由文本报告结构化解析结果",
        "实验编号": config["experiment_id"] if config else source.get("experiment_id"),
        "dataset_id": dataset["dataset_id"],
        "dataset_release": dataset["release"],
        "source_root": dataset["source_root"],
        "label_provenance_zh": dataset["label_provenance_zh"],
        "split_manifest": dataset["split_manifest"],
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "sample_id": source.get("sample_id"),
        "sample_split": source.get("sample_split"),
        "类别中英对照": display_catalog(contract) if contract else config.get("display_names_zh_en"),
        "source_result_sha256": sha256_file(args.result_json),
        "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "解析说明": "只提取报告明示的 4x4 网格及固定表型标签；null 和无效字段在后续确定性评分中计为错误。本模块不直接评分。",
        **parsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"解析结果已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
