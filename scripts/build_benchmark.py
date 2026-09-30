from __future__ import annotations

import argparse
import json
from pathlib import Path

from purevision.benchmark import build_questions, build_rrg_cases, normalized_test_records
from purevision.dataset_contract import display_catalog, load_contract, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phenotype-questions", type=int, required=True)
    parser.add_argument("--grounding-questions", type=int, required=True)
    parser.add_argument("--rrg-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.is_relative_to(repo):
        raise ValueError("评测实例包含数据，输出目录不得位于代码仓库内")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("评测输出目录非空，不覆盖历史记录")
    contract = load_contract(args.dataset_contract)
    records = normalized_test_records(contract)
    questions, counts = build_questions(
        contract, records,
        phenotype_target=args.phenotype_questions,
        grounding_target=args.grounding_questions,
        seed=args.seed,
    )
    rrg_cases = build_rrg_cases(contract, records, args.rrg_cases, args.seed + 1000)
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "reference.jsonl"
    with reference.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    question_path = output / "questions.jsonl"
    with question_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    rrg_path = output / "rrg_reference.jsonl"
    with rrg_path.open("w", encoding="utf-8") as handle:
        for record in rrg_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    dataset = contract["dataset"]
    protocol = {
        "数据集ID": dataset["dataset_id"],
        "发布版": dataset["release"],
        "来源根目录": dataset["source_root"],
        "标签来源": dataset["label_provenance_zh"],
        "划分清单": dataset["split_manifest"],
        "划分清单SHA256": dataset["split_manifest_sha256"],
        "测试参考清单SHA256": sha256(reference),
        "题目清单SHA256": sha256(question_path),
        "RRG参考清单SHA256": sha256(rrg_path),
        "构造规范SHA256": sha256(args.dataset_contract),
        "划分": "患者级 train/val/test；仅从 test 出题",
        "类别中英对照": display_catalog(contract),
        "目标题数": {"grounding": args.grounding_questions, "每个表型": args.phenotype_questions, "RRG": args.rrg_cases},
        "实际题数": {**counts, "RRG": len(rrg_cases)},
        "随机种子": args.seed,
        "方法说明": "病灶 mask 仅用于测试题参考与单格筛选，不输入模型；本阶段不运行原始预训练模型、标准未修改推理、mask 条件池化、监督质心评估或零样本分类，也不将 t-SNE 图间隙当作嵌入距离。",
    }
    (output / "协议_ZH.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"输出目录": str(output), "实际题数": {**counts, "RRG": len(rrg_cases)}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
