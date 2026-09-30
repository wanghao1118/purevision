from __future__ import annotations

import argparse
import json
from pathlib import Path

from purevision.benchmark import score_rrg, score_vqa
from purevision.dataset_contract import display_catalog, load_contract, sha256


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mode", choices=("vqa", "rrg"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖既有评测记录")
    contract = load_contract(args.dataset_contract)
    references = read_jsonl(args.reference)
    predictions = read_jsonl(args.predictions)
    groups = [dimension["id"] for dimension in contract["phenotypes"]]
    if args.mode == "vqa":
        if args.questions is None:
            raise ValueError("VQA 评分需要 --questions")
        keys = [row["question_id"] for row in predictions]
        if len(keys) != len(set(keys)):
            raise ValueError("预测题号重复")
        questions = read_jsonl(args.questions)
        metrics = score_vqa(
            questions, {row["question_id"]: row.get("answer") for row in predictions}, groups
        )
    else:
        keys = [row["sample_id"] for row in predictions]
        if len(keys) != len(set(keys)):
            raise ValueError("预测病例号重复")
        reference_labels = [
            {"sample_id": row["sample_id"], "grid_cell": row["grounding_cell"], **row["phenotypes"]}
            for row in references
        ]
        prediction_labels = {
            row["sample_id"]: row.get("parsed_labels", {}) for row in predictions
        }
        metrics = score_rrg(reference_labels, prediction_labels, ["grid_cell", *groups])
    dataset = contract["dataset"]
    output = {
        "数据集ID": dataset["dataset_id"],
        "发布版": dataset["release"],
        "来源根目录": dataset["source_root"],
        "标签来源": dataset["label_provenance_zh"],
        "划分清单": dataset["split_manifest"],
        "划分清单SHA256": dataset["split_manifest_sha256"],
        "构造规范SHA256": sha256(args.dataset_contract),
        "参考清单SHA256": sha256(args.reference),
        "预测清单SHA256": sha256(args.predictions),
        "题目清单SHA256": sha256(args.questions) if args.questions else None,
        "评测任务": args.mode,
        "类别中英对照": display_catalog(contract),
        "评分规则": "缺失、无效和冲突预测均计为错误；定位按 4×4 单元精确匹配；VQA 表型对维度求宏平均，RRG 对每个维度先按参考类别均衡再对维度求平均。",
        "评测指标": metrics,
        "方法边界": "评分使用冻结参考，不向模型输入 mask；此记录不代表原始预训练权重的标准未修改推理、训练期 mask 条件池化、监督质心评估或零样本分类，也不以 t-SNE 间隙量化嵌入距离。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
