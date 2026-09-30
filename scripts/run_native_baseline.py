from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
import yaml
from PIL import Image

from purevision.protocol import (
    sha256_file,
    validate_dataset_protocol,
    validate_weight_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 MedGemma 1.5 原始预训练权重的标准未修改推理 baseline"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--instruction")
    prompt.add_argument("--instruction-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--sample-id")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = validate_dataset_protocol(config, verify_files=True)
    weights = validate_weight_protocol(
        config,
        required_roles=("backbone_config", "backbone_index"),
        verify_files=True,
    )
    instruction = (
        args.instruction
        if args.instruction is not None
        else args.instruction_file.read_text(encoding="utf-8")
    )
    runtime = config.get("runtime", {})
    device = torch.device(runtime.get("device", "cuda:0"))
    dtype_name = str(runtime.get("dtype", "bfloat16"))
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported runtime dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = Path(config["paths"]["model"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=str(runtime.get("attention_implementation", "sdpa")),
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False).eval()
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device, dtype=dtype)
    input_length = int(inputs["input_ids"].shape[-1])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated_text = processor.decode(
        generation[0][input_length:], skip_special_tokens=True
    ).strip()
    peak_gpu_mib = (
        round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2)
        if device.type == "cuda"
        else None
    )
    record = {
        "schema_version": 1,
        "记录类型": "MedGemma 1.5 标准未修改推理 baseline 结果",
        "实验编号": str(config.get("experiment_id", "未设置")),
        "dataset_id": dataset["dataset_id"],
        "dataset_release": dataset["release"],
        "source_root": dataset["source_root"],
        "label_provenance_zh": dataset["label_provenance_zh"],
        "split_manifest": dataset["split_manifest"],
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "sample_id": args.sample_id,
        "sample_split": args.split,
        "类别中英对照": config.get("display_names_zh_en"),
        "权重协议": weights,
        "已加载权重角色": ["backbone_config", "backbone_index"],
        "流程说明_zh": (
            "使用 MedGemma 1.5 原始预训练权重执行标准未修改图像推理；不加载 PureEyes "
            "表型或解剖视觉编码器 checkpoint，不使用 mask-conditioned pooling、共享对齐器、"
            "病灶 patch 选择、soft semantic fusion、监督质心评估或 zero-shot subtype 分类。"
        ),
        "inference_masks_used": False,
        "purevision_checkpoints_used": False,
        "image_path": str(args.image.resolve()),
        "image_sha256": sha256_file(args.image),
        "instruction": instruction,
        "generated_text": generated_text,
        "runtime": {
            "device": str(device),
            "dtype": dtype_name,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_mib": peak_gpu_mib,
        },
    }
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"baseline 结果已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
