from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from purevision.protocol import write_dataset_companion_zh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中文数据集归属与固定划分侧录")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = write_dataset_companion_zh(args.output_dir, config)
    print(f"数据集侧录已写入：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
