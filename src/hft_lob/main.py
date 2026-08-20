"""hft_lob 流水线 CLI 入口：按阶段编排数据处理、训练与回测。"""

from __future__ import annotations

from typing import Any


def parse_args() -> Any:
    """解析命令行运行身份参数（--experiment_id / --stages / --dataset / --model）。

    Returns:
        解析后的命令行参数对象。
    """
    raise NotImplementedError("parse_args not implemented")


def main() -> None:
    """读取实验配置并按 stages 依次执行各处理阶段。"""
    raise NotImplementedError("main not implemented")
