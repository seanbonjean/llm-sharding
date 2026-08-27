"""Plot single-device long-context prefill results using CSV and matplotlib.

Relative paths are resolved from the repository root, independently of the working directory.
使用 CSV 和 matplotlib 绘制单设备长上下文结果；相对路径以项目根目录为基准。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = (
    "results/profiling/long_context_prefill/20260827-231615-zmq7xa6z/"
    "long_context_prefill_latency-NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super-6layers.csv"
)


def plot_long_context_prefill(csv_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Plot completed measurements from a full or partially finished profiling run.

    Args:
        csv_path: Profiling CSV; a relative path is based on PROJECT_ROOT.
            测量 CSV；相对路径以项目根目录为基准，支持未完成运行的结果。
        output_path: Optional image path, also relative to PROJECT_ROOT when not absolute.
            Defaults to the CSV basename plus .png.
            可选图片路径，相对路径同样以项目根目录为基准；默认与 CSV 同名，后缀为 .png。
    """
    source = Path(csv_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    groups: dict[tuple[bool, int], list[float]] = {}
    full_points: set[tuple[bool, int]] = set()
    completed_ids: set[int] = set()
    stops: list[tuple[int, str]] = []
    invalid_rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "attempt_id", "phase", "status", "input_token_count", "include_lm_head",
            "prefill_latency_per_layer_ms", "is_full_context",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing long-context profiling columns")
        for row in reader:
            try:
                attempt_id = int(row["attempt_id"])
                if row["status"] in {"oom", "context_limit"}:
                    completed_ids.add(attempt_id)
                    if row["input_token_count"]:
                        stops.append((int(row["input_token_count"]), row["status"]))
                    continue
                if row["status"] != "success":
                    continue
                token_count = int(row["input_token_count"])
                latency = float(row["prefill_latency_per_layer_ms"])
                if token_count < 1 or not math.isfinite(latency) or latency < 0:
                    raise ValueError("Invalid numeric measurement")
                if row["include_lm_head"] not in {"True", "False"}:
                    raise ValueError("Invalid include_lm_head")
                completed_ids.add(attempt_id)
                if row["phase"] != "measure":
                    continue
                key = (row["include_lm_head"] == "True", token_count)
                groups.setdefault(key, []).append(latency)
                if row["is_full_context"] == "True":
                    full_points.add(key)
            except (ValueError, TypeError, KeyError):
                invalid_rows += 1
    if not groups:
        raise ValueError("CSV contains no completed measured trials to plot")
    if invalid_rows:
        print(f"[PROFILE] skipped {invalid_rows} incomplete/invalid CSV rows")
    metadata: dict[str, Any] = {}
    try:
        candidate = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            metadata = candidate
    except (OSError, ValueError):
        pass

    # Load the plotting backend only when rendering; imports and CLI help remain lightweight.
    # 仅在实际绘图时加载后端，使模块导入与命令行帮助保持轻量。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        for include_lm_head in sorted({key[0] for key in groups}):
            lengths = sorted(key[1] for key in groups if key[0] == include_lm_head)
            samples = [groups[(include_lm_head, length)] for length in lengths]
            medians = [statistics.median(values) for values in samples]
            label = "Input to first token (LM head)" if include_lm_head else "Input to hidden state"
            axis.plot(lengths, medians, marker="o", label=label)
            axis.fill_between(
                lengths, [min(values) for values in samples], [max(values) for values in samples], alpha=0.15,
            )
            for length, median in zip(lengths, medians):
                if (include_lm_head, length) in full_points:
                    axis.annotate("full context", (length, median), xytext=(4, 8), textcoords="offset points", fontsize=8)
        for length, reason in stops:
            axis.axvline(length, linestyle="--", color="gray", label=f"{reason}: {length} tokens")
        pending = metadata.get("pending_attempt") or {}
        if isinstance(pending, dict) and pending and pending.get("attempt_id") not in completed_ids:
            axis.text(
                0.01, 0.99,
                f"Unfinished attempt: {pending.get('context_length')} {pending.get('context_length_unit')}; cause unconfirmed",
                transform=axis.transAxes, va="top", fontsize=8,
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Actual model input tokens (chat template included)")
        axis.set_ylabel("Input-to-output latency / loaded layers (ms)")
        axis.set_title("Long-context prefill latency: median and min/max")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        destination = Path(output_path) if output_path is not None else source.with_suffix(".png")
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> None:
    """Parse plotting arguments and print the saved figure path.

    Args:
        argv: Optional argument list; None reads the process command line.
            可选参数列表；None 表示读取当前进程的命令行参数。
    """
    parser = argparse.ArgumentParser(description="Plot single-device long-context prefill latency from CSV.")
    parser.add_argument(
        "csv_path", nargs="?", default=DEFAULT_CSV_PATH,
        help="Input CSV; relative paths use the repository root. Defaults to DEFAULT_CSV_PATH in this script.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image; relative paths use the repository root. Defaults to the CSV basename plus .png.",
    )
    arguments = parser.parse_args(argv)
    print(plot_long_context_prefill(arguments.csv_path, output_path=arguments.output))


if __name__ == "__main__":
    main()
