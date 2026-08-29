"""Plot per-Transformer-layer latency from a fixed-length long-context prefill run.

Run with ``python analysis/long_context_prefill_per_layer_fig.py [result directory or CSV]``.
When the argument is omitted, ``DEFAULT_RESULT_DIRECTORY`` selects the experiment run.
Relative paths are resolved from the repository root.
绘制固定长文本输入下各 Transformer 层的 prefill 时延；省略参数时由
``DEFAULT_RESULT_DIRECTORY`` 指定实验目录，相对路径以项目根目录为基准。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIRECTORY = (
    "results/profiling/long_context_prefill_per_layer/20260829-201056-e1x4ob6_"
)


def _resolve_per_layer_csv_path(result_path: str | Path) -> Path:
    """Resolve a per-layer profiling CSV from either its file or run directory.

    Args:
        result_path: Per-layer CSV or experiment run directory. Relative paths are based on
            PROJECT_ROOT. 逐层 CSV 或实验运行目录；相对路径以项目根目录为基准。

    Returns:
        The explicit CSV path or the only matching per-layer CSV in the run directory.
        显式 CSV 路径，或实验目录中唯一匹配的逐层 CSV 路径。
    """
    source = Path(result_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError(f"Per-layer result file must be a CSV: {source}")
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Per-layer result path does not exist: {source}")
    candidates = sorted(source.glob("long_context_prefill_per_layer_latency-*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No per-layer profiling CSV found in result directory: {source}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple per-layer profiling CSV files found in result directory: {source}"
        )
    return candidates[0]


def _short_device_name(source: Path, metadata: dict[str, Any]) -> str:
    """Build a concise device label from persisted result information.

    Args:
        source: Profiling CSV used as a filename fallback. 用作文件名回退来源的实验 CSV。
        metadata: Optional same-stem JSON object. 可选的同名 JSON 元数据。

    Returns:
        Compact device name, including a recorded Orin memory variant when available.
        简短设备名；若结果记录了 Orin 内存版本则包含该版本。
    """
    recorded_name = str(metadata.get("device_label") or source.stem)
    recorded_name = re.sub(
        r"^long_context_prefill_per_layer_latency-(.*)-[0-9]+layers-[0-9]+words$",
        r"\1", recorded_name,
    )
    recorded_name = " ".join(recorded_name.replace("_", " ").split())
    device_name = re.sub(r"^NVIDIA\s+", "", recorded_name, flags=re.IGNORECASE)
    model_name = metadata.get("jetson_model")
    if not isinstance(model_name, str):
        model_name = recorded_name
    orin_model = re.search(r"\borin\s+(nano|nx)\b", model_name, re.IGNORECASE)
    if orin_model:
        model_family = "Nano" if orin_model.group(1).lower() == "nano" else "NX"
        device_name = f"Jetson Orin {model_family}"
        valid_variants = ("4G", "8G") if model_family == "Nano" else ("8G", "16G")
        variant = metadata.get("jetson_memory_variant")
        if variant not in valid_variants:
            capacity_match = re.search(r"\b(4|8|16)\s*G(?:i?B)?\b", recorded_name, re.IGNORECASE)
            variant = f"{capacity_match.group(1)}G" if capacity_match else None
        if variant in valid_variants:
            device_name = f"{device_name} {variant}"
    return device_name


def plot_long_context_prefill_per_layer(
    csv_path: str | Path, output_path: str | Path | None = None,
) -> Path:
    """Plot median and min/max latency for every loaded Transformer layer.

    Args:
        csv_path: Per-layer profiling CSV or its experiment run directory; a relative path is
            based on PROJECT_ROOT. 逐层测量 CSV 或其实验运行目录；相对路径以项目根目录为基准，
            支持部分完成的运行。
        output_path: Optional image path, also relative to PROJECT_ROOT when not absolute.
            Defaults to the CSV basename plus .png.
            可选图片路径；相对路径以项目根目录为基准，默认与 CSV 同名并使用 .png 后缀。
    """
    source = _resolve_per_layer_csv_path(csv_path)
    groups: dict[int, list[float]] = {}
    completed_attempt_ids: set[int] = set()
    loaded_layer_counts: set[int] = set()
    input_token_counts: set[int] = set()
    context_word_counts: set[int] = set()
    failed_layers: list[tuple[int, str]] = []
    invalid_rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "attempt_id", "phase", "status", "context_word_count", "input_token_count",
            "loaded_layer_count", "layer_index", "layer_latency_ms",
        }
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing per-layer long-context profiling columns")
        for row in reader:
            try:
                attempt_id = int(row["attempt_id"])
                if row["status"] in {"oom", "context_limit"}:
                    completed_attempt_ids.add(attempt_id)
                    if row["layer_index"]:
                        failed_layers.append((int(row["layer_index"]), row["status"]))
                    continue
                if row["status"] != "success":
                    continue
                loaded_layer_count = int(row["loaded_layer_count"])
                input_token_count = int(row["input_token_count"])
                context_word_count = int(row["context_word_count"])
                layer_index = int(row["layer_index"])
                layer_latency_ms = float(row["layer_latency_ms"])
                if (
                    loaded_layer_count < 1 or input_token_count < 1 or context_word_count < 1
                    or layer_index < 0 or layer_index >= loaded_layer_count
                    or not math.isfinite(layer_latency_ms) or layer_latency_ms < 0
                ):
                    raise ValueError("Invalid numeric per-layer measurement")
                completed_attempt_ids.add(attempt_id)
                loaded_layer_counts.add(loaded_layer_count)
                input_token_counts.add(input_token_count)
                context_word_counts.add(context_word_count)
                if row["phase"] == "measure":
                    groups.setdefault(layer_index, []).append(layer_latency_ms)
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
    if not groups:
        raise ValueError("CSV contains no completed measured per-layer trials to plot")
    if len(loaded_layer_counts) != 1 or len(input_token_counts) != 1 or len(context_word_counts) != 1:
        raise ValueError("CSV mixes incompatible per-layer profiling conditions")
    loaded_layer_count = next(iter(loaded_layer_counts))
    if set(groups) != set(range(loaded_layer_count)):
        raise ValueError("CSV does not contain measured latency for every loaded layer")
    if invalid_rows:
        print(f"[PROFILE] skipped {invalid_rows} incomplete/invalid per-layer CSV rows")

    metadata: dict[str, Any] = {}
    try:
        candidate = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            metadata = candidate
    except (OSError, ValueError):
        pass

    # Load matplotlib only when rendering so imports and CLI help remain lightweight.
    # 仅在实际绘图时加载 matplotlib，使模块导入和命令行帮助保持轻量。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_indices = sorted(groups)
    samples = [groups[layer_index] for layer_index in layer_indices]
    medians = [statistics.median(values) for values in samples]
    context_word_count = next(iter(context_word_counts))
    input_token_count = next(iter(input_token_counts))
    figure, axis = plt.subplots(figsize=(10, 5))
    try:
        axis.plot(
            layer_indices, medians, marker="o",
            label=f"{_short_device_name(source, metadata)} | {loaded_layer_count} layers",
        )
        axis.fill_between(
            layer_indices,
            [min(values) for values in samples],
            [max(values) for values in samples],
            alpha=0.15,
        )
        for layer_index, reason in failed_layers:
            axis.axvline(
                layer_index, linestyle="--", color="gray",
                label=f"{reason} at layer {layer_index}",
            )
        pending_attempt = metadata.get("pending_attempt") or {}
        if (
            isinstance(pending_attempt, dict) and pending_attempt
            and pending_attempt.get("attempt_id") not in completed_attempt_ids
        ):
            axis.text(
                0.01, 0.99,
                f"Unfinished attempt: {pending_attempt.get('attempt_id')}; cause unconfirmed",
                transform=axis.transAxes, va="top", fontsize=8,
            )
        axis.set_xticks(layer_indices)
        axis.set_xlabel("Transformer layer index (zero-based)")
        axis.set_ylabel("Layer prefill latency (ms)")
        axis.set_title(
            f"Per-layer prefill latency: {context_word_count} context words / "
            f"{input_token_count} input tokens"
        )
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
    parser = argparse.ArgumentParser(description="Plot fixed-input per-layer prefill latency.")
    parser.add_argument(
        "result_path", nargs="?", default=DEFAULT_RESULT_DIRECTORY,
        help=(
            "Experiment directory or CSV; relative paths use the repository root. "
            "Defaults to DEFAULT_RESULT_DIRECTORY in this script."
        ),
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image; relative paths use the repository root. Defaults beside the CSV.",
    )
    arguments = parser.parse_args(argv)
    print(plot_long_context_prefill_per_layer(arguments.result_path, output_path=arguments.output))


if __name__ == "__main__":
    main()
