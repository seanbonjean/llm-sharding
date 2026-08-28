"""Compare two devices on shared long-context prefill measurements.

Only the standard library and matplotlib are required; relative paths use the project root.
对齐两台设备的长上下文 prefill 测量；仅依赖标准库和 matplotlib，相对路径以项目根目录为准。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import textwrap
import warnings
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH_A = (
    "results/profiling/long_context_prefill/20260827-231615-zmq7xa6z/"
    "long_context_prefill_latency-NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super-6layers.csv"
)
DEFAULT_CSV_PATH_B = (
    "results/profiling/long_context_prefill/20260828-003453-_krn076g/"
    "long_context_prefill_latency-Quadro_RTX_5000-6layers.csv"
)
DEFAULT_OUTPUT_DIR = "results/profiling/long_context_prefill/comparisons"
RUN_FIELDS = ("sample_id", "model_type", "question_type", "context_length_unit", "loaded_layer_count")
METADATA_FIELDS = (
    "dtype", "model_layer_count", "max_position_embeddings", "timing_scope", "per_layer_definition",
    "attention_implementation", "attention_mask", "final_norm_included", "lm_head_resident", "warmup_count",
)


class PrefillPoint(NamedTuple):
    """One context prefix and its successful per-layer repeat latencies.

    input_token_count includes the template/question; word/char counts and is_full_context identify
    the prefix. latencies contains milliseconds per loaded layer, excluding warmup.
    保存含模板/问题的 token 数、前缀词数/字符数、完整原文标志，以及正式重复的层均摊毫秒数。
    """

    input_token_count: int
    context_word_count: int
    context_char_count: int
    is_full_context: bool
    latencies: list[float]


class PrefillRun(NamedTuple):
    """Validated input: source CSV, common settings, context-length points and optional JSON metadata.
    source 是 CSV 路径，settings 是实验条件，points 按原文长度索引，metadata 是可选 JSON 属性。
    """

    source: Path
    settings: dict[str, str]
    points: dict[int, PrefillPoint]
    metadata: dict[str, Any]


def _read_run(csv_path: str | Path, include_lm_head: bool) -> PrefillRun:
    """Read successful formal repeats for a single output endpoint.

    Args:
        csv_path: Input CSV; relative paths are resolved from PROJECT_ROOT. 输入 CSV，相对项目根目录。
        include_lm_head: Select first-token measurements if True, hidden states otherwise.
            True 选择 LM head 首 token 终点，False 选择 hidden states 终点。
    """
    source = Path(csv_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    settings: dict[str, str] = {}
    points: dict[int, PrefillPoint] = {}
    invalid_rows = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(RUN_FIELDS) | {
            "phase", "status", "include_lm_head", "context_length", "context_word_count",
            "context_char_count", "input_token_count", "prefill_latency_per_layer_ms", "is_full_context",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source.name}: missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("phase") != "measure" or row.get("status") != "success":
                continue
            if row.get("include_lm_head") not in {"True", "False"}:
                invalid_rows += 1
                continue
            if row["include_lm_head"] != str(include_lm_head):
                continue
            try:
                row_settings = {field: row[field] for field in RUN_FIELDS}
                context_length = int(row["context_length"])
                point = PrefillPoint(
                    int(row["input_token_count"]), int(row["context_word_count"]),
                    int(row["context_char_count"]), row["is_full_context"] == "True",
                    [float(row["prefill_latency_per_layer_ms"])],
                )
                if (not all(row_settings.values()) or int(row_settings["loaded_layer_count"]) < 1
                        or context_length < 1 or point.input_token_count < 1
                        or point.context_word_count < 0 or point.context_char_count < 1
                        or row["is_full_context"] not in {"True", "False"}
                        or row_settings["context_length_unit"] not in {"words", "characters"}
                        or not math.isfinite(point.latencies[0]) or point.latencies[0] < 0):
                    raise ValueError("Invalid measurement fields")
            except (ValueError, TypeError, KeyError):
                invalid_rows += 1
                continue
            if settings and settings != row_settings:
                mismatches = [field for field in RUN_FIELDS if settings[field] != row_settings[field]]
                raise ValueError(f"{source.name}: mixed experiment settings: {', '.join(mismatches)}")
            settings = row_settings
            if context_length in points:
                # The final tuple field contains repeat latencies; all earlier input fields must match.
                # 元组末项是各次时延；合并重复测量前，仅比较前面的全部输入属性。
                if points[context_length][:-1] != point[:-1]:
                    raise ValueError(f"{source.name}: inconsistent input at context_length={context_length}")
                points[context_length].latencies.extend(point.latencies)
            else:
                points[context_length] = point
    if invalid_rows:
        warnings.warn(f"{source.name}: skipped {invalid_rows} invalid measured rows", stacklevel=2)
    if not points:
        raise ValueError(f"{source.name}: no completed measured trials for include_lm_head={include_lm_head}")
    metadata: dict[str, Any] = {}
    try:
        candidate = json.loads(source.with_suffix(".json").read_text(encoding="utf-8-sig"))
        if not isinstance(candidate, dict):
            raise ValueError("JSON metadata must be an object")
        metadata = candidate
    except (OSError, ValueError):
        warnings.warn(f"{source.name}: JSON metadata unavailable; validation is limited to CSV fields", stacklevel=2)
    for field in RUN_FIELDS:
        metadata_field = "layer_num" if field == "loaded_layer_count" else field
        if metadata_field in metadata and str(metadata[metadata_field]) != settings[field]:
            raise ValueError(f"{source.name}: CSV/JSON mismatch for {field}")
    return PrefillRun(source, settings, points, metadata)


def _common_context_lengths(first: PrefillRun, second: PrefillRun) -> list[int]:
    """Validate workload compatibility and return exact shared successful context lengths.

    Args:
        first: First device's validated run. 第一台设备的测量及元数据。
        second: Second device's validated run. 第二台设备的测量及元数据。
    """
    for field in RUN_FIELDS:
        if first.settings[field] != second.settings[field]:
            raise ValueError(f"Incompatible {field}: {first.settings[field]!r} != {second.settings[field]!r}")
    unchecked_fields = []
    for field in METADATA_FIELDS:
        if field not in first.metadata or field not in second.metadata:
            unchecked_fields.append(field)
        elif first.metadata[field] != second.metadata[field]:
            raise ValueError(f"Incompatible metadata {field}: {first.metadata[field]!r} != {second.metadata[field]!r}")
    # Paths belong to different hosts; compare only their model-directory names.
    # 两份结果的路径属于不同机器，只比较权重目录名，兼容 Windows 和 Linux 的分隔符。
    model_names = [str(run.metadata.get("shards_path") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                   for run in (first, second)]
    if all(model_names):
        if model_names[0] != model_names[1]:
            raise ValueError(f"Incompatible model directories: {model_names[0]!r} != {model_names[1]!r}")
    else:
        unchecked_fields.append("shards_path")
    if unchecked_fields:
        warnings.warn(f"Metadata cannot verify: {', '.join(unchecked_fields)}", stacklevel=2)
    versions = [run.metadata.get("torch_version") for run in (first, second)]
    if all(versions) and versions[0] != versions[1]:
        warnings.warn(f"Different PyTorch versions: {versions[0]} vs {versions[1]}; environment may affect latency", stacklevel=2)

    # Intersect measured prefixes: do not interpolate a missing point or treat failed attempts as timings.
    # 仅保留双方成功测量的同长度前缀，不插值，不把失败或 pending 尝试作为时延数据。
    common_upper_bound = min(max(first.points), max(second.points))
    common_lengths = sorted(length for length in first.points.keys() & second.points.keys()
                            if length <= common_upper_bound)
    if not common_lengths:
        raise ValueError("No shared successful context lengths for the selected endpoint")
    for length in common_lengths:
        first_point, second_point = first.points[length], second.points[length]
        if first_point.input_token_count != second_point.input_token_count:
            raise ValueError(f"input_token_count mismatch at context_length={length}; check tokenizer and prompt")
        prefix_fields = ("context_word_count", "context_char_count", "is_full_context")
        if any(getattr(first_point, field) != getattr(second_point, field) for field in prefix_fields):
            raise ValueError(f"Context prefix mismatch at context_length={length}; check dataset and truncation")
    return common_lengths


def _run_directory_hash(csv_path: Path) -> str:
    """Extract a profiling run's directory suffix for the comparison image name.

    Args:
        csv_path: Resolved CSV path whose parent normally has the form YYYYMMDD-HHMMSS-hash.
            已定位的 CSV 路径，父目录通常命名为 YYYYMMDD-HHMMSS-hash。

    Preserve the entire suffix, including leading underscores. For other directory layouts, use
    the first eight SHA-256 hex digits of the absolute CSV path as a stable, filename-safe fallback.
    保留完整后缀及开头的下划线；其他目录结构使用 CSV 绝对路径的 SHA-256 前八位作为稳定回退标识。
    """
    match = re.fullmatch(r"[0-9]{8}-[0-9]{6}-([A-Za-z0-9_-]+)", csv_path.parent.name)
    if match:
        return match.group(1)
    return hashlib.sha256(str(csv_path.resolve()).encode("utf-8")).hexdigest()[:8]


def compare_long_context_prefill(
    csv_path_a: str | Path | None = None,
    csv_path_b: str | Path | None = None,
    output_path: str | Path | None = None,
    include_lm_head: bool = False,
    label_a: str | None = None,
    label_b: str | None = None,
) -> Path:
    """Plot two devices' median per-layer latencies on their common input points.

    Args:
        csv_path_a: First result CSV; None uses DEFAULT_CSV_PATH_A. Relative paths use PROJECT_ROOT.
            第一组 CSV；None 使用 DEFAULT_CSV_PATH_A，相对路径以项目根目录为准。
        csv_path_b: Second result CSV; None uses DEFAULT_CSV_PATH_B. Relative paths use PROJECT_ROOT.
            第二组 CSV；None 使用 DEFAULT_CSV_PATH_B，相对路径以项目根目录为准。
        output_path: Image destination, relative to PROJECT_ROOT when not absolute. Defaults to
            DEFAULT_OUTPUT_DIR/long_context_prefill_comparison-<run A hash>-<run B hash>.png.
            图片输出路径；默认保存到 DEFAULT_OUTPUT_DIR，并按输入顺序使用两组运行目录的 hash 命名。
        include_lm_head: True compares first-token trials; False compares hidden-state trials.
            两条线共同选择的终点：True 带 LM head，False 为 hidden states。
        label_a: Optional first legend nickname; defaults to the short recorded device name.
            第一条图例昵称；默认使用简短设备名，Orin 设备附带记录的内存版本。
        label_b: Optional second legend nickname with the same formatting. 第二条图例昵称，格式规则同上。
    """
    if not isinstance(include_lm_head, bool):
        raise TypeError("include_lm_head must be bool")
    first = _read_run(DEFAULT_CSV_PATH_A if csv_path_a is None else csv_path_a, include_lm_head)
    second = _read_run(DEFAULT_CSV_PATH_B if csv_path_b is None else csv_path_b, include_lm_head)
    common_lengths = _common_context_lengths(first, second)
    destination = Path(output_path) if output_path is not None else Path(DEFAULT_OUTPUT_DIR) / (
        f"long_context_prefill_comparison-{_run_directory_hash(first.source)}-{_run_directory_hash(second.source)}.png"
    )
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    labels = []
    for custom_label, run in ((label_a, first), (label_b, second)):
        recorded_name = str(run.metadata.get("device_label") or run.source.stem)
        recorded_name = re.sub(r"^long_context_prefill_latency-(.*)-[0-9]+layers$", r"\1", recorded_name)
        recorded_name = " ".join(recorded_name.replace("_", " ").split())
        device_name = re.sub(r"^NVIDIA\s+", "", recorded_name, flags=re.IGNORECASE)
        model_name = run.metadata.get("jetson_model")
        if not isinstance(model_name, str):
            model_name = recorded_name
        orin_model = re.search(r"\borin\s+(nano|nx)\b", model_name, re.IGNORECASE)
        if orin_model:
            model_family = "Nano" if orin_model.group(1).lower() == "nano" else "NX"
            device_name = f"Jetson Orin {model_family}"
        label = custom_label or device_name
        if orin_model:
            valid_variants = ("4G", "8G") if model_family == "Nano" else ("8G", "16G")
            variant = run.metadata.get("jetson_memory_variant")
            capacity_pattern = r"\b(4|8|16)\s*G(?:i?B)?\b"
            if variant not in valid_variants:
                capacity_match = (re.search(capacity_pattern, custom_label or "", re.IGNORECASE)
                                  or re.search(capacity_pattern, recorded_name, re.IGNORECASE))
                variant = f"{capacity_match.group(1)}G" if capacity_match else None
            if variant in valid_variants:
                if re.search(capacity_pattern, label, re.IGNORECASE):
                    label = re.sub(capacity_pattern, variant, label, flags=re.IGNORECASE)
                else:
                    label = f"{label} {variant}"
        labels.append(label)
    if labels[0] == labels[1]:
        labels = [f"{label} (CSV {suffix})" for label, suffix in zip(labels, ("A", "B"))]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    try:
        for run, label, color in zip((first, second), labels, ("tab:blue", "tab:orange")):
            samples = [run.points[length] for length in common_lengths]
            tokens = [point.input_token_count for point in samples]
            axis.plot(tokens, [statistics.median(point.latencies) for point in samples],
                      marker="o", color=color, label=textwrap.fill(label, width=55))
            axis.fill_between(tokens, [min(point.latencies) for point in samples],
                              [max(point.latencies) for point in samples], color=color, alpha=0.15)
        unit = first.settings["context_length_unit"]
        layer_count = first.settings["loaded_layer_count"]
        endpoint_label = "input to first token (LM head)" if include_lm_head else "input to hidden state"
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Actual model input tokens (chat template included)")
        axis.set_ylabel("Input-to-output latency / loaded layers (ms)")
        axis.set_title(f"Long-context prefill comparison: {layer_count} layers, {endpoint_label}\n"
                       f"Shared context: {common_lengths[0]}–{common_lengths[-1]} {unit} | median and min/max")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)
    print(f"[COMPARE] {len(common_lengths)} shared points: {common_lengths[0]}-{common_lengths[-1]} {unit}; "
          f"{layer_count} layers; include_lm_head={include_lm_head}")
    for run, suffix in ((first, "A"), (second, "B")):
        print(f"[COMPARE] CSV {suffix}: excluded {len(run.points) - len(common_lengths)} unshared successful points; "
              f"repeats per plotted point: {min(len(run.points[n].latencies) for n in common_lengths)}-"
              f"{max(len(run.points[n].latencies) for n in common_lengths)}")
    return destination


def main(argv: list[str] | None = None) -> None:
    """Parse optional CSV overrides and comparison settings, then print the saved figure path.

    Args:
        argv: Explicit argument list or None to read the process command line.
            显式命令行参数列表；None 表示读取当前进程的参数。
    """
    parser = argparse.ArgumentParser(description="Compare two devices' long-context prefill latency on shared inputs.")
    parser.add_argument(
        "csv_path_a", nargs="?", default=DEFAULT_CSV_PATH_A,
        help="First CSV; defaults to DEFAULT_CSV_PATH_A. Relative paths use the repository root.",
    )
    parser.add_argument(
        "csv_path_b", nargs="?", default=DEFAULT_CSV_PATH_B,
        help="Second CSV; defaults to DEFAULT_CSV_PATH_B. Relative paths use the repository root.",
    )
    parser.add_argument(
        "-o", "--output",
        help=f"Output image; defaults to {DEFAULT_OUTPUT_DIR}/long_context_prefill_comparison-<hash A>-<hash B>.png.",
    )
    parser.add_argument("--include-lm-head", action="store_true", help="Compare first-token trials; default: hidden states.")
    parser.add_argument("--label-a", help="First device nickname; default: short device name with recorded Orin capacity.")
    parser.add_argument("--label-b", help="Second device nickname; default: short device name with recorded Orin capacity.")
    arguments = parser.parse_args(argv)
    try:
        output = compare_long_context_prefill(
            arguments.csv_path_a, arguments.csv_path_b, output_path=arguments.output,
            include_lm_head=arguments.include_lm_head, label_a=arguments.label_a, label_b=arguments.label_b,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
