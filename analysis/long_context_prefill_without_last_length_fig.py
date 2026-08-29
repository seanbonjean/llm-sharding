"""Plot a long-context prefill run without its last successful experiment length.

Run with ``python analysis/long_context_prefill_without_last_length_fig.py <results.csv>``.
The shared single-device renderer keeps statistics and styling aligned with
``long_context_prefill_result_fig.py``.
绘制长上下文 prefill 结果并排除最后一个成功实验长度；绘图口径与单设备通用脚本一致。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ANALYSIS_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIRECTORY.parent
if str(ANALYSIS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIRECTORY))

from long_context_prefill_result_fig import plot_long_context_prefill


def _last_successful_measurement_token_counts(csv_path: str | Path) -> tuple[int, frozenset[int]]:
    """Find input token counts belonging to the largest successful measured context length.

    Args:
        csv_path: Long-context profiling CSV; a relative path is based on PROJECT_ROOT.
            长上下文实验 CSV；相对路径以项目根目录为基准。

    Returns:
        The last successful context length and every actual token count recorded for it.
        最大成功正式 context_length，以及该长度对应的全部实际输入 token 数。
    """
    source = Path(csv_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    token_counts_by_context_length: dict[int, set[int]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"phase", "status", "context_length", "input_token_count"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing columns required to identify the last experiment length")
        for row in reader:
            if row["phase"] != "measure" or row["status"] != "success":
                continue
            try:
                context_length = int(row["context_length"])
                input_token_count = int(row["input_token_count"])
            except (TypeError, ValueError):
                continue
            if context_length < 1 or input_token_count < 1:
                continue
            token_counts_by_context_length.setdefault(context_length, set()).add(input_token_count)
    if len(token_counts_by_context_length) < 2:
        raise ValueError("CSV needs at least two successful measured context lengths")
    last_context_length = max(token_counts_by_context_length)
    return last_context_length, frozenset(token_counts_by_context_length[last_context_length])


def plot_long_context_prefill_without_last_length(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render a profiling run after omitting its largest successful measured context length.

    Args:
        csv_path: Long-context profiling CSV; a relative path is based on PROJECT_ROOT.
            长上下文实验 CSV；相对路径以项目根目录为基准。
        output_path: Optional image path relative to PROJECT_ROOT when not absolute.
            By default, a suffixed image is written beside the input CSV.
            可选输出图片路径；相对路径以项目根目录为基准，默认以带后缀的文件名保存在输入 CSV 同目录。
    """
    source = Path(csv_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    last_context_length, excluded_token_counts = _last_successful_measurement_token_counts(source)
    destination = output_path
    if destination is None:
        destination = source.with_name(f"{source.stem}-without-last-length.png")
    print(
        f"[PLOT] excluded last successful context length {last_context_length}; "
        f"actual input tokens: {', '.join(str(value) for value in sorted(excluded_token_counts))}",
    )
    return plot_long_context_prefill(
        source,
        output_path=destination,
        excluded_input_token_counts=excluded_token_counts,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse input/output paths and print the saved figure location.

    Args:
        argv: Optional command-line arguments; None reads the current process arguments.
            可选命令行参数；None 表示读取当前进程参数。
    """
    parser = argparse.ArgumentParser(
        description="Plot long-context prefill latency without the last successful experiment length.",
    )
    parser.add_argument(
        "csv_path",
        help="Input CSV; relative paths use the repository root.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image; defaults to a suffixed filename beside the input CSV.",
    )
    arguments = parser.parse_args(argv)
    print(plot_long_context_prefill_without_last_length(arguments.csv_path, output_path=arguments.output))


if __name__ == "__main__":
    main()
