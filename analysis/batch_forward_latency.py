"""Measure Llama prefill forward latency over a B/L grid and export balance rules.

测量 Llama 的请求数 B、严格 token 长度 L 对 prefill 前向时延的影响。
运行脚本才会加载模型、测量和绘图；导入模块不会启动实验。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.batch_latency_balance import find_equivalent_batches, fit_latency_rule


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "batch_forward_latency"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "LongBench-Pro" / "longbench_pro.json"
DEFAULT_SHARDS = PROJECT_ROOT / "shards" / "Llama-3___2-3B-Instruct_float16"
TRIAL_FIELDS = (
    "batch_size", "prompt_tokens", "phase", "repeat_index", "status", "latency_ms",
    "peak_allocated_bytes", "peak_reserved_bytes", "error",
)
SUMMARY_FIELDS = (
    "batch_size", "prompt_tokens", "status", "repeat_count", "median_ms", "p10_ms",
    "p90_ms", "min_ms", "max_ms", "cv", "peak_allocated_bytes", "failure_status",
)
PAIR_FIELDS = (
    "batch_a_size", "prompt_a_tokens", "latency_a_ms", "cv_a", "batch_b_size",
    "prompt_b_tokens", "latency_b_ms", "cv_b", "relative_gap", "tolerance",
)


@dataclass
class StageComponents:
    """Resident input modules and an exact prefix of Transformer layers.

    常驻输入模块与精确的 Transformer 前缀：config 保存模型结构，tokenizer 用于
    准备 token，embedding/rotary 仅用于计时外输入准备，layers 为实际计时的层，
    device/dtype/backend 指定执行条件，environment 保存可追溯的环境信息。
    """

    config: Any
    tokenizer: Any
    embedding: Any
    rotary: Any
    layers: list[Any]
    device: Any
    dtype: Any
    backend: str
    environment: dict[str, Any]


def resolve_path(path: str | Path) -> Path:
    """Resolve a supplied path relative to the repository root.

    Args:
        path: Input or output path; relative paths are interpreted under PROJECT_ROOT.
            输入或输出路径；相对路径统一基于项目根目录解析。
    """
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Save JSON by replacing a closed temporary file in the same directory.

    Args:
        path: Destination metadata file whose contents should be replaced atomically.
            待原子替换内容的元数据目标文件。
        payload: JSON-serializable run settings, progress or analysis results to persist.
            要保存的可 JSON 序列化运行设置、进度或分析结果。
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write a rectangular result table with a fixed column order.

    Args:
        path: CSV destination to create or replace with this derived result table.
            要创建或覆盖的派生结果 CSV 路径。
        fields: Ordered column names defining the exported schema.
            有序列名，用于定义导出表结构。
        rows: Result dictionaries supplying each row's values.
            为各行提供字段值的结果字典。
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def iter_dataset_records(dataset_path: Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON-array record at a time without loading the whole dataset.

    Args:
        dataset_path: Local UTF-8 LongBench-Pro JSON array used as the source of contexts.
            本地 UTF-8 LongBench-Pro JSON 数组文件，用来提供长文本。
    """
    decoder = json.JSONDecoder()
    with dataset_path.open("r", encoding="utf-8-sig") as stream:
        buffer = stream.read(1024 * 1024).lstrip()
        if not buffer.startswith("["):
            raise ValueError("The dataset must be a top-level JSON array")
        buffer, first_record = buffer[1:], True
        while True:
            buffer = buffer.lstrip()
            while not buffer:
                buffer = stream.read(1024 * 1024).lstrip()
                if not buffer:
                    raise ValueError("The dataset array ended unexpectedly")
            if buffer.startswith("]"):
                return
            if not first_record:
                if not buffer.startswith(","):
                    raise ValueError("Missing comma between dataset records")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        raise
                    buffer += chunk
            if not isinstance(record, dict):
                raise ValueError("Every dataset record must be a JSON object")
            yield record
            buffer, first_record = buffer[end:], False


def read_token_bank(
    dataset_path: Path, tokenizer: Any, sample_start: int, request_count: int,
    longest_prompt: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Select distinct records with sufficiently long, exactly truncated token prefixes.

    Args:
        dataset_path: Local dataset to stream for source contexts.
            提供源 context 的本地数据集路径。
        tokenizer: Loaded model tokenizer used to measure lengths in actual model tokens.
            已加载的模型分词器，用来按该模型的实际 token 计数。
        sample_start: Zero-based first record eligible for selection.
            允许选择的首条记录的零基索引。
        request_count: Number of distinct sufficiently long records needed for max B.
            最大 B 所需的不同长文本记录数量。
        longest_prompt: Required number of tokens retained from every selected context.
            从每条选中 context 中保留的 token 数，也是最长待测 L。
    """
    bank: list[list[int]] = []
    manifest: list[dict[str, Any]] = []
    for record_index, record in enumerate(iter_dataset_records(dataset_path)):
        if record_index < sample_start:
            continue
        context = record.get("context")
        if not isinstance(context, str) or not context:
            continue
        token_ids = tokenizer.encode(
            context, add_special_tokens=False, truncation=True, max_length=longest_prompt,
        )
        if len(token_ids) < longest_prompt:
            continue
        bank.append(token_ids)
        manifest.append({
            "record_index": record_index, "sample_id": str(record.get("id", record_index)),
            "retained_tokens": len(token_ids),
            "token_sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
        })
        if len(bank) == request_count:
            return bank, manifest
    raise ValueError(f"Dataset has fewer than {request_count} eligible contexts with {longest_prompt} model tokens")


def load_stage(arguments: argparse.Namespace) -> StageComponents:
    """Load a local Llama layer prefix from project shards or a full model directory.

    Args:
        arguments: Validated CLI settings selecting model source, layer count and hardware.
            已验证的命令行设置，用来选择模型来源、层数和硬件条件。
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    device = torch.device(arguments.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This profiler supports CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu explicitly for a CPU experiment")
    if arguments.cpu_threads is not None:
        torch.set_num_threads(arguments.cpu_threads)
    dtype_name = arguments.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device.type == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = arguments.allow_tf32
        torch.backends.cudnn.allow_tf32 = arguments.allow_tf32

    source = resolve_path(arguments.model_path or arguments.shards_path or DEFAULT_SHARDS)
    config = AutoConfig.from_pretrained(str(source), local_files_only=True)
    if config.model_type != "llama":
        raise ValueError("This stage profiler currently supports Llama-family model configurations")
    if not 1 <= arguments.layer_count <= config.num_hidden_layers:
        raise ValueError(f"--layer-count must be between 1 and {config.num_hidden_layers}")
    config._attn_implementation = arguments.attention_backend
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    if arguments.model_path:
        base_model = AutoModel.from_pretrained(
            str(source), config=config, local_files_only=True,
            torch_dtype=dtype, attn_implementation=arguments.attention_backend,
        )
        embedding = base_model.get_input_embeddings()
        layers = list(base_model.layers[:arguments.layer_count])
        rotary = base_model.rotary_emb
        del base_model
    else:
        embedding_state = torch.load(source / "embedding.pth", map_location="cpu", weights_only=True)
        embedding = torch.nn.Embedding.from_pretrained(embedding_state["weight"], freeze=True)
        del embedding_state
        layers = []
        for layer_index in range(arguments.layer_count):
            # Construct on meta, then assign saved tensors to avoid a second initialized copy.
            # 在 meta 上构造后直接装入已保存张量，避免额外分配一套随机初始化权重。
            with torch.device("meta"):
                layer = LlamaDecoderLayer(config, layer_idx=layer_index)
            layer.load_state_dict(torch.load(
                source / f"block_{layer_index}.pth", map_location="cpu", weights_only=True,
            ), assign=True)
            layers.append(layer)
        rotary = LlamaRotaryEmbedding(config=config, device=device)
    embedding = embedding.to(device=device, dtype=dtype).eval()
    layers = [layer.to(device=device, dtype=dtype).eval() for layer in layers]
    rotary = rotary.to(device=device).eval()
    for module in [embedding, rotary, *layers]:
        module.requires_grad_(False)
    environment = {
        "model_source": str(source), "source_type": "full_model" if arguments.model_path else "project_shards",
        "model_type": config.model_type, "hidden_size": config.hidden_size,
        "layer_count": len(layers), "layer_range": [0, len(layers)],
        "context_limit": config.max_position_embeddings, "dtype": dtype_name,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "total_device_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "sdpa_kernels_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(), "attention_backend": arguments.attention_backend,
        "allow_tf32": arguments.allow_tf32,
        "timing_scope": "Transformer layers [0,N), synchronized wall time, fresh KV per forward",
        "input_policy": "equal-length raw context prefixes; no padding; first B distinct records",
        "excluded_from_timing": ["tokenization", "input_copy", "embedding", "RoPE", "mask_preparation", "final_norm", "lm_head", "communication", "file_IO"],
        "allocator_policy": "retain free CUDA allocator blocks between trials; clear after OOM",
    }
    return StageComponents(config, tokenizer, embedding, rotary, layers, device, dtype,
                           arguments.attention_backend, environment)


def prepare_batch(stage: StageComponents, bank: list[list[int]], batch_size: int, prompt_tokens: int) -> dict[str, Any]:
    """Prepare contiguous equal-length GPU/CPU inputs outside the timed region.

    Args:
        stage: Resident model modules and execution settings used for input preparation.
            常驻模型模块和执行设置，用于准备输入。
        bank: Distinct model-token prefixes supplying reproducible request contents.
            不同文本的 token 前缀，用于提供可复现的请求内容。
        batch_size: Exact number of independent requests to place in this batch.
            当前 batch 中独立请求的严格数量。
        prompt_tokens: Exact L retained for every request, eliminating padding differences.
            每个请求保留的严格 L，用来排除补齐量差异。
    """
    import torch

    with torch.inference_mode():
        input_ids = torch.tensor([row[:prompt_tokens] for row in bank[:batch_size]],
                                 dtype=torch.long, device=stage.device)
        hidden_states = stage.embedding(input_ids).contiguous()
        cache_position = torch.arange(prompt_tokens, device=stage.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        rotary_embeddings = stage.rotary(hidden_states, position_ids)
        attention_mask = None
        if stage.backend == "eager":
            # Eager attention needs an explicit causal mask; SDPA uses is_causal.
            # eager 需要显式因果掩码；SDPA 在等长、空 KV 的 prefill 中使用 is_causal。
            attention_mask = torch.full((prompt_tokens, prompt_tokens), torch.finfo(stage.dtype).min,
                                        device=stage.device, dtype=stage.dtype).triu(diagonal=1)[None, None]
    return {"hidden_states": hidden_states, "position_ids": position_ids,
            "cache_position": cache_position, "position_embeddings": rotary_embeddings,
            "attention_mask": attention_mask}


def measure_forward(stage: StageComponents, prepared: dict[str, Any]) -> dict[str, Any]:
    """Time one layer-prefix traversal with a fresh KV cache and completed GPU work.

    Args:
        stage: Resident Transformer layers and execution device used for this traversal.
            当前遍历使用的常驻 Transformer 层和执行设备。
        prepared: Ready tensors from prepare_batch; reused without changing their contents.
            prepare_batch 生成的已就绪张量；复用但不修改其内容。
    """
    import torch
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    hidden_states = None
    try:
        with torch.inference_mode():
            if stage.device.type == "cuda":
                torch.cuda.synchronize(stage.device)
                torch.cuda.reset_peak_memory_stats(stage.device)
            started = time.perf_counter()
            hidden_states = prepared["hidden_states"]
            for layer in stage.layers:
                hidden_states = layer(
                    hidden_states=hidden_states, attention_mask=prepared["attention_mask"],
                    position_ids=prepared["position_ids"], past_key_value=cache,
                    use_cache=True, cache_position=prepared["cache_position"],
                    position_embeddings=prepared["position_embeddings"], output_attentions=False,
                )[0]
            if stage.device.type == "cuda":
                torch.cuda.synchronize(stage.device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if cache.get_seq_length() != prepared["hidden_states"].shape[1]:
                raise RuntimeError("The forward did not create a fresh L-token KV cache")
            return {
                "latency_ms": elapsed_ms,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(stage.device) if stage.device.type == "cuda" else "",
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(stage.device) if stage.device.type == "cuda" else "",
            }
    finally:
        hidden_states = cache = None


def run_measurements(
    stage: StageComponents, bank: list[list[int]], arguments: argparse.Namespace,
    run_directory: Path, metadata: dict[str, Any],
) -> None:
    """Persist warmups, repeats and failures for every shuffled B/L combination.

    Args:
        stage: Loaded modules held constant across all measured shapes.
            在所有形状测量中保持不变的已加载模块。
        bank: CPU token prefixes reused to make B/L input construction reproducible.
            复用的 CPU token 前缀，用来保证 B/L 输入构造可复现。
        arguments: Sweep sizes, warmup/repeat counts and deterministic shuffle seed.
            扫描范围、预热/重复次数与确定性的打乱种子。
        run_directory: Dedicated output directory receiving durable raw measurement rows.
            用来持续保存原始测量记录的独立运行目录。
        metadata: Mutable progress object saved before each shape and after completion.
            可变进度对象，在每种形状开始前与结束后保存。
    """
    import torch

    shapes = list(itertools.product(arguments.batch_sizes, arguments.prompt_lengths))
    random.Random(arguments.seed).shuffle(shapes)
    metadata["shape_order"] = shapes
    with (run_directory / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        stream.flush()
        for batch_size, prompt_tokens in shapes:
            metadata["pending_shape"] = {"batch_size": batch_size, "prompt_tokens": prompt_tokens}
            write_json(run_directory / "run.json", metadata)
            base_row = {"batch_size": batch_size, "prompt_tokens": prompt_tokens}
            if prompt_tokens > stage.config.max_position_embeddings:
                writer.writerow(dict(base_row, phase="measure", repeat_index=0, status="context_limit"))
                stream.flush()
                continue
            prepared = None
            phase, repeat_index = "prepare", 0
            try:
                prepared = prepare_batch(stage, bank, batch_size, prompt_tokens)
                for phase, count in (("warmup", arguments.warmup), ("measure", arguments.repeats)):
                    for repeat_index in range(count):
                        observation = measure_forward(stage, prepared)
                        writer.writerow(dict(base_row, phase=phase, repeat_index=repeat_index,
                                             status="success", **observation))
                        stream.flush()
                print(f"[MEASURED] B={batch_size}, L={prompt_tokens}", flush=True)
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                writer.writerow(dict(base_row, phase=phase, repeat_index=repeat_index,
                                     status="oom", error=str(error)[:1000]))
                stream.flush()
                prepared = None
                error.__traceback__ = None
                gc.collect()
                if stage.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[OOM] B={batch_size}, L={prompt_tokens}; continuing with other shapes", flush=True)
            finally:
                prepared = None
        metadata["pending_shape"] = None


def percentile(values: Sequence[float], fraction: float) -> float:
    """Interpolate one descriptive percentile of a non-empty sample.

    Args:
        values: Repeated latency observations used to summarize spread.
            用来描述离散程度的重复时延观测。
        fraction: Quantile position between zero and one, such as 0.1 or 0.9.
            零到一之间的分位位置，例如 0.1 或 0.9。
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_trials(run_directory: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate completed measurement repeats, keeping missing/OOM shapes explicit.

    Args:
        run_directory: Experiment directory containing trials.csv.
            包含 trials.csv 的实验目录。
        metadata: Run settings defining planned shapes and the required repeat count.
            定义计划形状与所需重复次数的运行设置。
    """
    settings = metadata["settings"]
    grouped: dict[tuple[int, int], dict[int, float]] = {}
    failures: dict[tuple[int, int], str] = {}
    peaks: dict[tuple[int, int], int] = {}
    with (run_directory / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                shape = (int(row["batch_size"]), int(row["prompt_tokens"]))
                if row["status"] != "success":
                    failures[shape] = row["status"]
                if row["phase"] != "measure" or row["status"] != "success":
                    continue
                repeat_index, latency = int(row["repeat_index"]), float(row["latency_ms"])
                if not 0 <= repeat_index < settings["repeats"] or not math.isfinite(latency) or latency <= 0:
                    continue
                grouped.setdefault(shape, {})[repeat_index] = latency
                if row.get("peak_allocated_bytes"):
                    peaks[shape] = max(peaks.get(shape, 0), int(row["peak_allocated_bytes"]))
            except (KeyError, TypeError, ValueError):
                continue
    summaries = []
    for shape in itertools.product(settings["batch_sizes"], settings["prompt_lengths"]):
        values = list(grouped.get(shape, {}).values())
        complete = len(values) == settings["repeats"] and shape not in failures
        summary: dict[str, Any] = {
            "batch_size": shape[0], "prompt_tokens": shape[1],
            "status": "success" if complete else failures.get(shape, "partial" if values else "unmeasured"),
            "repeat_count": len(values), "failure_status": failures.get(shape, ""),
        }
        if values:
            summary.update(median_ms=statistics.median(values), p10_ms=percentile(values, 0.1),
                           p90_ms=percentile(values, 0.9), min_ms=min(values), max_ms=max(values),
                           cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0,
                           peak_allocated_bytes=peaks.get(shape, ""))
        summaries.append(summary)
    return summaries


def plot_latency_curves(
    summaries: Sequence[dict[str, Any]], run_directory: Path, metadata: dict[str, Any],
) -> list[str]:
    """Plot latency versus B, versus L and the measured B/L latency map.

    Args:
        summaries: Aggregated measurements; only complete successful shapes are drawn.
            聚合测量点；只绘制重复次数完整的成功形状。
        run_directory: Directory receiving the three PNG charts.
            用于保存三张 PNG 曲线/热力图的目录。
        metadata: Hardware and experiment settings used for titles and planned grid axes.
            硬件与实验设置，用来生成标题和计划网格坐标。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    points = {(point["batch_size"], point["prompt_tokens"]): point
              for point in summaries if point["status"] == "success"}
    if not points:
        return []
    batches = metadata["settings"]["batch_sizes"]
    lengths = metadata["settings"]["prompt_lengths"]
    environment = metadata.get("environment", {})
    title = f"{environment.get('device_name', 'device')} | {environment.get('layer_count', '?')} layers | {environment.get('attention_backend', '?')}"
    output_names = []
    for varying_batch in (True, False):
        figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        try:
            coordinates, series = (batches, lengths) if varying_batch else (lengths, batches)
            for fixed in series:
                selected = [points.get((value, fixed) if varying_batch else (fixed, value))
                            for value in coordinates]
                medians = [point["median_ms"] if point else math.nan for point in selected]
                lower = [point["p10_ms"] if point else math.nan for point in selected]
                upper = [point["p90_ms"] if point else math.nan for point in selected]
                label = f"L={fixed} tokens" if varying_batch else f"B={fixed}"
                line, = axis.plot(coordinates, medians, marker="o", label=label)
                axis.fill_between(coordinates, lower, upper, color=line.get_color(), alpha=0.12)
            axis.set_xlabel("Batch size B (requests)" if varying_batch else "Prompt length L (tokens/request)")
            axis.set_ylabel("Transformer prefill forward latency (ms)")
            axis.set_title(title + "\nMedian; shading: 10th-90th percentiles")
            axis.set_ylim(bottom=0)
            axis.set_xticks(coordinates)
            axis.grid(alpha=0.25)
            axis.legend(ncol=2, fontsize=9)
            filename = "latency_vs_batch_size.png" if varying_batch else "latency_vs_prompt_length.png"
            figure.savefig(run_directory / filename, dpi=300)
            output_names.append(filename)
        finally:
            plt.close(figure)
    matrix = np.asarray([[points[(batch, length)]["median_ms"] if (batch, length) in points else np.nan
                          for length in lengths] for batch in batches])
    figure, axis = plt.subplots(figsize=(max(7, len(lengths) * 1.05), max(4, len(batches) * 0.65)), constrained_layout=True)
    try:
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(lengths)), labels=lengths)
        axis.set_yticks(range(len(batches)), labels=batches)
        axis.set_xlabel("Prompt length L (tokens/request)")
        axis.set_ylabel("Batch size B (requests)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Median forward latency (ms)")
        for row_index, batch in enumerate(batches):
            for column_index, length in enumerate(lengths):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    color = "black" if image.norm(value) > 0.5 else "white"
                    text = f"{value:.2f}"
                else:
                    color = "black"
                    text = next(point["status"] for point in summaries
                                if point["batch_size"] == batch and point["prompt_tokens"] == length)
                axis.text(column_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
        figure.savefig(run_directory / "latency_map.png", dpi=300)
        output_names.append("latency_map.png")
    finally:
        plt.close(figure)
    return output_names


def analyze_results(run_directory: Path, tolerance: float | None = None) -> dict[str, Any]:
    """Generate charts, a validated function and measured equal-latency lookup tables.

    Args:
        run_directory: Existing measurement directory containing run.json and trials.csv.
            包含 run.json 与 trials.csv 的已有测量目录。
        tolerance: Optional matching tolerance overriding the saved experiment setting.
            可选近似相等容差；指定时覆盖保存的实验设置。
    """
    metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    relative_tolerance = metadata["settings"]["relative_tolerance"] if tolerance is None else tolerance
    summaries = summarize_trials(run_directory, metadata)
    write_csv(run_directory / "summary.csv", SUMMARY_FIELDS, summaries)
    successful = [point for point in summaries if point["status"] == "success"]
    rule = fit_latency_rule(successful, relative_tolerance)
    rule["environment"] = metadata.get("environment", {})
    rule["measurement_scope"] = "prefill Transformer layer prefix; no decode or communication"
    write_json(run_directory / "balance_rule.json", rule)
    pairs = find_equivalent_batches(successful, relative_tolerance)
    write_csv(run_directory / "equivalent_batches.csv", PAIR_FIELDS, pairs)
    # Save tables and calibration before plotting, so analysis survives backend issues.
    # 先保存表格与标定规则，再绘图，使绘图库故障不会丢失分析结果。
    charts = plot_latency_curves(summaries, run_directory, metadata)
    report = {"successful_shapes": len(successful), "measured_equivalent_pairs": len(pairs),
              "function_accepted": rule["accepted"], "reason": rule["reason"],
              "relative_tolerance": relative_tolerance, "charts": charts}
    write_json(run_directory / "analysis_summary.json", report)
    print("[BALANCE]", rule["reason"])
    if rule["accepted"]:
        coefficient = rule["coefficients"]
        print("[FUNCTION] t_ms(B,L) = "
              f"{coefficient['c0']:.12g} + B * ({coefficient['cB']:.12g} + "
              f"{coefficient['cBL']:.12g}*L + {coefficient['cBL2']:.12g}*L**2)")
        print("[INVERSE] B(T,L) = (T-c0)/(cB+cBL*L+cBL2*L**2); integer suggestions require confirmation")
    for pair in pairs[:10]:
        print(f"[MEASURED MATCH] B={pair['batch_a_size']}, L={pair['prompt_a_tokens']}: "
              f"{pair['latency_a_ms']:.3f} ms; B={pair['batch_b_size']}, L={pair['prompt_b_tokens']}: "
              f"{pair['latency_b_ms']:.3f} ms; gap={pair['relative_gap']:.2%}")
    if not pairs:
        print("[BALANCE] No stable measured pair meets the tolerance on this grid; see summary.csv")
    return report


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Create one run, collect measurements and analyze completed results.

    Args:
        arguments: Validated experiment settings used for loading, measurement and output.
            已验证的实验设置，用来控制加载、测量和结果输出。
    """
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_directory = resolve_path(arguments.output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    settings = vars(arguments).copy()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    metadata: dict[str, Any] = {"schema_version": 1, "run_id": run_id, "state": "loading",
                                "settings": settings, "pending_shape": None}
    write_json(run_directory / "run.json", metadata)
    stage, bank = None, None
    measurement_error: BaseException | None = None
    try:
        stage = load_stage(arguments)
        metadata["environment"] = stage.environment
        metadata["state"] = "preparing_inputs"
        write_json(run_directory / "run.json", metadata)
        valid_lengths = [length for length in arguments.prompt_lengths
                         if length <= stage.config.max_position_embeddings]
        if not valid_lengths:
            raise ValueError("All requested L values exceed the model context limit")
        bank, manifest = read_token_bank(resolve_path(arguments.dataset), stage.tokenizer,
                                        arguments.sample_start, max(arguments.batch_sizes), max(valid_lengths))
        write_json(run_directory / "input_manifest.json", {"dataset": str(resolve_path(arguments.dataset)), "samples": manifest})
        metadata["state"] = "measuring"
        run_measurements(stage, bank, arguments, run_directory, metadata)
        metadata["state"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        measurement_error = error
    finally:
        stage = bank = None
        gc.collect()
        write_json(run_directory / "run.json", metadata)
        print("[RESULTS]", run_directory, flush=True)
    if (run_directory / "trials.csv").exists():
        try:
            analyze_results(run_directory)
        except Exception as analysis_error:
            if measurement_error is None:
                raise
            print(f"[ANALYSIS] {analysis_error}; use --analyze-only to process completed rows", file=sys.stderr)
    if measurement_error is not None:
        raise measurement_error
    return run_directory


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate measurement or analysis-only settings.

    Args:
        arguments: Optional argument sequence for parsing; None reads the command line.
            可选待解析参数序列；None 表示读取实际命令行。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--shards-path", type=Path, help="Local project Llama shard directory")
    sources.add_argument("--model-path", type=Path, help="Local Hugging Face Llama model directory")
    parser.add_argument("--layer-count", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--relative-tolerance", type=float, default=None, help="Default 0.05; analysis-only defaults to the saved tolerance")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--analyze-only", type=Path, help="Process an existing run without importing torch or loading a model")
    parsed = parser.parse_args(arguments)
    if parsed.relative_tolerance is not None and not 0 < parsed.relative_tolerance < 1:
        parser.error("--relative-tolerance must be between 0 and 1")
    if parsed.analyze_only:
        return parsed
    parsed.relative_tolerance = 0.05 if parsed.relative_tolerance is None else parsed.relative_tolerance
    if (parsed.layer_count < 1 or parsed.warmup < 1 or parsed.repeats < 3
            or parsed.sample_start < 0 or (parsed.cpu_threads is not None and parsed.cpu_threads < 1)
            or any(value < 1 for value in parsed.batch_sizes + parsed.prompt_lengths)):
        parser.error("Use positive B/L/layers/threads, warmup >= 1, repeats >= 3 and sample-start >= 0")
    parsed.batch_sizes = sorted(set(parsed.batch_sizes))
    parsed.prompt_lengths = sorted(set(parsed.prompt_lengths))
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the requested measurement or reanalyze saved CSV data.

    Args:
        arguments: Optional CLI argument sequence; None uses this process's arguments.
            可选命令行参数序列；None 使用当前进程参数。
    """
    parsed = parse_arguments(arguments)
    if parsed.analyze_only:
        analyze_results(resolve_path(parsed.analyze_only), parsed.relative_tolerance)
    else:
        run_experiment(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
