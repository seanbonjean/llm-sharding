"""Standard-library regression tests for long-context profiling, with mocked ML APIs.

Run with ``python -B unittest.py``; model weights, torch and CUDA are not required.
使用 ``python -B unittest.py`` 运行；使用模拟 ML 接口，不需要权重、torch 或 CUDA。
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any

# Resolve the standard-library package rather than this file's identical name.
# 加载标准库 unittest 包，避免当前同名脚本遮蔽它。
PROJECT_ROOT = Path(__file__).resolve().parent
_original_import_paths = sys.path[:]
sys.path = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != PROJECT_ROOT]
import unittest
from unittest import mock
sys.path[:] = _original_import_paths


class FakeOutOfMemoryError(RuntimeError):
    """Represent a CUDA allocation failure in dependency-free tests. 模拟 CUDA OOM。"""


class FakeInferenceMode(contextlib.ContextDecorator):
    """Stand in for torch.inference_mode as both context manager and decorator. 模拟推理上下文。"""

    def __enter__(self) -> FakeInferenceMode:
        """Enter this test context. 进入测试上下文。"""
        return self

    def __exit__(self, *exception_info: Any) -> None:
        """Leave the context; exception_info contains the exception triple. 退出并保留异常传播。"""


def load_profile_module() -> types.ModuleType:
    """Import the real source with ML/communication dependencies replaced. 模拟外部依赖导入真实源码。"""
    dependency_names = (
        "torch", "transformers", "transformers.cache_utils",
        "transformers.models", "transformers.models.llama",
        "transformers.models.llama.modeling_llama", "utils.node_worker",
        "utils.shard_loader", "utils.forwarding_utils",
    )
    dependencies = {name: types.ModuleType(name) for name in dependency_names}
    fake_torch = dependencies["torch"]
    fake_torch.float16 = "float16"
    fake_torch.long = "long"
    fake_torch.__version__ = "test-double"
    fake_torch.inference_mode = FakeInferenceMode
    fake_torch.cuda = types.SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        empty_cache=mock.Mock(), reset_peak_memory_stats=mock.Mock(),
        get_device_name=mock.Mock(return_value="Mock GPU"),
    )
    dependencies["transformers"].LlamaConfig = mock.Mock()
    dependencies["transformers"].AutoTokenizer = mock.Mock()
    dependencies["transformers.cache_utils"].DynamicCache = mock.Mock(side_effect=object)
    dependencies["transformers.models.llama.modeling_llama"].LlamaRotaryEmbedding = mock.Mock()
    dependencies["utils.node_worker"].NodeWorker = mock.Mock()
    dependencies["utils.shard_loader"].LlamaShardPart = mock.Mock()
    dependencies["utils.forwarding_utils"].build_position_ids = mock.Mock()
    spec = importlib.util.spec_from_file_location("long_context_profile_under_test", PROJECT_ROOT / "utils/node_profiler.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate node_profiler.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def make_profiler(module: types.ModuleType, model_layers: int = 4) -> Any:
    """Build a profiler without initializing a real model.

    Args:
        module: Imported production module with mocked dependencies. 模拟依赖后的真实模块。
        model_layers: Total model layers for boundary tests. 测试用模型总层数。
    """
    profiler = module.NodeProfiler.__new__(module.NodeProfiler)
    profiler.config = types.SimpleNamespace(max_position_embeddings=1000, _attn_implementation="eager")
    profiler.layer_num = model_layers
    profiler.device = types.SimpleNamespace(type="cpu")
    profiler.dtype = "float16"
    profiler.shards_path = "test-weights"
    profiler._long_context_load_components = mock.Mock(
        side_effect=lambda layer_num: {"shard": types.SimpleNamespace(config=profiler.config)},
    )
    profiler._long_context_run_trial = mock.Mock(return_value={
        "status": "success", "input_token_count": 12, "prefill_latency_ms": 8.0,
    })
    return profiler


def write_dataset(path: Path, language: str = "English") -> None:
    """Write a tiny bilingual fixture.

    Args:
        path: Temporary JSON destination. 临时 JSON 路径。
        language: Language metadata used to select the truncation unit. 截断单位选择用语言。
    """
    record = {
        "id": "sample-a", "language": language, "context": "one\n two\tthree four five",
        "question_nonthinking": "Answer directly.", "question_thinking": "Think first.",
        "answer": ["not part of the prompt"],
    }
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read trial rows; path identifies the generated CSV. 读取指定 CSV 的测试记录。"""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_with_mocked_backend(module: types.ModuleType, csv_path: Path) -> tuple[Path, Any, Any, Any]:
    """Exercise CSV processing with a recording plotting backend.

    Args:
        module: Production profiler module. 真实 profiler 模块。
        csv_path: Input CSV, optionally accompanied by metadata. 输入 CSV 及可选元数据。
    """
    pyplot = types.ModuleType("matplotlib.pyplot")
    figure, axis = mock.MagicMock(), mock.MagicMock()
    pyplot.subplots = mock.Mock(return_value=(figure, axis))
    pyplot.close = mock.Mock()
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = mock.Mock()
    matplotlib.pyplot = pyplot
    with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
        output = module.NodeProfiler.plot_long_context_prefill(csv_path)
    return output, figure, axis, pyplot


def crash_probe(directory: Path) -> None:
    """Exit during a synthetic trial, bypassing finally to emulate an uncatchable stop.

    Args:
        directory: Parent test's temporary output directory. 父测试的临时结果目录。
    """
    module = load_profile_module()
    profiler = make_profiler(module)
    dataset = directory / "dataset.json"
    write_dataset(dataset)
    calls = 0

    def interrupted_trial(*arguments: Any) -> dict[str, Any]:
        """Return two successes, then exit; arguments are the production trial arguments.
        前两次成功，第三次直接退出；arguments 为实际 trial 形参。
        """
        nonlocal calls
        calls += 1
        if calls == 3:
            os._exit(73)
        return {"status": "success", "input_token_count": 10, "prefill_latency_ms": 4.0}

    profiler._long_context_run_trial = interrupted_trial
    profiler.profile_long_context_prefill(2, dataset_path=dataset, output_dir=directory, initial_context_length=1, repeat_num=1)


class LongContextTests(unittest.TestCase):
    """Exercise production orchestration and timing boundaries. 验证真实调度代码与计时边界。"""

    def setUp(self) -> None:
        """Create an isolated workspace and dependency doubles. 创建隔离工作区和依赖替身。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dataset = self.directory / "dataset.json"
        write_dataset(self.dataset)
        self.module = load_profile_module()
        self.profiler = make_profiler(self.module)

    def run_profile(self, **overrides: Any) -> Path:
        """Run with test defaults; overrides replace individual function arguments.
        使用测试默认参数运行，overrides 覆盖指定参数。
        """
        arguments = dict(layer_num=2, dataset_path=self.dataset, output_dir=self.directory,
                         initial_context_length=2, repeat_num=2)
        arguments.update(overrides)
        return self.profiler.profile_long_context_prefill(**arguments)

    def test_streamed_selection_and_unicode(self) -> None:
        """Select records across tiny UTF-8 chunks without decoding the later record. 验证流式选样本。"""
        self.dataset.write_text('[{"id":"中文😀"}, {"id":"two"}, BROKEN]', encoding="utf-8")
        reader = self.profiler._long_context_read_sample
        self.assertEqual(reader(self.dataset, 0, 1)["id"], "中文😀")
        self.assertEqual(reader(self.dataset, 1, 2)["id"], "two")
        with self.assertRaises(json.JSONDecodeError):
            reader(self.dataset, 2, 3)

    def test_stream_errors(self) -> None:
        """Reject malformed arrays, absent records and non-object samples. 拒绝无效样本结构。"""
        for contents, index, exception in [('[]', 0, IndexError), ('[{}]', 1, IndexError),
                                           ('{}', 0, ValueError), ('[5]', 0, ValueError),
                                           ('[{},]', 1, ValueError), ('[', 0, ValueError)]:
            with self.subTest(contents=contents):
                self.dataset.write_text(contents, encoding="utf-8")
                with self.assertRaises(exception):
                    self.profiler._long_context_read_sample(self.dataset, index, 1)

    def test_prefixes_preserve_whitespace_and_full_tail(self) -> None:
        """Double word counts and preserve the complete final original text. 倍增词数并保留格式。"""
        original = "  alpha\n beta\t gamma  delta epsilon \n"
        prefixes = list(self.profiler._long_context_prefixes(original, 2, "words"))
        self.assertEqual([item[1] for item in prefixes], [2, 4, 5])
        self.assertEqual(prefixes[0][0], "  alpha\n beta")
        self.assertEqual(prefixes[-1], (original, 5, True))
        self.assertEqual(list(self.profiler._long_context_prefixes(original, 999, "words")), [(original, 5, True)])

    def test_character_prefixes(self) -> None:
        """Double character counts, including Unicode characters. 验证中文字符倍增。"""
        self.assertEqual(list(self.profiler._long_context_prefixes("中文😀测试", 2, "characters")),
                         [("中文", 2, False), ("中文😀测", 4, False), ("中文😀测试", 5, True)])

    def test_single_warmup_and_exact_layer_count(self) -> None:
        """Warm once, retain raw repeats, and divide elapsed by the actual N. 验证一次预热与精确层数。"""
        result = self.run_profile()
        rows = read_rows(result)
        self.profiler._long_context_load_components.assert_called_once_with(2)
        self.assertEqual(len([row for row in rows if row["phase"] == "warmup"]), 1)
        measured = [row for row in rows if row["phase"] == "measure"]
        self.assertEqual([int(row["context_length"]) for row in measured], [2, 2, 4, 4, 5, 5])
        self.assertTrue(all(row["include_lm_head"] == "False" for row in measured))
        self.assertTrue(all(float(row["prefill_latency_per_layer_ms"]) == 4 for row in measured))
        metadata = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["stop_reason"], "full_context")
        self.assertIsNone(metadata["pending_attempt"])
        self.assertTrue(result.name.startswith("long_context_prefill_latency-"))
        self.assertTrue(result.name.endswith("-2layers.csv"))
        self.module.NodeWorker.assert_not_called()

    def test_full_model_both_endpoints(self) -> None:
        """Full models warm once with the head and measure both endpoints. 全模型测两种终点。"""
        rows = read_rows(self.run_profile(layer_num=4, repeat_num=1, initial_context_length=10))
        self.assertEqual([(row["phase"], row["include_lm_head"]) for row in rows],
                         [("warmup", "True"), ("measure", "False"), ("measure", "True")])

    def test_component_loading_exact_layers_and_head(self) -> None:
        """Load the exact block prefix and only attach full-model output modules. 验证实际组件装载参数。"""
        self.profiler.config.vocab_size = 128
        self.profiler.config.hidden_size = 8
        self.module.torch.nn = types.SimpleNamespace(Embedding=mock.Mock(), Linear=mock.Mock())
        self.module.torch.load = mock.Mock(return_value={"weight": "fixture"})
        for layer_count in (2, 4):
            with self.subTest(layer_count=layer_count):
                components = self.module.NodeProfiler._long_context_load_components(self.profiler, layer_count)
                arguments = self.module.LlamaShardPart.call_args
                self.assertEqual(arguments.args[1], [f"block_{index}.pth" for index in range(layer_count)])
                self.assertEqual(arguments.args[2:4], (0, layer_count))
                self.assertEqual(arguments.kwargs["add_final_norm"], layer_count == 4)
                self.assertEqual(components["lm_head"] is not None, layer_count == 4)
        self.module.torch.nn.Linear.assert_called_once_with(8, 128, bias=False)
        self.module.NodeWorker.assert_not_called()

    def test_thinking_selection_and_chinese_auto_unit(self) -> None:
        """Choose the thinking field and character units from explicit metadata. 验证问题与语言选项。"""
        write_dataset(self.dataset, "Chinese")
        result = self.run_profile(use_thinking_question=True, initial_context_length=999, repeat_num=1)
        self.assertEqual(read_rows(result)[0]["context_length_unit"], "characters")
        self.assertEqual(self.profiler._long_context_run_trial.call_args.args[2], "Think first.")

    def test_validation_before_loading(self) -> None:
        """Validate model adapter, counts and boolean type before loading. 验证装载前参数检查。"""
        for values, error in [({"model_type": "future-model"}, NotImplementedError),
                              ({"layer_num": 5}, ValueError), ({"layer_num": -1}, ValueError),
                              ({"layer_num": True}, ValueError), ({"repeat_num": 0}, ValueError),
                              ({"sample_index": -1}, ValueError), ({"use_thinking_question": 1}, TypeError),
                              ({"device_label": 123}, TypeError),
                              ({"context_length_unit": "tokens"}, ValueError)]:
            with self.subTest(values=values), self.assertRaises(error):
                self.run_profile(**values)
        self.profiler._long_context_load_components.assert_not_called()

    def test_missing_question_rejected(self) -> None:
        """A missing selected question is a data error. 缺失所选问题时报数据错误。"""
        self.dataset.write_text('[{"context":"content"}]', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_profile()

    def test_oom_retains_prior_results_and_empty_failure_latency(self) -> None:
        """Preserve successful CSV rows and distinguish a confirmed allocation error. 保留 OOM 前结果。"""
        success = {"status": "success", "input_token_count": 10, "prefill_latency_ms": 4.0}
        self.profiler._long_context_run_trial.side_effect = [success, success, FakeOutOfMemoryError("allocation failed")]
        result = self.run_profile(repeat_num=1)
        rows = read_rows(result)
        self.assertEqual([row["status"] for row in rows], ["success", "success", "oom"])
        self.assertEqual(rows[-1]["prefill_latency_ms"], "")
        self.assertEqual(json.loads(result.with_suffix(".json").read_text())["stop_reason"], "oom")

    def test_context_limit_is_distinct(self) -> None:
        """Stop with a separately identified model context limit. 区分模型长度限制。"""
        self.profiler._long_context_run_trial.return_value = {"status": "context_limit", "input_token_count": 1001}
        result = self.run_profile()
        self.assertEqual(len(read_rows(result)), 1)
        self.assertEqual(read_rows(result)[0]["status"], "context_limit")

    def test_warmup_oom_stops_before_measurements(self) -> None:
        """Preserve an initial allocation failure without advancing the sweep. 预热 OOM 后立即终止。"""
        self.profiler._long_context_run_trial.side_effect = MemoryError("host allocation failed")
        result = self.run_profile()
        rows = read_rows(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["phase"], rows[0]["status"]), ("warmup", "oom"))
        self.assertEqual(rows[0]["input_token_count"], "")
        self.assertEqual(rows[0]["prefill_latency_per_layer_ms"], "")

    def test_unexpected_error_is_not_classified_as_oom(self) -> None:
        """Keep diagnostic progress and propagate non-OOM errors. 保留诊断进度并传播其他错误。"""
        self.profiler._long_context_run_trial.side_effect = RuntimeError("invalid tensor")
        with self.assertRaisesRegex(RuntimeError, "invalid tensor"):
            self.run_profile()
        metadata_file = next(self.directory.glob("*/long_context*.json"))
        metadata = json.loads(metadata_file.read_text())
        self.assertEqual(metadata["status"], "error")
        self.assertEqual(metadata["pending_attempt"]["attempt_id"], 1)

    def test_pending_snapshot_precedes_trial(self) -> None:
        """The run snapshot and CSV header exist before risky model computation. 计算前已落盘。"""
        def inspect_pending(*arguments: Any) -> dict[str, Any]:
            """Inspect persisted progress; arguments are the production trial arguments.
            检查持久化进度；arguments 为实际 trial 参数。
            """
            metadata_file = next(self.directory.glob("*/long_context*.json"))
            pending = json.loads(metadata_file.read_text())["pending_attempt"]
            self.assertEqual(pending["status"], "started")
            self.assertTrue(metadata_file.with_suffix(".csv").read_text().startswith("attempt_id,"))
            return {"status": "context_limit", "input_token_count": 2000}
        self.profiler._long_context_run_trial.side_effect = inspect_pending
        self.run_profile()

    def test_process_exit_retains_durable_progress(self) -> None:
        """A child exit bypasses handlers while keeping flushed results and pending metadata.
        子进程直接退出绕过异常处理，已落盘的结果和 pending 进度仍在。
        """
        process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--crash-probe", str(self.directory)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 73, process.stderr)
        result = next(self.directory.glob("*/long_context*.csv"))
        self.assertEqual(len(read_rows(result)), 2)
        metadata = json.loads(result.with_suffix(".json").read_text())
        self.assertEqual(metadata["status"], "running")
        self.assertEqual(metadata["pending_attempt"]["attempt_id"], 3)

    def test_unique_run_directories(self) -> None:
        """Repeated invocations keep independent files. 重复运行保留独立文件。"""
        first = self.run_profile(initial_context_length=999, repeat_num=1)
        second = self.run_profile(initial_context_length=999, repeat_num=1)
        self.assertNotEqual(first.parent, second.parent)
        self.assertTrue(first.is_file())

    def test_device_detection_fallbacks(self) -> None:
        """Handle Jetson, CUDA-query failure, CPU and invalid filename characters. 验证设备识别回退。"""
        self.profiler.device.type = "cuda"
        with mock.patch.object(Path, "read_text", return_value="NVIDIA Jetson Orin Nano\x00"):
            self.assertEqual(self.profiler._long_context_device_label(None), "NVIDIA_Jetson_Orin_Nano")
            self.module.torch.cuda.get_device_name.assert_not_called()
        self.module.torch.cuda.get_device_name.side_effect = RuntimeError("not supported")
        with mock.patch.object(Path, "read_text", side_effect=OSError("absent")), mock.patch.object(self.module.platform, "processor", side_effect=OSError()):
            self.assertEqual(self.profiler._long_context_device_label(None), "cuda")
        self.assertEqual(self.profiler._long_context_device_label('PC: GPU/0*?'), "PC_GPU_0")

    def test_optional_memory_counter_failure(self) -> None:
        """Unavailable counters remain empty on CUDA or CPU. 不可用内存计数器留空。"""
        self.profiler.device.type = "cuda"
        self.module.torch.cuda.max_memory_allocated = mock.Mock(side_effect=RuntimeError("unavailable"))
        self.assertIsNone(self.profiler._long_context_cuda_stat("max_memory_allocated"))
        self.profiler.device.type = "cpu"
        self.assertIsNone(self.profiler._long_context_cuda_stat("max_memory_allocated"))
        self.module.torch.cuda.max_memory_allocated.assert_called_once()

    def test_trial_timer_boundaries_and_last_position_head(self) -> None:
        """Use a continuous timer and read memory after its endpoint. 验证连续计时与末位置 head。"""
        events: list[str] = []

        class Tensor:
            """Trace minimal tensor operations. 跟踪最小张量操作。"""
            shape = (1, 12, 8)

            def to(self, **options: Any) -> Tensor:
                """Record a local copy; options select device and dtype. 记录本机设备和精度转换。"""
                events.append("transfer")
                return self

            def __getitem__(self, index: Any) -> Tensor:
                """Record the head slice; index selects the final input position. 记录末位置切片。"""
                events.append("slice")
                self.indices.append(index)
                return self

            def item(self) -> int:
                """Materialize the first token ID. 输出首 token ID。"""
                events.append("first_token")
                return 42

        tensor = Tensor()
        tensor.shape = (1, 12)
        tensor.indices = []
        components = {
            "tokenizer": types.SimpleNamespace(apply_chat_template=mock.Mock(side_effect=lambda *a, **kw: events.append("tokenize") or tensor)),
            "embedding": mock.Mock(side_effect=lambda *a: events.append("embedding") or tensor),
            "rope": mock.Mock(side_effect=lambda *a: events.append("rope") or (tensor, tensor)),
            "shard": mock.Mock(side_effect=lambda *a, **kw: events.append("shard") or tensor),
            "lm_head": mock.Mock(side_effect=lambda *a: events.append("head") or tensor),
        }
        self.module.build_position_ids = mock.Mock(return_value=tensor)
        self.module.torch.argmax = mock.Mock(return_value=tensor)
        self.profiler.device.type = "cuda"
        self.profiler._synchronize_device = mock.Mock(side_effect=lambda: events.append("sync"))
        self.module.torch.cuda.reset_peak_memory_stats.side_effect = lambda *a: events.append("reset_peak")
        self.profiler._long_context_cuda_stat = mock.Mock(side_effect=lambda *a: events.append("memory") or 100)
        with mock.patch.object(self.module.time, "perf_counter", side_effect=lambda: events.append("timer") or float(len(events))):
            result = self.module.NodeProfiler._long_context_run_trial(
                self.profiler, components, "context", "question", "llama3.2ins", True, {},
            )
        self.assertEqual(events[:4], ["sync", "reset_peak", "timer", "tokenize"])
        self.assertEqual(events[-4:], ["sync", "timer", "memory", "memory"])
        self.assertEqual(events.count("timer"), 2)
        self.assertEqual(events.count("sync"), 2)
        self.assertEqual(result["first_token_id"], 42)
        prompt_args = components["tokenizer"].apply_chat_template.call_args
        self.assertEqual(prompt_args.args[0][0]["content"], "context\n\n\n\nquestion")
        self.assertFalse(prompt_args.kwargs["truncation"])
        components["shard"].assert_called_once()
        self.assertNotIn("attention_mask", components["shard"].call_args.kwargs)
        self.assertEqual(components["lm_head"].call_count, 1)
        self.assertEqual(tensor.indices[0], (slice(None), slice(-1, None), slice(None)))

    def test_each_trial_uses_new_cache_and_hidden_endpoint(self) -> None:
        """Independent hidden-state trials create new KV caches and skip head execution. 验证每轮独立 KV。"""
        tensor = mock.Mock()
        tensor.shape = (1, 12)
        components = {"tokenizer": mock.Mock(), "embedding": mock.Mock(return_value=tensor),
                      "rope": mock.Mock(return_value=(tensor, tensor)),
                      "shard": mock.Mock(return_value=tensor), "lm_head": mock.Mock()}
        components["tokenizer"].apply_chat_template.return_value = tensor
        self.profiler._synchronize_device = mock.Mock()
        for _ in range(2):
            result = self.module.NodeProfiler._long_context_run_trial(
                self.profiler, components, "context", "question", "llama3.2ins", False, {},
            )
            self.assertEqual(result["status"], "success")
            self.assertIsNone(result["first_token_id"])
        caches = [call.kwargs["past_key_value"] for call in components["shard"].call_args_list]
        self.assertIsNot(caches[0], caches[1])
        components["lm_head"].assert_not_called()

    def test_real_trial_rejects_oversized_input_before_forward(self) -> None:
        """Check model input limit after templating and before tensor transfer. tokenize 后检查长度。"""
        components = {"tokenizer": mock.Mock()}
        components["tokenizer"].apply_chat_template.return_value.shape = (1, 1001)
        self.profiler._synchronize_device = mock.Mock()
        result = self.module.NodeProfiler._long_context_run_trial(
            self.profiler, components, "text", "question", "llama3.2ins", False, {},
        )
        self.assertEqual(result["status"], "context_limit")
        components["tokenizer"].apply_chat_template.return_value.to.assert_not_called()

    def test_plot_reads_csv_without_model_instance(self) -> None:
        """Plot raw repeats and skip warm-up with mocked matplotlib. 绘图不创建模型实例。"""
        result = self.run_profile(layer_num=4, initial_context_length=999, repeat_num=2)
        output, figure, axis, pyplot = plot_with_mocked_backend(self.module, result)
        self.assertEqual(output, result.with_suffix(".png"))
        self.assertEqual(axis.plot.call_count, 2)
        self.assertEqual(axis.fill_between.call_count, 2)
        figure.savefig.assert_called_once_with(output, dpi=150)
        pyplot.close.assert_called_once_with(figure)

    def test_plot_incomplete_tail_and_pending_attempt(self) -> None:
        """Plot durable rows and annotate unfinished work without asserting OOM. 绘制已完成记录并标记未完成轮次。"""
        result = self.run_profile(initial_context_length=999, repeat_num=1)
        with result.open("a", encoding="utf-8") as stream:
            stream.write("3,incomplete")
        metadata = {"pending_attempt": {"attempt_id": 3, "context_length": 10, "context_length_unit": "words"}}
        result.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
        _, _, axis, _ = plot_with_mocked_backend(self.module, result)
        self.assertIn("cause unconfirmed", axis.text.call_args.args[2])
        self.assertEqual(axis.plot.call_count, 1)

    def test_plot_optional_metadata_validation(self) -> None:
        """CSV data remains usable when optional metadata is malformed or stale. 元数据异常时仍可使用 CSV。"""
        result = self.run_profile(initial_context_length=999, repeat_num=1)
        for metadata in ([], {"pending_attempt": "invalid"}, {"pending_attempt": {"attempt_id": 2}}):
            with self.subTest(metadata=metadata):
                result.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
                _, _, axis, _ = plot_with_mocked_backend(self.module, result)
                axis.text.assert_not_called()

    def test_plot_with_no_measurements(self) -> None:
        """An empty measured dataset produces a clear error. 无正式测量数据时报明确错误。"""
        self.profiler._long_context_run_trial.return_value = {"status": "context_limit", "input_token_count": 2000}
        result = self.run_profile()
        with self.assertRaisesRegex(ValueError, "no completed measured"):
            self.module.NodeProfiler.plot_long_context_prefill(result)

    def test_python_sources_compile_without_importing_dependencies(self) -> None:
        """Parse and compile project Python files without loading ML dependencies. 静态检查项目代码。"""
        files = [*PROJECT_ROOT.glob("*.py"), *PROJECT_ROOT.glob("utils/*.py"), *PROJECT_ROOT.glob("test/*.py")]
        for path in files:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8-sig")
                compile(ast.parse(source, filename=str(path)), str(path), "exec")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--crash-probe":
        crash_probe(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
