"""Standard-library regression tests for long-context profiling, with mocked ML APIs.

Run with ``python -B test/unittest_node_profiler.py`` from the repository root.
Model weights, torch and CUDA are not required.
从项目根目录运行 ``python -B test/unittest_node_profiler.py``；不需要权重、torch 或 CUDA。
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any

# Keep the test-only directory out of the standard-library import search.
# Direct execution adds this directory to sys.path; application entry points live elsewhere.
# 导入标准库时排除测试专用目录；直接运行会把该目录放入 sys.path，主程序入口位于其他目录。
TEST_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIRECTORY.parent
_original_import_paths = sys.path[:]
try:
    sys.path = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != TEST_DIRECTORY]
    import unittest
    from unittest import mock
finally:
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


def load_plot_module(filename: str = "long_context_prefill_result_fig.py") -> types.ModuleType:
    """Import the standalone plotting script using only its standard-library imports.
    仅通过脚本自身的标准库依赖导入独立绘图模块。

    Args:
        filename: Script basename in the analysis directory. analysis 目录中的绘图脚本文件名。
    """
    spec = importlib.util.spec_from_file_location(
        "long_context_plot_under_test", PROJECT_ROOT / "analysis" / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot locate {filename}")
    module = importlib.util.module_from_spec(spec)
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


def plot_with_mocked_backend(
    module: types.ModuleType, csv_path: Path, output_path: Path | None = None,
) -> tuple[Path, Any, Any, Any]:
    """Exercise CSV processing with a recording plotting backend.

    Args:
        module: Standalone plotting module. 独立绘图模块。
        csv_path: Input CSV, optionally accompanied by metadata. 输入 CSV 及可选元数据。
        output_path: Optional explicit figure destination. 可选的显式图片输出路径。
    """
    pyplot = types.ModuleType("matplotlib.pyplot")
    figure, axis = mock.MagicMock(), mock.MagicMock()
    pyplot.subplots = mock.Mock(return_value=(figure, axis))
    pyplot.close = mock.Mock()
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = mock.Mock()
    matplotlib.pyplot = pyplot
    with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
        output = module.plot_long_context_prefill(csv_path, output_path=output_path)
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
        self.plot_module = load_plot_module()

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

    def jetson_info(
        self, model: str = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
        module_id: str | None = None, total_gib: float | None = 7.4,
    ) -> dict[str, Any]:
        """Read Jetson metadata with controlled system evidence.

        Args:
            self: Test instance owning the profiler. 当前 profiler 测试实例。
            model: Device-tree model text returned by the system. 模拟系统型号文本。
            module_id: Physical module ID, or None when unavailable. 模拟模块 ID，None 表示不可用。
            total_gib: OS-visible physical RAM in GiB, or None when unavailable. 系统总内存 GiB 或 None。
        """
        total_bytes = int(total_gib * 1024 ** 3) if total_gib is not None else None
        with mock.patch.object(Path, "read_text", return_value=model), mock.patch.object(
            self.profiler, "_long_context_jetson_module_id", return_value=(module_id, "eeprom:/test/eeprom"),
        ), mock.patch.object(self.profiler, "_long_context_system_memory_bytes", return_value=total_bytes):
            return self.profiler._long_context_jetson_info()

    def test_jetson_variant_system_evidence_priority(self) -> None:
        """Prefer physical SKUs and explicit model variants over RAM-based inference.
        self 为当前测试实例；验证模块 SKU、系统型号与内存推断的优先级。
        """
        for module_id, family, variant in (
            ("P3767-0003", "Jetson Orin Nano", "8G"),
            ("P3767-0004", "Jetson Orin Nano", "4G"),
            ("P3767-0005", "Jetson Orin Nano", "8G"),
            ("P3767-0000", "Jetson Orin NX", "16G"),
            ("P3767-0001", "Jetson Orin NX", "8G"),
        ):
            with self.subTest(module_id=module_id):
                info = self.jetson_info(model="NVIDIA Jetson Orin Nano 4GB", module_id=module_id, total_gib=3.7)
                self.assertEqual(info["jetson_model"], family)
                self.assertEqual(info["jetson_memory_variant"], variant)
                self.assertEqual(info["jetson_memory_variant_source"], "eeprom:/test/eeprom")
        info = self.jetson_info(model="NVIDIA Jetson Orin Nano 4GB", total_gib=7.4)
        self.assertEqual(info["jetson_memory_variant"], "4G")
        self.assertTrue(info["jetson_memory_variant_source"].startswith("device_tree_model:"))
        self.assertEqual(self.jetson_info(module_id="P3767-0005", total_gib=None)["jetson_memory_variant"], "8G")
        self.assertIsNone(self.jetson_info(module_id="P3767-0009")["jetson_memory_variant"])

    def test_jetson_variant_uses_nominal_memory_ranges(self) -> None:
        """Classify 7.x/3.x GiB without using free RAM or guessing unusual totals.
        self 为当前测试实例；验证标称容量余量、异常容量和非 Jetson 设备。
        """
        for total_gib, variant in ((7.1, "8G"), (7.9, "8G"), (6.2, "8G"),
                                   (3.1, "4G"), (3.9, "4G"), (5.5, None), (16, None), (0, None), (None, None)):
            with self.subTest(total_gib=total_gib):
                info = self.jetson_info(total_gib=total_gib)
                self.assertEqual(info["jetson_memory_variant"], variant)
                self.assertEqual(info["jetson_memory_variant_source"],
                                 "system_memory_total_inferred" if variant else "unknown")
                self.assertEqual(info["system_memory_total_bytes"],
                                 int(total_gib * 1024 ** 3) if total_gib is not None else None)
        self.assertEqual(self.jetson_info(model="NVIDIA Quadro RTX 5000", total_gib=7.4), {})

    def test_system_memory_reader_and_fallbacks(self) -> None:
        """Use MemTotal in KiB and physical pages while tolerating missing Linux interfaces.
        self 为当前测试实例；验证总内存单位、sysconf 回退和接口不可用。
        """
        with mock.patch.object(Path, "read_text", return_value="MemTotal:  7800000 kB\nMemAvailable:  1 kB\n"), \
                mock.patch.object(self.module.os, "sysconf", create=True) as sysconf:
            self.assertEqual(self.profiler._long_context_system_memory_bytes(), 7800000 * 1024)
            sysconf.assert_not_called()
        with mock.patch.object(Path, "read_text", side_effect=PermissionError()), mock.patch.object(
            self.module.os, "sysconf", create=True, side_effect=[2000000, 4096],
        ):
            self.assertEqual(self.profiler._long_context_system_memory_bytes(), 2000000 * 4096)
        for error in (AttributeError(), OSError(), ValueError()):
            with mock.patch.object(Path, "read_text", return_value="MemTotal: invalid"), mock.patch.object(
                self.module.os, "sysconf", create=True, side_effect=error,
            ):
                self.assertIsNone(self.profiler._long_context_system_memory_bytes())

    def test_jetson_eeprom_module_id_and_permissions(self) -> None:
        """Read supported EEPROM part numbers and tolerate unsupported formats or access errors.
        self 为当前测试实例；验证 EEPROM 版本、模块/载板区分和读取权限。
        """
        eeprom_path = Path("/sys/bus/i2c/devices/0-0050/eeprom")
        for header, part_number, expected in (
            (b"\x02\x00", b"699-13767-0005-300 A.0", "P3767-0005"),
            (b"\x02\x00", b"699-13767-0004-300 A.0", "P3767-0004"),
            (b"\x02\x01", b"699-13767-0005-300 A.0", None),
            (b"\x01\x00", b"699-13767-0005-300 A.0", None),
            (b"\x02\x00", b"699-13768-0000-300 A.0", None),
        ):
            with self.subTest(part=part_number, header=header), mock.patch.object(
                Path, "glob", return_value=[eeprom_path],
            ), mock.patch.object(Path, "open", return_value=io.BytesIO(header + bytes(18) + part_number)), \
                    mock.patch.object(Path, "iterdir", side_effect=PermissionError()):
                module_id, source = self.profiler._long_context_jetson_module_id((Path("/proc/device-tree"),))
                self.assertEqual(module_id, expected)
                self.assertEqual(source, f"eeprom:{eeprom_path.as_posix()}" if expected else None)
        with mock.patch.object(Path, "glob", return_value=[eeprom_path]), mock.patch.object(
            Path, "open", side_effect=PermissionError(),
        ), mock.patch.object(Path, "iterdir", side_effect=OSError()):
            self.assertEqual(self.profiler._long_context_jetson_module_id((Path("/proc/device-tree"),)), (None, None))

    def test_jetson_device_tree_ids_and_model_fallback(self) -> None:
        """Use boot IDs or the alternate device-tree root; shared compatible text is not a SKU.
        self 为当前测试实例；验证启动 ID、备用设备树位置以及共用 DTB 的处理。
        """
        root = Path("/proc/device-tree")
        with mock.patch.object(Path, "glob", side_effect=PermissionError()), mock.patch.object(
            Path, "iterdir", return_value=[root / "chosen/plugin-manager/ids/3767-0004-300"],
        ):
            module_id, source = self.profiler._long_context_jetson_module_id((root,))
            self.assertEqual(module_id, "P3767-0004")
            self.assertTrue(source.startswith("device_tree_ids:"))

        def system_file(path: Path, **options: Any) -> str:
            """Return selected simulated system files.

            Args:
                path: Requested system path. 请求的系统文件路径。
                options: Path.read_text options, unused by the fixture. 测试中未使用的文件读取选项。
            """
            contents = {
                "/sys/firmware/devicetree/base/model": "NVIDIA Jetson Orin Nano\x00",
                "/sys/firmware/devicetree/base/compatible": "nvidia,p3767-0000\x00nvidia,tegra234\x00",
                "/proc/meminfo": "MemTotal: 7800000 kB\nMemAvailable: 1 kB\n",
            }
            if path.as_posix() in contents:
                return contents[path.as_posix()]
            raise FileNotFoundError(str(path))

        with mock.patch.object(Path, "read_text", system_file), mock.patch.object(
            Path, "glob", side_effect=PermissionError(),
        ), mock.patch.object(Path, "iterdir", side_effect=PermissionError()):
            info = self.profiler._long_context_jetson_info()
        self.assertEqual(info["jetson_memory_variant"], "8G")
        self.assertIsNone(info["jetson_module_id"])
        self.assertEqual(info["system_memory_total_bytes"], 7800000 * 1024)

    def test_jetson_metadata_saved_before_model_loading(self) -> None:
        """Persist detected capacity and provenance before any model loading or timed forward.
        self 为当前测试实例；验证设备版本元数据在装载前持久化，且不会改变文件标签覆盖行为。
        """
        detected_info = self.jetson_info(total_gib=7.4)

        def inspect_metadata(layer_num: int) -> dict[str, Any]:
            """Inspect the saved hardware record before returning mock components.

            Args:
                layer_num: Requested Transformer layer count. 请求加载的 Transformer 层数。
            """
            metadata_path = next(self.directory.glob("*/long_context*.json"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field, value in detected_info.items():
                self.assertEqual(metadata[field], value)
            self.assertEqual(metadata["layer_num"], layer_num)
            self.assertEqual(metadata["stage"], "load_components")
            return {"shard": types.SimpleNamespace(config=self.profiler.config)}

        self.profiler._long_context_load_components.side_effect = inspect_metadata
        with mock.patch.object(self.profiler, "_long_context_jetson_info", return_value=detected_info):
            result = self.run_profile(initial_context_length=999, repeat_num=1, device_label="edge-A")
        metadata = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["jetson_memory_variant"], "8G")
        self.assertEqual(metadata["device_label"], "edge-A")

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

    def test_plot_reads_csv_with_standalone_module(self) -> None:
        """Plot repeats with ML imports blocked and matplotlib mocked. 禁用 ML 导入后独立绘图。

        Args:
            self: Test instance containing synthetic CSV results. 持有模拟 CSV 结果的测试实例。
        """
        result = self.run_profile(layer_num=4, initial_context_length=999, repeat_num=2)
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "utils": None}):
            output, figure, axis, pyplot = plot_with_mocked_backend(self.plot_module, result)
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
        _, _, axis, _ = plot_with_mocked_backend(self.plot_module, result)
        self.assertIn("cause unconfirmed", axis.text.call_args.args[2])
        self.assertEqual(axis.plot.call_count, 1)

    def test_plot_optional_metadata_validation(self) -> None:
        """CSV data remains usable when optional metadata is malformed or stale. 元数据异常时仍可使用 CSV。"""
        result = self.run_profile(initial_context_length=999, repeat_num=1)
        for metadata in ([], {"pending_attempt": "invalid"}, {"pending_attempt": {"attempt_id": 2}}):
            with self.subTest(metadata=metadata):
                result.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
                _, _, axis, _ = plot_with_mocked_backend(self.plot_module, result)
                axis.text.assert_not_called()

    def test_plot_with_no_measurements(self) -> None:
        """An empty measured dataset produces a clear error. 无正式测量数据时报明确错误。"""
        self.profiler._long_context_run_trial.return_value = {"status": "context_limit", "input_token_count": 2000}
        result = self.run_profile()
        with self.assertRaisesRegex(ValueError, "no completed measured"):
            self.plot_module.plot_long_context_prefill(result)

    def test_plot_short_orin_names_and_endpoints(self) -> None:
        """Keep both single-device endpoints distinct while showing compact Orin capacities.
        self 为当前测试实例；验证单设备图的简短型号、内存版本和两个终点。
        """
        result = self.run_profile(layer_num=4, initial_context_length=999, repeat_num=1)
        for variant in ("4G", "8G"):
            result.with_suffix(".json").write_text(json.dumps({
                "device_label": "NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super",
                "jetson_model": "Jetson Orin Nano", "jetson_memory_variant": variant,
            }), encoding="utf-8")
            _, _, axis, _ = plot_with_mocked_backend(self.plot_module, result)
            self.assertEqual([call.kwargs["label"] for call in axis.plot.call_args_list], [
                f"Jetson Orin Nano {variant} | hidden states", f"Jetson Orin Nano {variant} | first token (LM head)",
            ])

    def test_short_device_labels_legacy_and_overrides(self) -> None:
        """Format compact device names from recorded evidence for legacy results.
        self 为当前测试实例；验证旧 JSON、显式标签和无效字段，避免依据当前机器猜测版本。
        """
        result = self.run_profile(initial_context_length=999, repeat_num=1)
        legacy = {"device_label": "NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super"}
        for metadata, expected_name in (
            (legacy, "Jetson Orin Nano"),
            (dict(legacy, jetson_memory_variant="8G"), "Jetson Orin Nano 8G"),
            (dict(legacy, jetson_model={"invalid": True}, jetson_memory_variant=[]), "Jetson Orin Nano"),
            ({"device_label": "NVIDIA_Quadro_RTX_5000"}, "Quadro RTX 5000"),
            ({"device_label": "NVIDIA_Jetson_Orin_Nano_8GB"}, "Jetson Orin Nano 8G"),
            (dict(legacy, jetson_model="Jetson Orin NX", jetson_memory_variant="16G"), "Jetson Orin NX 16G"),
        ):
            with self.subTest(expected_name=expected_name):
                result.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
                _, _, axis, _ = plot_with_mocked_backend(self.plot_module, result)
                self.assertEqual(axis.plot.call_args.kwargs["label"], f"{expected_name} | hidden states")

    def test_plot_relative_paths_use_project_root(self) -> None:
        """Resolve relative input and output paths against the project root.

        Args:
            self: Test instance supplying an isolated project root. 提供隔离根目录的测试实例。
        """
        result = self.run_profile(initial_context_length=999, repeat_num=1)
        with mock.patch.object(self.plot_module, "PROJECT_ROOT", self.directory):
            output, figure, _, _ = plot_with_mocked_backend(
                self.plot_module, result.relative_to(self.directory), Path("figures/prefill.png"),
            )
        self.assertEqual(output, self.directory / "figures/prefill.png")
        figure.savefig.assert_called_once_with(output, dpi=150)

    def test_plot_cli_default_and_explicit_paths(self) -> None:
        """Keep the configured default CSV and accept explicit CLI paths.

        Args:
            self: Test instance used for parser assertions. 用于校验参数解析的测试实例。
        """
        with mock.patch.object(self.plot_module, "plot_long_context_prefill", return_value=Path("prefill.png")) as plot:
            with mock.patch("builtins.print") as print_result:
                self.plot_module.main([])
                plot.assert_called_once_with(self.plot_module.DEFAULT_CSV_PATH, output_path=None)
                self.plot_module.main(["sample.csv", "--output", "figures/sample.png"])
                plot.assert_called_with("sample.csv", output_path="figures/sample.png")
                self.assertEqual(print_result.call_count, 2)

    def test_plot_help_runs_without_ml_dependencies(self) -> None:
        """Start the plotting script outside the repository with ML imports blocked.

        Args:
            self: Test instance supplying an unrelated working directory. 提供外部工作目录的测试实例。
        """
        probe = (
            "import runpy, sys; "
            "sys.modules.update({name: None for name in ('torch', 'transformers', 'utils', 'matplotlib', 'numpy')}); "
            "sys.argv = [sys.argv[1], '--help']; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        process = subprocess.run(
            [sys.executable, "-B", "-c", probe, str(PROJECT_ROOT / "analysis/long_context_prefill_result_fig.py")],
            cwd=self.directory, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--output", process.stdout)

    def test_python_sources_compile_without_importing_dependencies(self) -> None:
        """Parse and compile project Python files without loading ML dependencies. 静态检查项目代码。"""
        files = [*PROJECT_ROOT.glob("*.py"), *PROJECT_ROOT.glob("utils/*.py"),
                 *PROJECT_ROOT.glob("test/**/*.py"), *PROJECT_ROOT.glob("analysis/*.py")]
        for path in files:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8-sig")
                compile(ast.parse(source, filename=str(path)), str(path), "exec")

    def test_standard_library_imports_from_application_directories(self) -> None:
        """Check fresh imports from profiling and pipeline entry-point directories.

        Args:
            self: Test instance used for subprocess assertions.
                用于验证新进程结果的测试实例。

        A fresh process avoids an already cached unittest hiding a path collision.
        新进程避免已缓存的标准库 unittest 掩盖路径上的同名模块冲突。
        """
        probe = (
            "import json, unittest; from unittest import mock; "
            "assert mock.Mock(return_value=7)() == 7; "
            "assert hasattr(unittest, 'TestCase'); "
            "assert not hasattr(unittest, 'LongContextTests'); "
            "print(json.dumps(unittest.__file__))"
        )
        for directory in (PROJECT_ROOT, PROJECT_ROOT / "test", PROJECT_ROOT / "analysis"):
            with self.subTest(directory=directory):
                process = subprocess.run(
                    [sys.executable, "-B", "-c", probe], cwd=directory,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(Path(json.loads(process.stdout)).resolve(), Path(unittest.__file__).resolve())


class PrefillComparisonTests(unittest.TestCase):
    """Check shared-range plotting using synthetic runs and a recording backend.
    使用模拟数据与绘图替身验证共同范围、比较条件和命令行。
    """

    def setUp(self) -> None:
        """Create isolated fixtures; self holds the comparison module and temporary directory.
        为当前测试实例创建隔离目录并加载对比模块。
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.module = load_plot_module("long_context_prefill_comparison_fig.py")
        self.module.PROJECT_ROOT = self.directory

    def write_run(
        self, name: str, lengths: list[int], scale: float = 1.0,
        include_lm_head: bool = False, settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Build a CSV/JSON fixture with three repeats per context length.

        Args:
            self: Test instance owning the temporary directory. 当前测试实例。
            name: CSV basename without suffix; also the device label. CSV 文件名及设备标签。
            lengths: Context word counts to measure. 要生成的原文长度。
            scale: Multiplier for per-layer latency samples. 层均摊时延缩放倍数。
            include_lm_head: Endpoint stored in each row. 每条记录的 LM head 模式。
            settings: Optional CSV run-field overrides. 可选 CSV 实验条件覆盖值。
            metadata: Optional JSON overrides, including deliberately inconsistent values.
                可选 JSON 覆盖值，也用于构造不一致条件。
        """
        source = self.directory / f"{name}.csv"
        conditions = {"sample_id": "sample-a", "model_type": "llama3.2ins",
                      "question_type": "question_nonthinking", "context_length_unit": "words",
                      "loaded_layer_count": 6}
        conditions.update(settings or {})
        rows = []
        for length in lengths:
            for repeat, multiplier in enumerate((3, 1, 2), 1):
                rows.append(dict(conditions, phase="measure", status="success", attempt_id=len(rows) + 1,
                                 repeat_index=repeat, context_length=length, context_word_count=length,
                                 context_char_count=length * 3, input_token_count=length * 2 + 100,
                                 include_lm_head=include_lm_head, is_full_context=False,
                                 prefill_latency_per_layer_ms=length * scale * multiplier))
        self.write_rows(source, rows)
        properties = dict(conditions, layer_num=conditions["loaded_layer_count"], device_label=name,
                          dtype="torch.float16", model_layer_count=28, max_position_embeddings=131072,
                          timing_scope="input-to-output", per_layer_definition="total / N",
                          attention_implementation="eager", attention_mask=None, final_norm_included=False,
                          lm_head_resident=False, warmup_count=1, torch_version="test-version",
                          shards_path="/server/shards/Llama-3___2-3B-Instruct_float16")
        properties.update(metadata or {})
        source.with_suffix(".json").write_text(json.dumps(properties), encoding="utf-8")
        return source

    def write_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Save fixture rows with their column names.

        Args:
            self: Current test instance. 当前测试实例。
            path: Fixture CSV destination. CSV 测试文件路径。
            rows: Nonempty measurement records to serialize. 非空测量记录列表。
        """
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def plot(self, first: Path | None = None, second: Path | None = None, **options: Any) -> tuple[Path, Any, Any]:
        """Run the comparison with matplotlib and ML imports replaced.

        Args:
            self: Test instance owning the comparison module. 当前测试实例。
            first: First fixture CSV, or None to use the configured default. 第一组 CSV；None 使用默认常量。
            second: Second fixture CSV, or None to use the configured default. 第二组 CSV；None 使用默认常量。
            options: Keyword arguments forwarded to compare_long_context_prefill. 转发给绘图函数的参数。
        """
        figure, axis = mock.MagicMock(), mock.MagicMock()
        pyplot = types.SimpleNamespace(subplots=mock.Mock(return_value=(figure, axis)), close=mock.Mock())
        matplotlib = types.SimpleNamespace(use=mock.Mock(), pyplot=pyplot)
        with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot,
                                           "torch": None, "transformers": None, "utils": None}):
            destination = self.module.compare_long_context_prefill(first, second, **options)
        pyplot.close.assert_called_once_with(figure)
        return destination, figure, axis

    def test_shared_upper_bound_and_statistics(self) -> None:
        """Include 2048 in both series, discard larger inputs, and retain median/min/max per-layer values.
        self 为当前测试实例；验证共同上限包含 2048，且统计值已按层均摊。
        """
        lengths = [32, 64, 128, 256, 512, 1024, 2048]
        first = self.write_run("Jetson", lengths, scale=10)
        second = self.write_run("PC", lengths + [4096, 8192])
        output, figure, axis = self.plot(first, second)
        expected_name = (f"long_context_prefill_comparison-{self.module._run_directory_hash(first)}-"
                         f"{self.module._run_directory_hash(second)}.png")
        self.assertEqual(output, self.directory / self.module.DEFAULT_OUTPUT_DIR / expected_name)
        self.assertEqual(axis.plot.call_count, 2)
        for call in axis.plot.call_args_list:
            self.assertEqual(call.args[0], [length * 2 + 100 for length in lengths])
        self.assertEqual(axis.plot.call_args_list[0].args[1], [length * 20 for length in lengths])
        self.assertEqual(axis.plot.call_args_list[1].args[1], [length * 2 for length in lengths])
        self.assertEqual(axis.fill_between.call_args_list[0].args[1], [length * 10 for length in lengths])
        self.assertEqual(axis.fill_between.call_args_list[0].args[2], [length * 30 for length in lengths])
        self.assertIn("32–2048 words", axis.set_title.call_args.args[0])
        figure.savefig.assert_called_once_with(output, dpi=150)

    def test_exact_shared_points_and_single_point(self) -> None:
        """Use only identical prefixes, including a single shared full-context endpoint.
        self 为当前测试实例；验证离散交集及单个共同点。
        """
        first = self.write_run("A", [32, 64, 100])
        second = self.write_run("B", [48, 96, 100])
        for source in (first, second):
            rows = read_rows(source)
            for row in rows:
                row["is_full_context"] = str(int(row["context_length"]) == 100)
            self.write_rows(source, rows)
        _, _, axis = self.plot(first, second)
        self.assertEqual(axis.plot.call_args_list[0].args[0], [300])
        self.assertEqual(axis.plot.call_args_list[1].args[0], [300])

    def test_failed_warmup_and_pending_attempts_are_excluded(self) -> None:
        """Do not expand the measured range using warmup, OOM, incomplete CSV tails or pending JSON.
        self 为当前测试实例；验证异常终止后的测量范围。
        """
        first = self.write_run("A", [32, 64], metadata={"pending_attempt": {"context_length": 512}})
        second = self.write_run("B", [32, 64, 128, 256])
        rows = read_rows(first)
        rows.extend([dict(rows[0], phase="warmup", context_length=512),
                     dict(rows[0], status="oom", context_length=128, prefill_latency_per_layer_ms=""),
                     dict(rows[0], status="context_limit", context_length=256)])
        self.write_rows(first, rows)
        with first.open("a", encoding="utf-8") as stream:
            stream.write("incomplete,tail")
        _, _, axis = self.plot(first, second)
        self.assertEqual(axis.plot.call_args_list[0].args[0], [164, 228])

    def test_endpoint_selection_and_labels(self) -> None:
        """Select one shared endpoint and keep two independently named device series.
        self 为当前测试实例；验证 LM head 选择及图例标签。
        """
        first = self.write_run("A", [32], include_lm_head=True)
        second = self.write_run("B", [32], include_lm_head=True)
        for source in (first, second):
            rows = read_rows(source)
            rows.append(dict(rows[0], include_lm_head=False, prefill_latency_per_layer_ms=999))
            self.write_rows(source, rows)
        output, _, axis = self.plot(first, second, include_lm_head=True, label_a="Jetson", label_b="PC")
        self.assertEqual([call.kwargs["label"] for call in axis.plot.call_args_list], ["Jetson", "PC"])
        self.assertEqual(axis.plot.call_args_list[0].args[1], [64])
        hidden_output, _, hidden_axis = self.plot(first, second)
        self.assertEqual(output, hidden_output)
        self.assertEqual(hidden_axis.plot.call_args_list[0].args[1], [999])

    def test_comparison_short_device_names_and_orin_variant(self) -> None:
        """Use concise device names and preserve known Orin capacity with a custom nickname.
        self 为当前测试实例；验证双设备简短图例及昵称的内存版本。
        """
        first = self.write_run("A", [32], metadata={
            "device_label": "NVIDIA_Jetson_Orin_Nano_Engineering_Reference_Developer_Kit_Super",
            "jetson_model": "Jetson Orin Nano", "jetson_memory_variant": "8G",
        })
        second = self.write_run("B", [32], metadata={"device_label": "NVIDIA_Quadro_RTX_5000"})
        _, _, axis = self.plot(first, second)
        self.assertEqual([call.kwargs["label"] for call in axis.plot.call_args_list],
                         ["Jetson Orin Nano 8G", "Quadro RTX 5000"])
        _, _, axis = self.plot(first, second, label_a="Jetson")
        self.assertEqual(axis.plot.call_args_list[0].kwargs["label"], "Jetson 8G")
        first.with_suffix(".json").write_text(json.dumps({
            **json.loads(first.with_suffix(".json").read_text(encoding="utf-8")),
            "jetson_memory_variant": None,
        }), encoding="utf-8")
        _, _, axis = self.plot(first, second)
        self.assertEqual(axis.plot.call_args_list[0].kwargs["label"], "Jetson Orin Nano")
        _, _, axis = self.plot(first, second, label_a="Jetson Orin Nano 4G")
        self.assertEqual(axis.plot.call_args_list[0].kwargs["label"], "Jetson Orin Nano 4G")

    def test_incompatible_csv_settings(self) -> None:
        """Reject mismatched samples, model adapters, question variants, units and layer counts.
        self 为当前测试实例；逐项检查 CSV 实验条件。
        """
        first = self.write_run("A", [32])
        for field, value in (("sample_id", "sample-b"), ("model_type", "other"),
                             ("question_type", "question_thinking"), ("context_length_unit", "characters"),
                             ("loaded_layer_count", 7)):
            with self.subTest(field=field):
                second = self.write_run("B", [32], settings={field: value})
                with self.assertRaisesRegex(ValueError, field):
                    self.plot(first, second)

    def test_inconsistent_tokens_prefixes_and_run_rows(self) -> None:
        """Reject different inputs and inconsistent repeats rather than merging them.
        self 为当前测试实例；验证输入属性及单 CSV 内混合条件。
        """
        first = self.write_run("A", [32])
        for field, value in (("input_token_count", "999"), ("context_char_count", "999"),
                             ("is_full_context", "True")):
            with self.subTest(field=field):
                second = self.write_run("B", [32])
                rows = read_rows(second)
                self.write_rows(second, [dict(row, **{field: value}) for row in rows])
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.plot(first, second)
        for field, value, message in (("input_token_count", "999", "inconsistent input"),
                                      ("loaded_layer_count", "7", "mixed experiment")):
            second = self.write_run("B", [32])
            rows = read_rows(second)
            rows[-1][field] = value
            self.write_rows(second, rows)
            with self.assertRaisesRegex(ValueError, message):
                self.plot(first, second)

    def test_metadata_compatibility_and_host_paths(self) -> None:
        """Compare workload metadata while tolerating Windows/Linux model path prefixes.
        self 为当前测试实例；验证元数据条件和跨主机路径。
        """
        first = self.write_run("A", [32])
        second = self.write_run("B", [32], metadata={"shards_path": "C:\\weights\\Llama-3___2-3B-Instruct_float16"})
        self.plot(first, second)
        for field, value in (("dtype", "torch.float32"), ("attention_implementation", "sdpa"),
                             ("timing_scope", "different"), ("attention_mask", "causal"),
                             ("model_layer_count", 32), ("shards_path", "/weights/other-model"),
                             ("layer_num", 7)):
            with self.subTest(field=field):
                second = self.write_run("B", [32], metadata={field: value})
                with self.assertRaises(ValueError):
                    self.plot(first, second)
        second = self.write_run("B", [32], metadata={"torch_version": "different-version"})
        with self.assertWarnsRegex(UserWarning, "Different PyTorch versions"):
            self.plot(first, second)

    def test_optional_metadata_and_duplicate_labels(self) -> None:
        """Use CSV-only checks when JSON is missing or invalid, and distinguish identical labels.
        self 为当前测试实例；验证 JSON 回退及同名设备图例。
        """
        first = self.write_run("A", [32])
        second = self.write_run("B", [32])
        second.with_suffix(".json").unlink()
        with self.assertWarnsRegex(UserWarning, "metadata"):
            self.plot(first, second)
        for contents in ("[]", "invalid", "{}"):
            second.with_suffix(".json").write_text(contents, encoding="utf-8")
            with self.assertWarns(UserWarning):
                _, _, axis = self.plot(first, second, label_a="GPU", label_b="GPU")
            self.assertEqual([call.kwargs["label"] for call in axis.plot.call_args_list], ["GPU (CSV A)", "GPU (CSV B)"])

    def test_empty_overlap_missing_endpoint_and_bad_columns(self) -> None:
        """Report unusable inputs before importing matplotlib.
        self 为当前测试实例；验证无交集、无所选终点和缺列错误。
        """
        first = self.write_run("A", [32])
        second = self.write_run("B", [64])
        with self.assertRaisesRegex(ValueError, "No shared"):
            self.plot(first, second)
        with self.assertRaisesRegex(ValueError, "no completed measured"):
            self.plot(first, second, include_lm_head=True)
        with self.assertRaises(TypeError):
            self.plot(first, second, include_lm_head=1)
        second.write_text("status,phase\nsuccess,measure\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing CSV columns"):
            self.plot(first, second)

    def test_invalid_numeric_rows_are_skipped(self) -> None:
        """Skip incomplete or nonfinite repeats without changing valid-point statistics.
        self 为当前测试实例；验证不完整和非有限测量值。
        """
        first = self.write_run("A", [32])
        second = self.write_run("B", [32])
        rows = read_rows(second)
        for value in ("nan", "inf", "-1", ""):
            rows.append(dict(rows[0], prefill_latency_per_layer_ms=value))
        self.write_rows(second, rows)
        with self.assertWarnsRegex(UserWarning, "skipped 4"):
            _, _, axis = self.plot(first, second)
        self.assertEqual(axis.plot.call_args_list[1].args[1], [64])

    def test_relative_paths_and_cli(self) -> None:
        """Resolve all paths at the project root and forward CLI comparison options.
        self 为当前测试实例；验证相对路径与命令行参数。
        """
        first = self.write_run("A", [32])
        second = self.write_run("B", [32])
        with mock.patch.object(self.module, "PROJECT_ROOT", self.directory):
            output, _, _ = self.plot(Path(first.name), Path(second.name), output_path="figures/comparison.png")
        self.assertEqual(output, self.directory / "figures/comparison.png")
        with mock.patch.object(self.module, "compare_long_context_prefill", return_value=output) as plot:
            self.module.main(["A.csv", "B.csv", "--output", "out.png", "--include-lm-head",
                              "--label-a", "Jetson", "--label-b", "PC"])
            plot.assert_called_once_with("A.csv", "B.csv", output_path="out.png", include_lm_head=True,
                                         label_a="Jetson", label_b="PC")

    def test_default_inputs_and_run_hash_output(self) -> None:
        """Use configured CSV defaults and preserve both run-directory hashes in input order.
        self 为当前测试实例；验证默认路径、输出目录、hash 顺序和开头的下划线。
        """
        sources = []
        for name, run_directory in (("A", "20260827-231615-zmq7xa6z"), ("B", "20260828-003453-_krn076g")):
            fixture_directory = self.directory / run_directory
            fixture_directory.mkdir()
            with mock.patch.object(self, "directory", fixture_directory):
                sources.append(self.write_run(name, [32]))
        with mock.patch.multiple(self.module,
                                 DEFAULT_CSV_PATH_A=str(sources[0].relative_to(self.directory)),
                                 DEFAULT_CSV_PATH_B=str(sources[1].relative_to(self.directory))):
            output, figure, _ = self.plot()
        self.assertEqual(output, self.directory / "results/profiling/long_context_prefill/comparisons"
                         / "long_context_prefill_comparison-zmq7xa6z-_krn076g.png")
        figure.savefig.assert_called_once_with(output, dpi=150)
        reverse_output, _, _ = self.plot(sources[1], sources[0])
        self.assertEqual(reverse_output.name, "long_context_prefill_comparison-_krn076g-zmq7xa6z.png")

    def test_cli_default_and_partial_csv_overrides(self) -> None:
        """Accept zero, one or two CSV arguments while retaining explicit overrides.
        self 为当前测试实例；验证省略全部参数、仅覆盖第一组及覆盖两组输入。
        """
        for arguments, expected_a, expected_b in (
            ([], self.module.DEFAULT_CSV_PATH_A, self.module.DEFAULT_CSV_PATH_B),
            (["custom-a.csv"], "custom-a.csv", self.module.DEFAULT_CSV_PATH_B),
            (["custom-a.csv", "custom-b.csv"], "custom-a.csv", "custom-b.csv"),
        ):
            with self.subTest(arguments=arguments), mock.patch.object(
                self.module, "compare_long_context_prefill", return_value=Path("comparison.png"),
            ) as plot:
                self.module.main(arguments)
                plot.assert_called_once_with(expected_a, expected_b, output_path=None,
                                             include_lm_head=False, label_a=None, label_b=None)

    def test_run_hash_fallback_for_other_directories(self) -> None:
        """Use stable short path hashes when the profiling directory convention is unavailable.
        self 为当前测试实例；验证普通目录中不同 CSV 的稳定、安全回退标识。
        """
        first, second = self.directory / "A.csv", self.directory / "B.csv"
        first_hash = self.module._run_directory_hash(first)
        self.assertRegex(first_hash, r"^[0-9a-f]{8}$")
        self.assertEqual(first_hash, self.module._run_directory_hash(first.parent / "." / first.name))
        self.assertNotEqual(first_hash, self.module._run_directory_hash(second))

    def test_help_without_ml_or_plotting_dependencies(self) -> None:
        """Run CLI help from outside the repository with optional imports blocked.
        self 为当前测试实例；验证外部工作目录和轻量命令行入口。
        """
        probe = (
            "import runpy, sys; "
            "sys.modules.update({name: None for name in ('torch', 'transformers', 'utils', 'matplotlib', 'numpy')}); "
            "sys.argv = [sys.argv[1], '--help']; runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        process = subprocess.run(
            [sys.executable, "-B", "-c", probe, str(PROJECT_ROOT / "analysis/long_context_prefill_comparison_fig.py")],
            cwd=self.directory, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--include-lm-head", process.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--crash-probe":
        crash_probe(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
