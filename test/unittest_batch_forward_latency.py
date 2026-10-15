"""Regression tests for B/L aggregation, fitting and equal-latency recommendations.

检查 B/L 聚合、拟合与等时延推荐；这些测试不加载模型、不执行 GPU 前向。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.batch_forward_latency import (
    TRIAL_FIELDS, iter_dataset_records, parse_arguments, read_token_bank, summarize_trials,
)
from analysis.batch_latency_balance import (
    find_equivalent_batches, fit_latency_rule, predict_latency,
    recommend_equivalent_batch, relative_latency_gap,
)


def make_point(batch_size: int, prompt_tokens: int, latency_ms: float, cv: float = 0.001) -> dict:
    """Build one complete measured-shape fixture.

    Args:
        batch_size: Request count recorded in the fixture.
            测试测量点记录的请求数。
        prompt_tokens: Exact token length recorded in the fixture.
            测试测量点记录的严格 token 长度。
        latency_ms: Positive median latency supplied as the expected measurement.
            作为预期测量值提供的正时延中位数。
        cv: Repeat variability used to exercise recommendation quality gates.
            重复测量变异系数，用来测试推荐质量门槛。
    """
    return {"batch_size": batch_size, "prompt_tokens": prompt_tokens,
            "median_ms": latency_ms, "cv": cv, "repeat_count": 7, "status": "success"}


class BalanceRuleTests(unittest.TestCase):
    """Check model acceptance, measured matches and bounded integer suggestions.

    检查拟合采纳条件、实测匹配与整数请求数推荐范围。
    """

    def test_full_cost_formula_is_recovered(self) -> None:
        """Recover a nonnegative synthetic formula using held-out complete shapes.

        使用整种形状留出的验证方法恢复已知非负合成时延函数。
        """
        points = [make_point(batch, length, 2 + batch * (0.2 + 0.003 * length + 0.00001 * length**2))
                  for batch in (1, 2, 4) for length in (128, 256, 512)]
        rule = fit_latency_rule(points)
        self.assertTrue(rule["accepted"])
        self.assertEqual(rule["method"], "token_count_and_attention")
        self.assertEqual(len(rule["validation"]["validation_predictions_ms"]), len(points))
        self.assertAlmostEqual(predict_latency(3, 384, rule), 2 + 3 * (0.2 + 0.003 * 384 + 0.00001 * 384**2))
        for batch, length in ((5, 256), (1, 1024), (1, 64)):
            with self.subTest(batch=batch, length=length), self.assertRaises(ValueError):
                predict_latency(batch, length, rule)

    def test_simple_formula_is_preferred_when_valid(self) -> None:
        """Prefer the lower-parameter model when BL already explains all observations.

        当 BL 已经解释全部观测时，优先选择参数更少的函数。
        """
        points = [make_point(batch, length, 1 + 0.01 * batch * length)
                  for batch in (1, 2, 4) for length in (128, 256, 512)]
        rule = fit_latency_rule(points)
        self.assertTrue(rule["accepted"])
        self.assertEqual(rule["method"], "token_count")

    def test_sparse_noisy_and_irregular_data_use_lookup(self) -> None:
        """Fall back to lookup when a function lacks sufficient or reliable support.

        数据不足、波动过大或预测误差过大时使用实测查表。
        """
        self.assertFalse(fit_latency_rule([make_point(1, 128, 10)])["accepted"])
        points = [make_point(batch, length, 1 + 0.01 * batch * length, cv=0.1)
                  for batch in (1, 2, 4) for length in (128, 256, 512)]
        self.assertFalse(fit_latency_rule(points)["accepted"])
        for index, point in enumerate(points):
            point.update(cv=0.001, median_ms=10.0 if index % 2 else 100.0)
        self.assertFalse(fit_latency_rule(points)["accepted"])

    def test_measured_pairs_require_tradeoff_and_stability(self) -> None:
        """Export only stable near-equal pairs with opposite changes in B and L.

        只导出 B 与 L 反向变化且时延稳定、接近的实测组合。
        """
        points = [make_point(1, 512, 10), make_point(4, 128, 10.2),
                  make_point(2, 256, 10.1, cv=0.2), make_point(1, 128, 10.1)]
        pairs = find_equivalent_batches(points)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["relative_gap"], 0.2 / 10.1)
        self.assertAlmostEqual(relative_latency_gap(10, 10.2), relative_latency_gap(10.2, 10))

    def test_recommendation_prefers_stable_measured_match(self) -> None:
        """Choose a stable observed alternative even when a closer noisy point exists.

        存在更接近但波动较大的点时，仍选择满足容差的稳定实测点。
        """
        points = [make_point(1, 512, 10), make_point(4, 128, 10.01, cv=0.1),
                  make_point(3, 128, 10.2)]
        rule = fit_latency_rule(points)
        recommendation = recommend_equivalent_batch(rule, 1, 512, 128)
        self.assertEqual(recommendation["source"], "measured")
        self.assertEqual(recommendation["batch_size"], 3)
        self.assertTrue(recommendation["within_tolerance"])

    def test_model_estimates_do_not_override_measured_mismatches(self) -> None:
        """Respect existing observations when an inverse formula proposes a known shape.

        反函数提出已测形状时，尊重其实测不匹配结果。
        """
        rule = {"accepted": True, "tolerance": 0.05,
                "coefficients": {"c0": 0, "cB": 0, "cBL": 0.01, "cBL2": 0},
                "points": [make_point(1, 1000, 10), make_point(2, 500, 12), make_point(4, 1000, 40)]}
        result = recommend_equivalent_batch(rule, 1, 1000, 500)
        self.assertFalse(result["within_tolerance"])
        self.assertEqual(result["source"], "none")

    def test_integer_prediction_is_labeled_as_an_estimate(self) -> None:
        """Mark unmeasured inverse-rule suggestions as needing a confirming measurement.

        未实测的反函数推荐明确标记为预测，并要求测量确认。
        """
        rule = {"accepted": True, "tolerance": 0.05,
                "coefficients": {"c0": 1, "cB": 0, "cBL": 0.01, "cBL2": 0},
                "points": [make_point(1, 768, 8.68), make_point(4, 1024, 41.96),
                           make_point(4, 128, 6.12)]}
        result = recommend_equivalent_batch(rule, 1, 768, 256)
        self.assertEqual(result["source"], "model_estimate")
        self.assertEqual(result["batch_size"], 3)
        self.assertTrue(result["requires_measurement"])
        self.assertTrue(result["within_tolerance"])


class DataPreparationTests(unittest.TestCase):
    """Check bounded dataset reading, exact token counts and partial-run aggregation.

    检查流式数据读取、严格 token 数与中断运行的结果聚合。
    """

    def test_streaming_and_exact_token_bank_selection(self) -> None:
        """Skip short records and retain distinct long prefixes without fabricating text.

        跳过过短记录并保留不同长文本的前缀，不重复拼接文本凑长度。
        """
        class CharacterTokenizer:
            """Use character IDs as a deterministic token-count fixture.

            使用字符 ID 作为确定性的 token 数测试分词器。
            """

            def encode(self, text: str, add_special_tokens: bool, truncation: bool, max_length: int) -> list[int]:
                """Encode bounded fixture characters.

                Args:
                    self: Tokenizer fixture instance used for deterministic conversion.
                        用于确定性转换的测试分词器实例。
                    text: Fixture context whose characters provide token IDs.
                        为 token ID 提供字符的测试 context。
                    add_special_tokens: Interface flag; this fixture adds no special tokens.
                        接口开关；测试分词器不添加特殊 token。
                    truncation: Interface flag indicating that prefixes may be truncated.
                        表示允许截断前缀的接口开关。
                    max_length: Exact upper limit used when retaining fixture token IDs.
                        保留测试 token ID 时使用的严格上限。
                """
                return [ord(character) for character in text[:max_length]]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps([{"id": "short", "context": "abc"},
                                        {"id": "one", "context": "abcdefghij"},
                                        {"id": "two", "context": "klmnopqrst"}]), encoding="utf-8")
            self.assertEqual(len(list(iter_dataset_records(path))), 3)
            bank, manifest = read_token_bank(path, CharacterTokenizer(), 0, 2, 8)
            self.assertEqual([len(tokens) for tokens in bank], [8, 8])
            self.assertEqual([record["sample_id"] for record in manifest], ["one", "two"])
            self.assertNotEqual(bank[0], bank[1])

    def test_warmups_failures_and_partial_shapes_are_excluded(self) -> None:
        """Keep only complete, successful repeats eligible for fitting and plotting.

        只有重复次数完整且无失败的测量点才进入拟合与绘图。
        """
        metadata = {"settings": {"batch_sizes": [1, 2, 4], "prompt_lengths": [128], "repeats": 3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                writer.writerow(dict(batch_size=1, prompt_tokens=128, phase="warmup", repeat_index=0,
                                     status="success", latency_ms=9999))
                for repeat_index, latency in enumerate((10, 11, 12)):
                    writer.writerow(dict(batch_size=1, prompt_tokens=128, phase="measure",
                                         repeat_index=repeat_index, status="success", latency_ms=latency))
                writer.writerow(dict(batch_size=2, prompt_tokens=128, phase="measure", repeat_index=0,
                                     status="success", latency_ms=20))
                writer.writerow(dict(batch_size=4, prompt_tokens=128, phase="prepare", repeat_index=0,
                                     status="oom"))
            summaries = summarize_trials(Path(directory), metadata)
        self.assertEqual([point["status"] for point in summaries], ["success", "partial", "oom"])
        self.assertEqual(summaries[0]["median_ms"], 11)
        self.assertEqual(summaries[0]["repeat_count"], 3)
        self.assertTrue(math.isfinite(summaries[0]["cv"]))

    def test_cli_deduplicates_grid_and_rejects_invalid_counts(self) -> None:
        """Normalize grid sizes and reject invalid settings before importing ML packages.

        在导入 ML 包前整理网格，并拒绝非法实验设置。
        """
        arguments = parse_arguments(["--batch-sizes", "4", "1", "4", "--prompt-lengths", "512", "128"])
        self.assertEqual(arguments.batch_sizes, [1, 4])
        self.assertEqual(arguments.prompt_lengths, [128, 512])
        for option in (["--repeats", "1"], ["--batch-sizes", "0"], ["--relative-tolerance", "2"]):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_arguments(option)


if __name__ == "__main__":
    unittest.main()
