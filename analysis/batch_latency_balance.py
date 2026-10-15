"""Fit prefill latency and recommend batches with similar measured latency.

拟合 prefill 时延，并推荐实测时延接近的 batch；导入本模块不会启动模型或实验。
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence


def relative_latency_gap(first_ms: float, second_ms: float) -> float:
    """Calculate a symmetric relative difference between two positive latencies.

    Args:
        first_ms: First positive latency in milliseconds, used as one comparison endpoint.
            第一个正时延，单位毫秒，用作比较的一端。
        second_ms: Second positive latency in milliseconds, used as the other endpoint.
            第二个正时延，单位毫秒，用作比较的另一端。
    """
    if not all(math.isfinite(value) and value > 0 for value in (first_ms, second_ms)):
        raise ValueError("Latencies must be finite and positive")
    return abs(first_ms - second_ms) / ((first_ms + second_ms) / 2)


def _features(batch_size: int, prompt_tokens: int) -> list[float]:
    """Build the constant, B, BL and BL-squared explanatory terms.

    Args:
        batch_size: Number of requests in one forward pass; determines B.
            一次前向的请求数，用来构造 B 项。
        prompt_tokens: Exact token count per request; determines L.
            每个请求的严格 token 长度，用来构造 L 项。
    """
    return [1.0, float(batch_size), float(batch_size * prompt_tokens),
            float(batch_size * prompt_tokens * prompt_tokens)]


def _fit_coefficients(
    points: Sequence[dict[str, Any]], feature_indices: tuple[int, ...],
) -> list[float] | None:
    """Fit a small nonnegative least-squares model with column scaling.

    Args:
        points: Successful shape-level median measurements used as fitting observations.
            按形状聚合后的成功测量中位数，用作拟合样本。
        feature_indices: Selected positions in [1, B, BL, BL2]; defines the model family.
            从 [1, B, BL, BL2] 中选择的项，用来指定模型结构。
    """
    import numpy as np

    design = np.asarray([
        [_features(int(point["batch_size"]), int(point["prompt_tokens"]))[index]
         for index in feature_indices] for point in points
    ], dtype=float)
    observed = np.asarray([float(point["median_ms"]) for point in points])
    column_scales = np.max(np.abs(design), axis=0)
    scaled_design = design / column_scales
    if np.linalg.matrix_rank(scaled_design) < len(feature_indices):
        return None
    best_weights = np.zeros(len(feature_indices))
    best_squared_error = float(observed @ observed)
    # Four or fewer features allow exact active-set enumeration without SciPy.
    # 特征最多四项，可枚举非负最小二乘的活跃集合，无需额外安装 SciPy。
    for count in range(1, len(feature_indices) + 1):
        for active_indices in itertools.combinations(range(len(feature_indices)), count):
            active_design = scaled_design[:, active_indices]
            weights, _, rank, _ = np.linalg.lstsq(active_design, observed, rcond=None)
            if rank < count or np.any(weights < -1e-10):
                continue
            candidate = np.zeros(len(feature_indices))
            candidate[list(active_indices)] = np.maximum(weights, 0)
            residual = observed - scaled_design @ candidate
            squared_error = float(residual @ residual)
            if squared_error < best_squared_error:
                best_squared_error, best_weights = squared_error, candidate
    coefficients = [0.0] * 4
    for index, value in zip(feature_indices, best_weights / column_scales):
        coefficients[index] = float(value)
    return coefficients


def fit_latency_rule(
    points: Sequence[dict[str, Any]], tolerance: float = 0.05,
) -> dict[str, Any]:
    """Validate a latency formula by leaving out complete B/L shapes.

    Args:
        points: Successful measurements with batch_size, prompt_tokens, median_ms and cv.
            成功测量点，包含请求数、token 长度、时延中位数与变异系数。
        tolerance: Maximum symmetric relative difference for approximately equal batches.
            判断 batch 时延近似相等的最大对称相对差；也决定拟合误差门槛。
    """
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1")
    observations = [dict(point) for point in points]
    rule: dict[str, Any] = {
        "schema_version": 1,
        "accepted": False,
        "method": "measured_lookup",
        "tolerance": tolerance,
        "formula": "t_ms(B,L) = c0 + B * (cB + cBL*L + cBL2*L**2)",
        "points": observations,
        "candidates": [],
        "validation_error_limit": tolerance / 2,
        "noise_cv_limit": tolerance / 2,
    }
    if (len(observations) < 8
            or len({point["batch_size"] for point in observations}) < 2
            or len({point["prompt_tokens"] for point in observations}) < 3):
        rule["reason"] = "At least eight successful shapes, two B values and three L values are required"
        return rule
    if any(float(point["cv"]) > tolerance / 2 for point in observations):
        rule["reason"] = "Measurement variability exceeds half the requested matching tolerance"
        return rule

    for name, indices in (("token_count", (0, 2)), ("token_count_and_attention", (0, 1, 2, 3))):
        predictions: list[float] = []
        for held_out_index, point in enumerate(observations):
            training = observations[:held_out_index] + observations[held_out_index + 1:]
            coefficients = _fit_coefficients(training, indices)
            if coefficients is None:
                break
            features = _features(int(point["batch_size"]), int(point["prompt_tokens"]))
            predictions.append(sum(weight * feature for weight, feature in zip(coefficients, features)))
        candidate: dict[str, Any] = {"name": name, "parameter_count": len(indices)}
        if len(predictions) != len(observations) or any(value <= 0 for value in predictions):
            candidate.update(accepted=False, reason="Rank-deficient validation fold or nonpositive prediction")
            rule["candidates"].append(candidate)
            continue
        relative_errors = [
            abs(prediction - float(point["median_ms"])) / float(point["median_ms"])
            for prediction, point in zip(predictions, observations)
        ]
        maximum_error = max(relative_errors)
        candidate.update(
            accepted=maximum_error <= tolerance / 2,
            validation="leave_one_shape_out",
            maximum_relative_error=maximum_error,
            mean_relative_error=sum(relative_errors) / len(relative_errors),
            validation_predictions_ms=predictions,
        )
        rule["candidates"].append(candidate)

    accepted = [candidate for candidate in rule["candidates"] if candidate["accepted"]]
    if not accepted:
        rule["reason"] = "No formula met the leave-one-shape-out error limit; use measured CSV lookup"
        return rule
    selected = min(accepted, key=lambda candidate: (candidate["parameter_count"], candidate["maximum_relative_error"]))
    selected_indices = (0, 2) if selected["name"] == "token_count" else (0, 1, 2, 3)
    coefficients = _fit_coefficients(observations, selected_indices)
    if coefficients is None:
        rule["reason"] = "Final design is rank deficient; use measured CSV lookup"
        return rule
    rule.update(
        accepted=True,
        method=selected["name"],
        coefficients=dict(zip(("c0", "cB", "cBL", "cBL2"), coefficients)),
        validation=selected,
        inverse_formula="B_continuous(T,L) = (T-c0)/(cB+cBL*L+cBL2*L**2)",
        reason="Validated locally within the measured shape envelope",
    )
    return rule


def predict_latency(batch_size: int, prompt_tokens: int, rule: dict[str, Any]) -> float:
    """Predict latency inside the calibrated, successfully measured size envelope.

    Args:
        batch_size: Positive integer request count whose forward latency is requested.
            待预测前向时延的正整数请求数。
        prompt_tokens: Positive integer prompt token length whose latency is requested.
            待预测前向时延的正整数 token 长度。
        rule: Accepted fit exported by fit_latency_rule, including measured points.
            已通过验证的拟合规则，包含系数和实测点，用来限制预测范围。
    """
    if not rule.get("accepted"):
        raise ValueError("No accepted function; use measured CSV lookup")
    if batch_size < 1 or prompt_tokens < 1:
        raise ValueError("B and L must be positive")
    points = rule["points"]
    if (batch_size < min(point["batch_size"] for point in points)
            or prompt_tokens < min(point["prompt_tokens"] for point in points)
            or not any(point["batch_size"] >= batch_size and point["prompt_tokens"] >= prompt_tokens
                       for point in points)):
        raise ValueError("Requested shape is outside the successfully measured envelope")
    coefficients = rule["coefficients"]
    return sum(coefficients[name] * feature for name, feature in zip(
        ("c0", "cB", "cBL", "cBL2"), _features(batch_size, prompt_tokens)
    ))


def find_equivalent_batches(
    points: Sequence[dict[str, Any]], tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Find stable measured pairs trading request count against prompt length.

    Args:
        points: Successful shape-level measurements used to compare actual latencies.
            成功的形状级测量点，用于比较实际时延。
        tolerance: Maximum symmetric median difference accepted as a measured match.
            实测中位数近似相等所允许的最大对称相对差。
    """
    pairs = []
    for first, second in itertools.combinations(points, 2):
        if ((first["batch_size"] - second["batch_size"])
                * (first["prompt_tokens"] - second["prompt_tokens"]) >= 0):
            continue
        gap = relative_latency_gap(first["median_ms"], second["median_ms"])
        if gap > tolerance or max(first["cv"], second["cv"]) > tolerance / 2:
            continue
        pairs.append({
            "batch_a_size": first["batch_size"], "prompt_a_tokens": first["prompt_tokens"],
            "latency_a_ms": first["median_ms"], "cv_a": first["cv"],
            "batch_b_size": second["batch_size"], "prompt_b_tokens": second["prompt_tokens"],
            "latency_b_ms": second["median_ms"], "cv_b": second["cv"],
            "relative_gap": gap, "tolerance": tolerance,
        })
    return sorted(pairs, key=lambda pair: pair["relative_gap"])


def recommend_equivalent_batch(
    rule: dict[str, Any], reference_batch_size: int, reference_prompt_tokens: int,
    target_prompt_tokens: int,
) -> dict[str, Any]:
    """Recommend B at a new L, preferring verified measured pairs to model estimates.

    Args:
        rule: Saved balance_rule.json content with measurements and optional fit.
            balance_rule.json 的内容，提供实测表与可用的拟合规则。
        reference_batch_size: B of an actually measured reference batch.
            实际测过的参考 batch 请求数，用来确定目标时延。
        reference_prompt_tokens: L of that same measured reference batch.
            同一参考 batch 的 token 长度，用来确定目标时延。
        target_prompt_tokens: Desired L whose suitable integer B is to be recommended.
            希望采用的新 token 长度，函数为它推荐整数请求数。
    """
    if target_prompt_tokens < 1:
        raise ValueError("target_prompt_tokens must be positive")
    tolerance = float(rule["tolerance"])
    points = rule["points"]
    reference = next((point for point in points
                      if point["batch_size"] == reference_batch_size
                      and point["prompt_tokens"] == reference_prompt_tokens), None)
    if reference is None:
        raise ValueError("The reference B/L pair must be a successful measured shape")
    target_latency = float(reference["median_ms"])
    measured = sorted((point for point in points if point["prompt_tokens"] == target_prompt_tokens),
                      key=lambda point: relative_latency_gap(target_latency, point["median_ms"]))
    result: dict[str, Any] = {
        "reference": reference, "target_prompt_tokens": target_prompt_tokens,
        "target_latency_ms": target_latency, "tolerance": tolerance,
        "source": "none", "within_tolerance": False,
    }
    if measured:
        nearest = measured[0]
        gap = relative_latency_gap(target_latency, nearest["median_ms"])
        stable = max(reference["cv"], nearest["cv"]) <= tolerance / 2
        result["nearest_measured"] = dict(nearest, relative_gap=gap, stable=stable)
        stable_matches = [point for point in measured
                          if relative_latency_gap(target_latency, point["median_ms"]) <= tolerance
                          and max(reference["cv"], point["cv"]) <= tolerance / 2]
        if stable_matches:
            nearest = stable_matches[0]
            result.update(source="measured", within_tolerance=True,
                          batch_size=nearest["batch_size"], latency_ms=nearest["median_ms"])
            return result
    if not rule.get("accepted"):
        result["reason"] = "No stable measured match at this L; add measurements or relax the tolerance"
        return result
    coefficients = rule["coefficients"]
    denominator = (coefficients["cB"] + coefficients["cBL"] * target_prompt_tokens
                   + coefficients["cBL2"] * target_prompt_tokens**2)
    if denominator <= 0:
        result["reason"] = "The fitted function cannot be inverted at this L"
        return result
    continuous_size = (target_latency - coefficients["c0"]) / denominator
    candidates = []
    for batch_size in sorted({math.floor(continuous_size), math.ceil(continuous_size)}):
        # An existing measurement outranks a prediction, including a measured mismatch.
        # 已有实测值优先于预测值；实测不匹配的形状不能被模型重新推荐为匹配。
        if any(point["batch_size"] == batch_size and point["prompt_tokens"] == target_prompt_tokens
               for point in points):
            continue
        try:
            prediction = predict_latency(batch_size, target_prompt_tokens, rule)
        except ValueError:
            continue
        candidates.append((relative_latency_gap(target_latency, prediction), batch_size, prediction))
    if candidates:
        gap, batch_size, prediction = min(candidates)
        result.update(source="model_estimate", batch_size=batch_size, latency_ms=prediction,
                      continuous_batch_size=continuous_size, predicted_relative_gap=gap,
                      within_tolerance=gap <= tolerance, requires_measurement=True)
    else:
        result["reason"] = "No eligible unmeasured integer B remains; use the observed candidates or extend the grid"
    return result
