"""Post-market evaluation and conservative parameter correction proposals."""

import math


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_prediction(prediction, actual):
    """Compare one snapshot with the next available daily OHLC record."""
    predicted_close = _number(prediction.get("predicted_next_close"))
    previous_close = _number(prediction.get("spot_close"))
    actual_close = _number(actual.get("spot_close"))
    if not all(value is not None for value in (predicted_close, previous_close, actual_close)):
        return None

    actual_return = (actual_close / previous_close - 1.0) * 100
    predicted_direction = prediction.get("predicted_direction", "보합")
    actual_direction = "상승" if actual_return > 0 else "하락" if actual_return < 0 else "보합"
    resistance = _number(prediction.get("energy_dynamic_resistance"))
    support = _number(prediction.get("energy_dynamic_support"))
    actual_high = _number(actual.get("actual_high"))
    actual_low = _number(actual.get("actual_low"))
    return {
        "prediction_date": prediction.get("prediction_date"),
        "actual_date": actual.get("prediction_date"),
        "predicted_close": predicted_close,
        "actual_close": actual_close,
        "predicted_return_pct": _number(prediction.get("predicted_return_pct")) or 0.0,
        "actual_return_pct": actual_return,
        "close_error_pct": (actual_close / predicted_close - 1.0) * 100,
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "direction_hit": predicted_direction == actual_direction,
        "actual_high": actual_high,
        "actual_low": actual_low,
        "resistance_touched": actual_high >= resistance if actual_high is not None and resistance is not None else None,
        "support_broken": actual_low < support if actual_low is not None and support is not None else None,
    }


def correction_proposal(records, window=5):
    """Return a reviewable proposal; never mutates model settings."""
    evaluations = [record.get("evaluation") for record in records]
    recent = [item for item in evaluations if isinstance(item, dict)][-window:]
    if not recent:
        return None

    hit_rate = sum(bool(item.get("direction_hit")) for item in recent) / len(recent)
    errors = [abs(_number(item.get("close_error_pct")) or 0.0) for item in recent]
    proposal = {
        "sample_count": len(recent),
        "direction_hit_rate": hit_rate * 100,
        "mean_abs_close_error_pct": sum(errors) / len(errors),
        "support_break_count": sum(bool(item.get("support_broken")) for item in recent),
        "resistance_touch_count": sum(bool(item.get("resistance_touched")) for item in recent),
        "changes": {},
        "reason": "최근 평가 표본이 부족하거나 현재 설정 유지",
    }
    if len(recent) >= 3 and (hit_rate < 0.4 or proposal["mean_abs_close_error_pct"] >= 2.0):
        proposal["changes"] = {"energy_buffer_multiplier": 1.15, "risk_aversion_delta": 0.5}
        proposal["reason"] = "최근 방향 적중률 또는 종가 오차가 기준을 벗어나 밴드와 리스크를 보수적으로 확대"
    elif len(recent) >= 3 and hit_rate >= 0.8 and proposal["mean_abs_close_error_pct"] < 1.0:
        proposal["changes"] = {"energy_buffer_multiplier": 1.0, "risk_aversion_delta": -0.25}
        proposal["reason"] = "최근 성과가 안정적이어서 과도한 보수화를 일부 완화"
    return proposal