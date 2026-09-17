#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOSPI Market Decision Dashboard
- KOSPI 종합지수 기준 계산 / KOSPI200·선물은 참고용
- 표준편차(1σ~6σ) 밴드, ATR, 변동성
- 테일러 급수 기반 국소 곡률 분석(실험적)
- GJR-GARCH(1,1,1) 조건부 변동성(레버리지 효과 포함)
- 벨만 최적화(동적계획법) 기반 다단계 포지션 계획
- 0/25/50/75/100% 포지션 레벨
- Streamlit 웹 대시보드

실행:
    pip install -r requirements.txt
    pip install arch   # GJR-GARCH 모형에 필요 (requirements.txt에 없다면 별도 설치)
    streamlit run kospi_model.py

데이터:
    yfinance 무료 데이터 사용.
    무료 데이터 제공처(야후 파이낸스) 특성상 한국 선물(KOSPI200 선물) 데이터는
    정상적으로 제공되지 않는 경우가 많다. 이 경우 앱은 자동으로 현물 데이터만으로
    분석하도록 설계되어 있으며, 기본값도 현물 위주(KODEX 200 ETF)로 맞춰져 있다.
    필요하면 사이드바에서 직접 티커를 수정할 수 있다.
"""

import os
import json
import subprocess
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from backtest_evaluator import correction_proposal, evaluate_prediction
from history_logger import read_adjustments, read_records, update_record, upsert_record, write_adjustments
from advanced_gates import (
    LEADER_TICKERS_DEFAULT,
    build_session_report,
    futures_collision_gate,
    intraday_delta_acceleration,
    leader_divergence_check,
)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from pykrx import stock as krx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

st.set_page_config(
    page_title="KOSPI Market Engine",
    page_icon="📈",
    layout="wide",
)
DEFAULTS = {
    "KOSPI 현물": "^KS11",
    "KOSPI200 현물": "069500.KS",  # KODEX 200 ETF. 야후에서 안정적으로 일봉 이력 제공
    "KOSPI200 선물": "",  # 무료 데이터로는 신뢰할 수 있는 한국 선물 데이터가 없어 기본은 빈칸
    "USD/KRW": "KRW=X",
    "VIX": "^VIX",
    "미국장 프록시": "^IXIC",
}

# ---- 신뢰도 보정 임계값 (필요시 조정) ----
VOLUME_RATIO_STRONG = 1.5      # 이 배수 이상 거래량이 터져야 '진짜 이탈' 가능성 높음
VOLATILITY_RATIO_HOT = 1.2     # 단기/장기 변동성 비율이 이 값을 넘으면 '과열' 구간
WEEKLY_CONFLUENCE_PCT = 0.005  # 주봉 MA20과 이 비율(0.5%) 이내로 겹치면 '신뢰 구간'
VIX_RISK_LEVEL = 25.0          # VIX 공포지수 위험 기준
KRW_RISK_5D_PCT = 1.0          # 5일간 원화 약세(환율 상승) 위험 기준(%)

# ---- 벨만 최적화(동적계획법) 설정 ----
POSITION_STATES = [0, 25, 50, 75, 100]  # 이산화된 헤지/인버스 포지션 레벨
BELLMAN_HORIZON = 5                      # 몇 영업일 앞까지 다단계로 계획할지

# 야후파이낸스가 한국 지수(특히 ^KS11)를 특정 시점 이후 갱신하지 않는 현상이
# 실제로 확인되어(2026-09-15), 코스피 종합지수는 KRX, 네이버, 야후 순으로 시도합니다.
KRX_INDEX_CODE_MAP = {"^KS11": "1001", "^KS200": "1028"}
PERIOD_DAYS = {"6mo": 200, "1y": 380, "2y": 760, "5y": 1900}
NAVER_INDEX_CODE_MAP = {"^KS11": "KOSPI", "^KS200": "KOSPI200"}
PREDICTION_LOG_PATH = os.environ.get(
    "KOSPI_PREDICTION_LOG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_log_v2.jsonl"),
)
MODEL_ADJUSTMENT_PATH = os.environ.get(
    "KOSPI_MODEL_ADJUSTMENTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_adjustments_v2.json"),
)


def deployment_revision():
    """실행 중인 코드의 커밋 식별자를 표시한다."""
    configured_revision = os.environ.get("GIT_COMMIT", "").strip()
    if configured_revision:
        return configured_revision[:12]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        return revision or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _data_result(df, source, with_source):
    return (df, source) if with_source else df


def _load_naver_index(ticker, period):
    code = NAVER_INDEX_CODE_MAP[ticker]
    count = PERIOD_DAYS.get(period, 380)
    url = (
        f"https://api.stock.naver.com/chart/domestic/index/{code}"
        f"?periodType=dayCandle&count={count}"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)

    rows = payload.get("priceInfos", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={
        "localDate": "Date",
        "openPrice": "Open",
        "highPrice": "High",
        "lowPrice": "Low",
        "closePrice": "Close",
        "accumulatedTradingVolume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()


def _read_prediction_log(path):
    return read_records(path)


def log_prediction_and_compare(record, path=PREDICTION_LOG_PATH):
    records = _read_prediction_log(path)
    current_date = record["prediction_date"]
    previous_records = [
        item for item in records
        if item.get("prediction_date", "") < current_date
    ]
    comparison = None

    if previous_records:
        previous = max(previous_records, key=lambda item: item["prediction_date"])
        previous_close = float(previous["spot_close"])
        actual_close = float(record["spot_close"])
        actual_return_pct = (actual_close / previous_close - 1.0) * 100
        predicted_return_pct = float(previous["predicted_return_pct"])
        actual_direction = (
            "상승" if actual_return_pct > 0 else
            "하락" if actual_return_pct < 0 else "보합"
        )
        predicted_direction = previous.get("predicted_direction", "보합")
        comparison = evaluate_prediction(previous, record)
        if comparison is not None:
            update_record(path, previous["prediction_date"], {"evaluation": comparison})

    # Streamlit rerun이 같은 거래일에 여러 번 발생해도 해당 날짜의 기록은 하나만 유지한다.
    records = [
        item for item in records
        if item.get("prediction_date") != current_date
    ]
    records.append(record)
    records.sort(key=lambda item: item.get("prediction_date", ""))

    upsert_record(path, record)

    return comparison


def prediction_history_rows(records):
    """예측 로그를 신뢰도 추이와 사후 방향 적중률 표로 변환한다."""
    rows = []
    ordered = sorted(records, key=lambda item: item.get("prediction_date", ""))
    for index, record in enumerate(ordered):
        row = {
            "예측일": record.get("prediction_date", "-"),
            "방향 점수": record.get("direction_score", np.nan),
            "신뢰도(%)": record.get("confidence", np.nan),
            "국면": record.get("regime", "-") ,
            "예측 방향": record.get("predicted_direction", "보합"),
            "헤지 단계(%)": record.get("current_position", np.nan),
            "실제 방향": "미확인",
            "방향 적중": "미확인",
        }
        if index + 1 < len(ordered):
            next_record = ordered[index + 1]
            try:
                actual_return_pct = (
                    float(next_record["spot_close"]) / float(record["spot_close"]) - 1.0
                ) * 100
                actual_direction = (
                    "상승" if actual_return_pct > 0 else
                    "하락" if actual_return_pct < 0 else "보합"
                )
                row["실제 방향"] = actual_direction
                row["방향 적중"] = (
                    "적중" if record.get("predicted_direction", "보합") == actual_direction
                    else "불일치"
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass
        rows.append(row)
    return rows


@st.cache_data(ttl=300)
def load_data(
    ticker: str, period: str = "1y", interval: str = "1d", with_source: bool = False
):
    if not ticker:
        return _data_result(pd.DataFrame(), None, with_source)

    # 코스피/코스피200은 KRX 원천 우선 시도 (야후 지연 문제 우회)
    if PYKRX_AVAILABLE and ticker in KRX_INDEX_CODE_MAP and interval == "1d":
        try:
            code = KRX_INDEX_CODE_MAP[ticker]
            end = datetime.now()
            start = end - timedelta(days=PERIOD_DAYS.get(period, 380))
            krx_df = krx_stock.get_index_ohlcv_by_date(
                start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
            )
            if not krx_df.empty:
                krx_df = krx_df.rename(columns={
                    "시가": "Open", "고가": "High", "저가": "Low",
                    "종가": "Close", "거래량": "Volume",
                })
                return _data_result(
                    krx_df[["Open", "High", "Low", "Close", "Volume"]].dropna(),
                    "KRX (pykrx)",
                    with_source,
                )
        except Exception:
            pass  # 실패 시 아래 야후파이낸스로 폴백

    if ticker in NAVER_INDEX_CODE_MAP and interval == "1d":
        try:
            naver_df = _load_naver_index(ticker, period)
            if not naver_df.empty:
                return _data_result(naver_df, "Naver Finance", with_source)
        except Exception:
            pass  # 실패 시 아래 야후파이낸스로 폴백

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return _data_result(pd.DataFrame(), None, with_source)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return _data_result(pd.DataFrame(), None, with_source)

        df = df[required + ([c for c in ["Volume"] if c in df.columns])]
        return _data_result(df.dropna(), "Yahoo Finance", with_source)
    except Exception:
        return _data_result(pd.DataFrame(), None, with_source)


def pct_change(df, n=1):
    return df["Close"].pct_change(n) * 100


def load_live_quote(ticker):
    """fast_info로 근사 실시간가와 전일종가 대비 변동률(%)을 가져온다. 실패 시 (nan, nan)."""
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        last = float(fast["last_price"])
        prev_close = float(fast["previous_close"])
        pct = (last - prev_close) / prev_close * 100
        return last, pct
    except Exception:
        return np.nan, np.nan


def atr(df, n=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def zscore(series, n=60):
    mean = series.rolling(n).mean()
    std = series.rolling(n).std()
    return (series - mean) / std.replace(0, np.nan)


def make_levels(df):
    close = float(df["Close"].iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
    a = float(atr(df).iloc[-1])

    # 변동성에 따른 동적 밴드
    resistance = max(ma20, close) + 1.0 * a
    support = min(ma20, close) - 1.0 * a

    # 최근 60일 고저점
    high60 = float(df["High"].rolling(60).max().iloc[-1])
    low60 = float(df["Low"].rolling(60).min().iloc[-1])

    return {
        "close": close,
        "ma20": ma20,
        "ma60": ma60,
        "atr": a,
        "resistance": resistance,
        "support": support,
        "high60": high60,
        "low60": low60,
    }


def std_bands(df, ma_n=20, std_n=20, max_k=6):
    """
    이동평균(ma_n일) ± k * 표준편차(std_n일) 밴드를 k=1..max_k 까지 계산.
    반환: (기준 이동평균, 표준편차, {k: {"support":..., "resistance":...}})
    """
    close = df["Close"]
    ma = float(close.rolling(ma_n).mean().iloc[-1])
    std = float(close.rolling(std_n).std().iloc[-1])

    bands = {}
    for k in range(1, max_k + 1):
        bands[k] = {
            "support": ma - k * std,
            "resistance": ma + k * std,
        }
    return ma, std, bands


def volume_ratio(df, n=20):
    """당일 거래량 / 최근 n일 평균 거래량. Volume 데이터가 없으면 NaN."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return np.nan
    avg_vol = df["Volume"].rolling(n).mean().iloc[-1]
    today_vol = df["Volume"].iloc[-1]
    if not avg_vol or np.isnan(avg_vol) or avg_vol == 0:
        return np.nan
    return float(today_vol / avg_vol)


def volatility_ratio(df, short_n=5, long_n=60):
    """단기(short_n일) 표준편차 / 장기(long_n일) 표준편차. 1보다 크면 변동성 확대 국면."""
    close = df["Close"]
    std_short = close.rolling(short_n).std().iloc[-1]
    std_long = close.rolling(long_n).std().iloc[-1]
    if not std_long or np.isnan(std_long) or std_long == 0:
        return np.nan
    return float(std_short / std_long)


def _percentile_rank(series, value):
    """현재 값이 관측 표본에서 차지하는 백분위. 표본이 없으면 NaN."""
    values = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or pd.isna(value) or not np.isfinite(value):
        return np.nan
    return float((values <= value).mean() * 100)


def price_volume_momentum(df, window=120):
    """일간 수익률×거래량을 계산하고 최근 표본 대비 에너지 백분위를 반환한다."""
    if (
        "Volume" not in df.columns
        or df["Volume"].isna().all()
        or not (df["Volume"] > 0).any()
        or len(df) < 2
    ):
        return None

    returns = df["Close"].pct_change()
    raw_energy = returns * df["Volume"]
    history = raw_energy.abs().tail(window).dropna()
    current = float(raw_energy.iloc[-1])
    if not np.isfinite(current):
        return None
    return {
        "raw": current,
        "absolute": abs(current),
        "percentile": _percentile_rank(history, abs(current)),
        "return_pct": float(returns.iloc[-1] * 100),
        "volume_ratio": volume_ratio(df),
        "window": window,
    }


def obv_analysis(df, window=20):
    """OBV와 가격-OBV 방향 불일치(다이버전스)를 계산한다."""
    if "Volume" not in df.columns or df["Volume"].isna().all() or len(df) <= window:
        return None

    close = df["Close"]
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * df["Volume"].fillna(0)).cumsum()
    price_change = float(close.pct_change(window).iloc[-1] * 100)
    obv_change = float(obv.iloc[-1] - obv.iloc[-window - 1])
    average_volume = float(df["Volume"].tail(window).mean())
    obv_change_units = obv_change / average_volume if average_volume > 0 else np.nan

    bullish = price_change < 0 and obv_change > 0
    bearish = price_change > 0 and obv_change < 0
    signal = "강세 다이버전스" if bullish else "약세 다이버전스" if bearish else "확인 없음"
    return {
        "obv": float(obv.iloc[-1]),
        "price_change_pct": price_change,
        "obv_change": obv_change,
        "obv_change_units": obv_change_units,
        "bullish_divergence": bullish,
        "bearish_divergence": bearish,
        "signal": signal,
        "window": window,
    }


def volatility_volume_energy(df, atr_window=14, volume_window=20, percentile_window=120):
    """ATR을 가격으로 정규화한 뒤 거래량 배수를 곱한 변동성-거래량 에너지."""
    if (
        len(df) < max(atr_window, volume_window) + 1
        or "Volume" not in df.columns
        or df["Volume"].isna().all()
        or not (df["Volume"] > 0).any()
    ):
        return None

    atr_series = atr(df, n=atr_window)
    close = df["Close"]
    normalized_atr = atr_series / close
    volume_series = (
        df["Volume"] / df["Volume"].rolling(volume_window).mean()
        if "Volume" in df.columns and not df["Volume"].isna().all()
        else pd.Series(1.0, index=df.index)
    )
    energy_series = normalized_atr * volume_series
    current = float(energy_series.iloc[-1])
    if not np.isfinite(current):
        return None
    return {
        "value": current,
        "percentile": _percentile_rank(energy_series.tail(percentile_window), current),
        "atr_pct": float(normalized_atr.iloc[-1] * 100),
        "volume_ratio": float(volume_series.iloc[-1]),
        "window": percentile_window,
    }


def energy_levels(df, volume_window=20, momentum_window=5, buffer_multiplier=1.0):
    """ATR·거래량·5일 모멘텀으로 현재 에너지 상단/하단을 계산한다."""
    if len(df) < max(volume_window, momentum_window, 14) + 1:
        return None

    current = float(df["Close"].iloc[-1])
    atr_value = float(atr(df, n=14).iloc[-1])
    momentum_return = float(df["Close"].pct_change(momentum_window).iloc[-1])
    if np.isnan(atr_value) or np.isnan(momentum_return):
        return None

    volume_available = "Volume" in df.columns and not df["Volume"].isna().all()
    if volume_available:
        average_volume = df["Volume"].rolling(volume_window).mean().iloc[-1]
        actual_volume = df["Volume"].iloc[-1]
        volume_ratio_value = (
            float(actual_volume / average_volume)
            if average_volume and not np.isnan(average_volume) else 1.0
        )
    else:
        volume_ratio_value = 1.0

    volume_ratio_value = max(volume_ratio_value, 0.0)
    momentum_factor = float(np.clip(1.0 - abs(momentum_return), 0.0, 1.0))
    energy_width = atr_value * volume_ratio_value * momentum_factor * buffer_multiplier
    return {
        "current": current,
        "atr14": atr_value,
        "momentum_return_5d": momentum_return,
        "momentum_factor": momentum_factor,
        "volume_ratio": volume_ratio_value,
        "volume_available": volume_available,
        "energy_width": energy_width,
        "dynamic_resistance": current + energy_width,
        "dynamic_support": current - energy_width,
    }


def energy_level_backtest(df, horizon=5, volume_window=20, momentum_window=5, buffer_multiplier=1.0):
    """과거 에너지 상·하단의 터치와 이후 방어/거부 여부를 검증한다."""
    required = {"High", "Low", "Close"}
    minimum = max(14, volume_window, momentum_window) + horizon + 1
    if not required.issubset(df.columns) or len(df) < minimum:
        return {"resistance": None, "support": None, "sample_count": 0}

    events = []
    for index in range(max(14, volume_window, momentum_window), len(df) - horizon):
        window = df.iloc[:index + 1]
        levels = energy_levels(window, volume_window, momentum_window, buffer_multiplier)
        if levels is None or levels["energy_width"] <= 0:
            continue
        future = df.iloc[index + 1:index + horizon + 1]
        resistance_touch = bool((future["High"] >= levels["dynamic_resistance"]).any())
        support_touch = bool((future["Low"] <= levels["dynamic_support"]).any())
        events.append({
            "resistance_touch": resistance_touch,
            "resistance_hold": resistance_touch and float(future["Close"].iloc[-1]) < levels["dynamic_resistance"],
            "support_touch": support_touch,
            "support_hold": support_touch and float(future["Close"].iloc[-1]) > levels["dynamic_support"],
        })

    def summarize(touch_key, hold_key):
        touched = [event for event in events if event[touch_key]]
        if not touched:
            return {"touch_count": 0, "hold_count": 0, "hold_probability": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        hold_count = sum(event[hold_key] for event in touched)
        lower, upper = _wilson_interval(hold_count, len(touched))
        return {
            "touch_count": len(touched),
            "hold_count": hold_count,
            "hold_probability": hold_count / len(touched) * 100,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return {
        "resistance": summarize("resistance_touch", "resistance_hold"),
        "support": summarize("support_touch", "support_hold"),
        "sample_count": len(events),
        "horizon": horizon,
    }


def overnight_gap_stats(df, threshold_pct=1.0):
    """종가 대비 다음 거래일 시가 갭의 실제 과거 분포를 계산한다."""
    required = {"Open", "Close"}
    if not required.issubset(df.columns) or len(df) < 2:
        return None

    gaps = (df["Open"].shift(-1) / df["Close"] - 1.0) * 100
    gaps = gaps.replace([np.inf, -np.inf], np.nan).dropna()
    if gaps.empty:
        return None

    absolute_gaps = gaps.abs()
    return {
        "count": int(gaps.size),
        "mean": float(gaps.mean()),
        "median": float(gaps.median()),
        "std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
        "p10": float(gaps.quantile(0.10)),
        "p90": float(gaps.quantile(0.90)),
        "mean_abs": float(absolute_gaps.mean()),
        "p90_abs": float(absolute_gaps.quantile(0.90)),
        "up_probability": float((gaps > 0).mean() * 100),
        "down_probability": float((gaps < 0).mean() * 100),
        "large_gap_probability": float((absolute_gaps >= threshold_pct).mean() * 100),
        "latest_gap": float(gaps.iloc[-1]) if len(gaps) else np.nan,
    }


def conditional_overnight_gap_stats(kospi_df, global_df, threshold_pct=1.0):
    """직전 미국장 수익률 조건별로 다음 KOSPI 시가 갭을 비교한다."""
    required = {"Open", "Close"}
    if not required.issubset(kospi_df.columns) or "Close" not in global_df.columns:
        return None
    if len(kospi_df) < 2 or len(global_df) < 2:
        return None

    kospi_index = pd.DatetimeIndex(pd.to_datetime(kospi_df.index)).tz_localize(None)
    global_index = pd.DatetimeIndex(pd.to_datetime(global_df.index)).tz_localize(None)
    kospi = kospi_df.copy()
    global_returns = global_df["Close"].pct_change() * 100
    global_returns.index = global_index
    kospi.index = kospi_index

    gap_frame = pd.DataFrame({
        "event_date": kospi.index[:-1],
        "next_date": kospi.index[1:],
        "gap": (kospi["Open"].iloc[1:].to_numpy() / kospi["Close"].iloc[:-1].to_numpy() - 1.0) * 100,
    })
    gap_frame["us_lookup_date"] = gap_frame["next_date"] - pd.Timedelta(days=1)
    us_frame = global_returns.rename("us_return").reset_index()
    us_frame.columns = ["us_date", "us_return"]
    us_frame = us_frame.dropna().sort_values("us_date")
    gap_frame = pd.merge_asof(
        gap_frame.sort_values("us_lookup_date"),
        us_frame,
        left_on="us_lookup_date",
        right_on="us_date",
        direction="backward",
        tolerance=pd.Timedelta(days=4),
    ).dropna(subset=["us_return", "gap"])
    if gap_frame.empty:
        return None

    gap_frame["condition"] = np.select(
        [gap_frame["us_return"] >= 0.5, gap_frame["us_return"] <= -0.5],
        ["미국장 상승", "미국장 하락"],
        default="미국장 혼조",
    )
    rows = []
    for condition, group in gap_frame.groupby("condition", sort=False):
        absolute_gaps = group["gap"].abs()
        lower, upper = _wilson_interval(int((group["gap"] > 0).sum()), len(group))
        rows.append({
            "condition": condition,
            "count": int(len(group)),
            "up_probability": float((group["gap"] > 0).mean() * 100),
            "down_probability": float((group["gap"] < 0).mean() * 100),
            "large_gap_probability": float((absolute_gaps >= threshold_pct).mean() * 100),
            "mean_abs": float(absolute_gaps.mean()),
            "ci_lower": lower,
            "ci_upper": upper,
        })

    latest_us_return = float(global_returns.iloc[-1]) if not global_returns.empty else np.nan
    return {
        "rows": rows,
        "sample_count": int(len(gap_frame)),
        "latest_us_return": latest_us_return,
        "latest_condition": (
            "미국장 상승" if latest_us_return >= 0.5 else
            "미국장 하락" if latest_us_return <= -0.5 else "미국장 혼조"
        ) if not np.isnan(latest_us_return) else "데이터 없음",
    }


def overnight_action_plan(level, overnight_stats, conditional_stats, rebound_analysis, volume_ratio_value):
    """갭 노출·반등 형태·거래량을 결합해 다음 거래일 실행안을 만든다."""
    if overnight_stats is None:
        return {"level": level, "action": "데이터 부족", "confidence": 0, "reasons": []}

    reasons = []
    risk_score = 0
    if overnight_stats["large_gap_probability"] >= 50:
        risk_score += 2
        reasons.append(f"±1% 이상 갭 과거 빈도 {overnight_stats['large_gap_probability']:.1f}%")
    elif overnight_stats["large_gap_probability"] >= 30:
        risk_score += 1

    if not np.isnan(volume_ratio_value) and volume_ratio_value < 0.8:
        risk_score += 1
        reasons.append(f"거래량 {volume_ratio_value:.2f}배로 반등 확인 부족")

    if rebound_analysis.get("current"):
        current = rebound_analysis["current"]
        if current["is_rebound"] and not current["support_touch"]:
            risk_score += 1
            reasons.append("반등은 발생했지만 주요 지지선 확인이 약함")

    current_condition = conditional_stats.get("latest_condition") if conditional_stats else None
    conditional_row = next(
        (row for row in (conditional_stats or {}).get("rows", []) if row["condition"] == current_condition),
        None,
    )
    if conditional_row and conditional_row["large_gap_probability"] >= 50:
        risk_score += 1
        reasons.append(f"현재 미국장 조건의 큰 갭 빈도 {conditional_row['large_gap_probability']:.1f}%")

    if risk_score >= 3:
        action = "장 마감 전 50% 축소 · 내일 시가와 첫 15분 수급 확인 후 재진입"
        recommended_level = round(level * 0.5 / 25) * 25
    elif risk_score >= 1:
        action = "오버나이트 75%만 유지 · 갭 발생 시 추가 진입은 보류"
        recommended_level = round(level * 0.75 / 25) * 25
    else:
        action = "현재 헤지 수준 유지 · 단, 시가 갭 손실 한도 사전 설정"
        recommended_level = level

    return {
        "level": int(recommended_level),
        "action": action,
        "confidence": int(max(0, min(100, 100 - risk_score * 20))),
        "risk_score": risk_score,
        "reasons": reasons,
        "conditional_row": conditional_row,
    }


def overnight_execution_timeline(current_level, overnight_plan, overnight_stats):
    """갭 위험에 따른 장 마감 전·개장 직후·수급 확인 타임라인을 만든다."""
    if overnight_plan is None or overnight_stats is None:
        return None

    recommended_level = int(overnight_plan["level"])
    gap_risk = (
        "높음" if overnight_stats["large_gap_probability"] >= 50
        else "중간" if overnight_stats["large_gap_probability"] >= 30
        else "낮음"
    )
    if recommended_level < current_level:
        close_action = f"헤지/인버스 {current_level}% → {recommended_level}%로 축소"
    else:
        close_action = f"헤지/인버스 {current_level}% 유지"

    return {
        "gap_risk": gap_risk,
        "recommended_level": recommended_level,
        "stages": [
            {
                "시간": "오늘 15:20~15:30",
                "행동": close_action,
                "확인": f"±1% 이상 갭 과거 빈도 {overnight_stats['large_gap_probability']:.1f}%",
            },
            {
                "시간": "내일 09:00~09:15",
                "행동": "자동 주문 금지 · 시가와 첫 15분 흐름 관찰",
                "확인": "개장 갭 방향과 첫 15분 매수·매도 수급",
            },
            {
                "시간": "내일 09:15 이후",
                "행동": "남은 물량 재조정",
                "확인": "예측과 반대 방향이면 사전 손절 기준 준수, 일치하면 단계적 대응",
            },
        ],
    }


def _cluster_pivot_levels(levels, tolerance):
    """가격 허용오차 안에 있는 스윙 포인트를 하나의 반응 클러스터로 묶는다."""
    if not levels:
        return []

    clusters = []
    for level, date in sorted(levels, key=lambda item: item[0]):
        matching = [
            cluster for cluster in clusters
            if abs(level - cluster["level"]) <= tolerance
        ]
        if matching:
            cluster = min(matching, key=lambda item: abs(level - item["level"]))
            cluster["prices"].append(level)
            cluster["dates"].append(date)
            cluster["level"] = float(np.mean(cluster["prices"]))
        else:
            clusters.append({"level": float(level), "prices": [level], "dates": [date]})

    return [
        {
            "level": cluster["level"],
            "reactions": len(cluster["prices"]),
            "last_reaction": max(cluster["dates"]),
        }
        for cluster in clusters
    ]


def swing_level_clusters(df, lookback=5, min_reactions=2):
    """스윙 고점·저점을 ATR 기반 허용오차로 묶어 지지·저항 클러스터를 찾는다."""
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns) or len(df) < lookback * 2 + 1:
        return {"support": [], "resistance": [], "tolerance": np.nan}

    atr_value = float(atr(df).iloc[-1])
    close = float(df["Close"].iloc[-1])
    if np.isnan(atr_value) or atr_value <= 0:
        atr_value = close * 0.005
    tolerance = max(atr_value * 0.5, close * 0.0025)

    highs = []
    lows = []
    for index in range(lookback, len(df) - lookback):
        high_window = df["High"].iloc[index - lookback:index + lookback + 1]
        low_window = df["Low"].iloc[index - lookback:index + lookback + 1]
        date = df.index[index].date().isoformat()
        if df["High"].iloc[index] == high_window.max():
            highs.append((float(df["High"].iloc[index]), date))
        if df["Low"].iloc[index] == low_window.min():
            lows.append((float(df["Low"].iloc[index]), date))

    support = [
        cluster for cluster in _cluster_pivot_levels(lows, tolerance)
        if cluster["level"] <= close and cluster["reactions"] >= min_reactions
    ]
    resistance = [
        cluster for cluster in _cluster_pivot_levels(highs, tolerance)
        if cluster["level"] >= close and cluster["reactions"] >= min_reactions
    ]
    support.sort(key=lambda item: item["level"], reverse=True)
    resistance.sort(key=lambda item: item["level"])

    return {
        "support": support,
        "resistance": resistance,
        "tolerance": float(tolerance),
        "lookback": lookback,
        "close": close,
    }


def rebound_exit_guide(spot, swing_levels, energy_level, resistance_level, volume_ratio_value, direction_score):
    """반등 시 저항 목표와 30/40/30 분할 청산 기준을 계산한다."""
    resistance_clusters = swing_levels.get("resistance", [])
    nearest_cluster = next(
        (item for item in resistance_clusters if item["level"] > spot),
        None,
    )
    candidates = []
    if nearest_cluster is not None:
        candidates.append((float(nearest_cluster["level"]), "스윙 저항 클러스터"))
    if energy_level is not None and energy_level["dynamic_resistance"] > spot:
        candidates.append((float(energy_level["dynamic_resistance"]), "동적 저항선"))
    if not candidates:
        return None

    first_target, target_source = min(candidates, key=lambda item: item[0])
    distance_pct = (first_target / spot - 1.0) * 100
    tolerance = swing_levels.get("tolerance", 0.0)
    target_zone_low = max(spot, first_target - tolerance)
    target_zone_high = first_target + tolerance
    breakout_confirmed = (
        first_target > spot
        and not np.isnan(volume_ratio_value)
        and volume_ratio_value >= VOLUME_RATIO_STRONG
    )
    trend_holds = direction_score < -45
    return {
        "first_target": first_target,
        "target_source": target_source,
        "distance_pct": distance_pct,
        "target_zone_low": target_zone_low,
        "target_zone_high": target_zone_high,
        "tolerance": tolerance,
        "breakout_confirmed": breakout_confirmed,
        "trend_holds": trend_holds,
        "stage_1_pct": 30,
        "stage_2_pct": 40,
        "stage_3_pct": 30,
    }


def _wilson_interval(successes, total, z=1.96):
    """이항 비율의 표본 크기 보정 95% Wilson 신뢰구간을 반환한다."""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = (
        z / denominator
        * np.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2))
    )
    return (center - margin) * 100, (center + margin) * 100


def rebound_scenario_analysis(df, support_levels=None, horizon=5):
    """과거 반등 사례를 이용해 기술적 반등과 추세 전환 시나리오를 검증한다."""
    result = {
        "current": None,
        "scenarios": [],
        "sample_count": 0,
        "sample_label": "표본 부족",
        "dominant": None,
    }
    if not {"High", "Low", "Close"}.issubset(df.columns) or len(df) < 80 + horizon:
        return result

    close = df["Close"]
    atr_series = atr(df)
    ma20 = close.rolling(20).mean()
    rolling_low = df["Low"].rolling(60).min()
    returns_1d = close.pct_change() * 100
    returns_5d = close.pct_change(5) * 100

    historical_events = []
    for index in range(60, len(df) - horizon):
        event_close = float(close.iloc[index])
        event_atr = float(atr_series.iloc[index])
        if np.isnan(event_atr) or event_atr <= 0:
            continue
        is_rebound = returns_1d.iloc[index] >= 1.0 and returns_5d.iloc[index] <= -2.0
        support_touch = event_close <= rolling_low.iloc[index] + 1.5 * event_atr
        if not is_rebound:
            continue

        future_return = (close.iloc[index + horizon] / event_close - 1.0) * 100
        if future_return >= 2.0:
            label = "국면 전환"
        elif future_return <= -1.0:
            label = "기술적 반등 실패"
        else:
            label = "추세 진행 중"
        historical_events.append({"label": label, "support_touch": support_touch})

    support_events = [event for event in historical_events if event["support_touch"]]
    events = support_events if len(support_events) >= 10 else historical_events
    if not events:
        return result

    labels = ["기술적 반등 실패", "국면 전환", "추세 진행 중"]
    scenario_rows = []
    for label in labels:
        successes = sum(event["label"] == label for event in events)
        lower, upper = _wilson_interval(successes, len(events))
        scenario_rows.append({
            "scenario": label,
            "count": successes,
            "probability": successes / len(events) * 100,
            "lower": lower,
            "upper": upper,
        })
    scenario_rows.sort(key=lambda row: row["probability"], reverse=True)

    current_close = float(close.iloc[-1])
    current_atr = float(atr_series.iloc[-1])
    current_support = np.nan
    if support_levels:
        current_support = float(support_levels[0]["level"])
    elif not np.isnan(current_atr):
        current_support = float(rolling_low.iloc[-1])
    support_distance = (
        abs(current_close - current_support) / current_atr
        if not np.isnan(current_support) and current_atr > 0 else np.nan
    )
    current = {
        "today_return": float(returns_1d.iloc[-1]),
        "return_5d": float(returns_5d.iloc[-1]),
        "support": current_support,
        "support_distance_atr": support_distance,
        "is_rebound": bool(returns_1d.iloc[-1] >= 1.0 and returns_5d.iloc[-1] <= -2.0),
        "support_touch": bool(not np.isnan(support_distance) and support_distance <= 1.5),
        "ma20": float(ma20.iloc[-1]),
    }
    current["setup_score"] = round(
        100 * (
            (1 / 3 if current["is_rebound"] else 0)
            + (1 / 3 if current["support_touch"] else 0)
            + (1 / 3 if current_close > current["ma20"] else 0)
        )
    )

    result.update({
        "current": current,
        "scenarios": scenario_rows,
        "sample_count": len(events),
        "sample_label": "지지선 접촉 사례" if events is support_events else "전체 반등 사례",
        "dominant": scenario_rows[0],
    })
    return result


def weekly_ma20(df, ma_n=20):
    """일봉 데이터를 주봉으로 리샘플링한 뒤 ma_n주 이동평균의 마지막 값."""
    weekly_close = df["Close"].resample("W").last().dropna()
    if len(weekly_close) < ma_n:
        return np.nan
    return float(weekly_close.rolling(ma_n).mean().iloc[-1])


def band_reliability_tags(k, level_value, vol_ratio_val, vola_ratio_val, weekly_ma_val):
    """
    특정 σ 밴드 레벨(level_value, k차수)에 대한 신뢰도 태그 목록을 반환.
    - 변동성 과열 국면에서는 1~2σ를 낮은 신뢰도로 표시
    - 거래량이 평소 대비 충분히 터지지 않으면 낮은 신뢰도로 표시
    - 주봉 MA20과 겹치면 '신뢰 구간'으로 가점 표시
    """
    tags = []

    if not np.isnan(vola_ratio_val) and vola_ratio_val > VOLATILITY_RATIO_HOT and k <= 2:
        tags.append("변동성 과열 · 낮은 신뢰")

    if not np.isnan(vol_ratio_val) and vol_ratio_val < VOLUME_RATIO_STRONG:
        tags.append("거래량 부족 · 가짜 이탈 주의")

    if (
        not np.isnan(weekly_ma_val)
        and weekly_ma_val != 0
        and abs(level_value - weekly_ma_val) / weekly_ma_val <= WEEKLY_CONFLUENCE_PCT
    ):
        tags.append("★ 주봉 MA20 겹침 (신뢰 구간)")

    if not tags:
        tags.append("보통")

    return " / ".join(tags)


def macro_gate(krw_df, vix_df):
    """
    원/달러 환율(5일 변화율)과 VIX 수준을 이용한 매크로 위험 게이트.
    반환: (krw_5d_pct, vix_last, is_risk_on, multiplier)
    위험(원화 약세 + VIX 급등) 동시 충족 시 포지션 진입 강도를 절반으로 낮춘다.
    """
    krw_5d_pct = np.nan
    vix_last = np.nan

    if not krw_df.empty and len(krw_df) > 5:
        krw_5d_pct = float(krw_df["Close"].pct_change(5).iloc[-1] * 100)

    if not vix_df.empty:
        vix_last = float(vix_df["Close"].iloc[-1])

    is_risk = (
        not np.isnan(krw_5d_pct)
        and not np.isnan(vix_last)
        and krw_5d_pct > KRW_RISK_5D_PCT
        and vix_last > VIX_RISK_LEVEL
    )
    multiplier = 0.5 if is_risk else 1.0
    return krw_5d_pct, vix_last, is_risk, multiplier


def taylor_fit(df, window=20, degree=3):
    """
    최근 window일 종가에 degree차 다항식을 피팅하고, 마지막 시점(a=오늘)에서의
    함수값/1차미분(모멘텀)/2차미분(곡률)/3차미분(곡률 변화율)을 계산.
    ※ 실제 주가는 매끄러운 함수가 아니므로 이는 엄밀한 테일러 급수가 아니라
    '국소 다항식 피팅 기반 근사'다. 예측이라기보다 현재 추세의 휘어짐을
    수치화하는 보조 지표로 사용한다.
    """
    close = df["Close"].tail(window).values
    if len(close) < window:
        return None

    x = np.arange(window, dtype=float)
    coeffs = np.polyfit(x, close, degree)
    poly = np.poly1d(coeffs)
    d1 = poly.deriv(1)
    d2 = poly.deriv(2)
    d3 = poly.deriv(3) if degree >= 3 else np.poly1d([0.0])

    a = window - 1  # 기준점 a = 오늘(윈도우 마지막 날)
    fitted = poly(x)
    resid = close - fitted
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    return {
        "f_a": float(poly(a)),
        "f1_a": float(d1(a)),
        "f2_a": float(d2(a)),
        "f3_a": float(d3(a)),
        "rmse": rmse,
        "window": window,
        "degree": degree,
    }


def taylor_projection(fit, horizons=(1, 2, 3, 4, 5)):
    """
    테일러 전개식 f(a+h) ≈ f(a) + f'(a)h + f''(a)/2 h² + f'''(a)/6 h³ 로
    h영업일 뒤 경로를 근사하고, 피팅 잔차(RMSE)를 기준점에서 멀어질수록
    커지는 불확실성 구간(오차 밴드)으로 함께 제시한다.
    """
    f_a, f1, f2, f3, rmse = (
        fit["f_a"], fit["f1_a"], fit["f2_a"], fit["f3_a"], fit["rmse"]
    )
    rows = []
    for h in horizons:
        proj = f_a + f1 * h + (f2 / 2) * h ** 2 + (f3 / 6) * h ** 3
        band = rmse * np.sqrt(1 + h)  # 기준점에서 멀수록 오차 확대(단순 근사)
        rows.append({"h": h, "proj": proj, "upper": proj + band, "lower": proj - band})
    return rows


def turning_point_signal(df, short_window=10, long_window=20, degree=2):
    """
    단기(short_window)와 장기(long_window) 윈도우로 각각 다항식을 피팅해
    2차미분(곡률) 부호를 비교. 부호가 서로 다르면 '변곡점이 임박했을 가능성'으로 본다.
    """
    fit_short = taylor_fit(df, window=short_window, degree=degree)
    fit_long = taylor_fit(df, window=long_window, degree=degree)
    if fit_short is None or fit_long is None:
        return None

    short_curv = fit_short["f2_a"]
    long_curv = fit_long["f2_a"]
    sign_flip = (
        abs(short_curv) > 1e-9
        and abs(long_curv) > 1e-9
        and (short_curv > 0) != (long_curv > 0)
    )
    return {"short_curv": short_curv, "long_curv": long_curv, "sign_flip": sign_flip}


def fit_gjr_garch(kospi_df, min_obs=100):
    """
    로그수익률(%) 기준 GJR-GARCH(1,1,1) 모형 적합.
    arch 패키지의 vol='GARCH', o=1 옵션이 곧 GJR-GARCH(비대칭 지시함수 포함) 사양이다.
    반환: (모형결과 또는 None, 오늘의 조건부 일간변동성(%), 향후 BELLMAN_HORIZON일 변동성 예측 경로(%, ndarray))
    """
    if not ARCH_AVAILABLE:
        return None, np.nan, None

    close = kospi_df["Close"]
    rets = 100 * np.log(close / close.shift(1)).dropna()
    if len(rets) < min_obs:
        return None, np.nan, None

    try:
        am = arch_model(rets, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off")
    except Exception:
        return None, np.nan, None

    today_vol = float(res.conditional_volatility.iloc[-1])  # 일간 변동성(%)

    try:
        fc = res.forecast(horizon=BELLMAN_HORIZON, reindex=False)
        var_path = fc.variance.values[-1]
        vol_path = np.sqrt(var_path)  # 일간 변동성(%) 경로
    except Exception:
        vol_path = np.full(BELLMAN_HORIZON, today_vol)

    return res, today_vol, vol_path


def ewma_volatility_path(kospi_df, horizon=BELLMAN_HORIZON, decay=0.94):
    """arch 대체용 EWMA 일간 변동성(%)과 향후 경로를 반환한다."""
    returns = 100 * np.log(kospi_df["Close"] / kospi_df["Close"].shift(1)).dropna()
    if len(returns) < 20:
        return np.nan, None

    variance = float(returns.iloc[:20].var())
    for value in returns.iloc[20:]:
        variance = decay * variance + (1.0 - decay) * float(value) ** 2
    today_vol = float(np.sqrt(max(variance, 0.0)))
    return today_vol, np.full(horizon, today_vol, dtype=float)


def extended_garch_variance(garch_today_vol_pct, vol_ratio_val, basis_z_val, delta1=0.3, delta2=0.2):
    """
    확장형 조건부분산 h_t_ext = h_t * (1 + δ1·X1 + δ2·X2) 근사.
    원 논문식 h_t = ω + αε² + γε²I(ε<0) + βh_{t-1} + Σδ_j X_{t-j} 를
    MLE로 동시추정하려면 arch 패키지의 우도함수를 직접 뜯어고쳐야 하므로,
    실무적으로는 이미 적합된 GJR-GARCH 분산(h_t)에 외생변수 보정을
    사후적으로(post-hoc) 곱연산으로 얹는 근사 방식을 쓴다.

    X1(거래량 소진): 거래량비율이 1보다 작을수록(평균 이하 거래) 소진 신호로 간주해 0~1로 스케일.
    X2(프리미엄/괴리 극단도): 현·선물 Basis Z-score의 절댓값 (선물 데이터 없으면 0으로 처리됨).
    반환: (h_t 원본, h_t_ext 확장분산) — 둘 다 %^2 단위.
    """
    if np.isnan(garch_today_vol_pct):
        return np.nan, np.nan

    h_t = garch_today_vol_pct ** 2

    x1 = min(max(0.0, 1.0 - vol_ratio_val), 1.0) if not np.isnan(vol_ratio_val) else 0.0
    x2 = abs(basis_z_val) if not np.isnan(basis_z_val) else 0.0

    h_t_ext = h_t * (1.0 + delta1 * x1 + delta2 * x2)
    return h_t, h_t_ext


def fit_garch_m(kospi_df, garch_res):
    """
    GARCH-M(평균결합) 근사: r_t = mu + lambda*sqrt(h_t) + e_t 를
    이미 적합된 GJR-GARCH의 조건부변동성(sqrt(h_t))을 설명변수로 하는
    2단계 OLS로 추정한다. (완전결합 MLE 대비 효율성은 낮지만 방향성 파악엔 충분)
    반환: (mu_hat(%), lambda_hat) — 둘 다 % 수익률 단위 기준.
    """
    if garch_res is None:
        return np.nan, np.nan

    close = kospi_df["Close"]
    rets = 100 * np.log(close / close.shift(1)).dropna()
    cond_vol = garch_res.conditional_volatility

    n = min(len(rets), len(cond_vol))
    if n < 30:
        return np.nan, np.nan

    r = rets.iloc[-n:].values
    h_sqrt = cond_vol.iloc[-n:].values
    X = np.column_stack([np.ones(n), h_sqrt])

    try:
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        mu_hat, lambda_hat = float(beta[0]), float(beta[1])
    except Exception:
        return np.nan, np.nan

    return mu_hat, lambda_hat


def hybrid_bands(ma_t, daily_vol_ext_price, k1, k2, gamma, neg_shock_indicator, theta_call_price):
    """
    3번 공식(최종 하이브리드 밴드) 구현:
      Upper_t = MA_t + k1·√h_t_ext - θ_call
      Lower_t = MA_t - (k2 + γ·I(ε<0))·√h_t_ext
    daily_vol_ext_price = √h_t_ext 를 가격 단위로 환산한 값(스팟가격 × %변동성/100).
    gamma는 GJR-GARCH의 비대칭계수(음수 충격 시 추가 증폭), neg_shock_indicator는
    직전 충격(어제 수익률)이 음수였는지(0 또는 1).
    """
    gamma_eff = max(gamma, 0.0) if not np.isnan(gamma) else 0.0
    upper = ma_t + k1 * daily_vol_ext_price - theta_call_price
    lower = ma_t - (k2 + gamma_eff * neg_shock_indicator) * daily_vol_ext_price
    return upper, lower


def dynamic_band_forecast(
    spot,
    taylor_rows,
    volatility_path_pct,
    atr_value,
    gamma,
    last_return,
    taylor_error_multiplier=1.0,
    horizon=BELLMAN_HORIZON,
):
    """테일러 중심값과 GJR-GARCH 변동성으로 h일 동적 밴드를 계산한다."""
    if volatility_path_pct is None or len(volatility_path_pct) == 0 or np.isnan(spot):
        return []

    gamma_eff = max(float(gamma), 0.0) if not np.isnan(gamma) else 0.0
    negative_shock = last_return < 0
    rows = []
    for index, volatility_pct in enumerate(volatility_path_pct[:horizon]):
        taylor_row = taylor_rows[index] if index < len(taylor_rows) else None
        center = taylor_row["proj"] if taylor_row is not None else spot
        taylor_error = (
            (taylor_row["upper"] - center) * taylor_error_multiplier
            if taylor_row is not None else 0.0
        )
        garch_width = spot * float(volatility_pct) / 100.0
        base_width = max(garch_width, atr_value if not np.isnan(atr_value) else 0.0)
        lower_multiplier = 1.0 + gamma_eff if negative_shock else 1.0
        upper = center + base_width + taylor_error
        lower = center - (base_width * lower_multiplier) - taylor_error
        rows.append({
            "h": index + 1,
            "center": center,
            "volatility_pct": float(volatility_pct),
            "upper": upper,
            "lower": lower,
            "width": base_width,
            "taylor_error": taylor_error,
            "leverage_applied": negative_shock and gamma_eff > 0,
        })
    return rows


def bellman_optimal_path(
    current_position,
    mu_daily,
    sigma_daily_path,
    states=POSITION_STATES,
    risk_aversion=3.0,
    rebal_cost=0.02,
):
    """
    벨만의 최적성 원리(backward induction, 가치반복)를 이용해
    향후 len(sigma_daily_path)영업일에 걸친 최적 포지션 경로를 계산한다.

    상태(state) = 헤지/인버스 포지션 비중 (0/25/50/75/100%)
    헤지/인버스 포지션은 기초자산(코스피)과 반대 방향으로 수익이 나므로,
    기대수익 항에는 -mu_daily(부호 반전)를 사용한다 — 즉 하락 신호(mu<0)일수록
    헤지 비중을 높이는 것이 보상을 극대화한다.

    각 시점의 보상(reward) = 기대수익 - 리스크비용 - 리밸런싱비용
        R(p, p_prev, t) = (p/100)*(-mu_daily)
                          - risk_aversion * (p/100)^2 * sigma_daily_path[t]^2
                          - rebal_cost * |p - p_prev| / 100
    종료조건: V_H(p) = 0 (마지막 시점 이후 가치는 0으로 정규화)
    이 값을 뒤에서부터(backward) 채워나가는 것이 벨만 방정식의 핵심이다:
        V_t(p_prev) = max_p [ R(p, p_prev, t) + V_{t+1}(p) ]

    ※ mu_daily, sigma_daily_path는 실제 확률분포가 아니라 방향점수/GARCH변동성에서
    유도한 단순화된 추정치이므로, 이 경로는 '참고용 다단계 계획'이지 확정적 예측이 아니다.
    """
    hedge_mu = -mu_daily  # 헤지/인버스 포지션 수익은 기초자산과 반대 부호
    H = len(sigma_daily_path)
    V_next = {p: 0.0 for p in states}
    policies = []  # policies[t][p_prev] = 그 시점에서의 최적 다음 포지션

    for t in reversed(range(H)):
        sigma_t = sigma_daily_path[t]
        V_curr = {}
        policy_t = {}
        for p_prev in states:
            best_val, best_p = -np.inf, p_prev
            for p in states:
                reward = (
                    (p / 100.0) * hedge_mu
                    - risk_aversion * ((p / 100.0) ** 2) * (sigma_t ** 2)
                    - rebal_cost * abs(p - p_prev) / 100.0
                )
                val = reward + V_next[p]
                if val > best_val:
                    best_val, best_p = val, p
            V_curr[p_prev] = best_val
            policy_t[p_prev] = best_p
        policies.insert(0, policy_t)
        V_next = V_curr

    v0 = V_next[current_position]

    # 초기 포지션에서 시작해 정책을 따라 앞으로(forward) 시뮬레이션
    path = [current_position]
    p_prev = current_position
    for t in range(H):
        p_next = policies[t][p_prev]
        path.append(p_next)
        p_prev = p_next

    return v0, path


def directional_score(df):
    """-100 ~ +100. +는 상승, -는 하락."""
    if len(df) < 70:
        return 0.0, 0.0, "데이터 부족"

    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    r5 = close.pct_change(5).iloc[-1] * 100
    r20 = close.pct_change(20).iloc[-1] * 100

    score = 0.0

    # 추세
    score += 30 if close.iloc[-1] > ma20.iloc[-1] else -30
    score += 25 if ma20.iloc[-1] > ma60.iloc[-1] else -25

    # 단기 모멘텀
    score += np.clip(r5 * 8, -20, 20)
    score += np.clip(r20 * 3, -15, 15)

    # 최근 종가 위치
    hi = close.rolling(60).max().iloc[-1]
    lo = close.rolling(60).min().iloc[-1]
    pos = (close.iloc[-1] - lo) / (hi - lo) if hi != lo else 0.5
    score += (pos - 0.5) * 20

    score = float(np.clip(score, -100, 100))
    confidence = float(np.clip(abs(score), 0, 100))

    if score >= 25:
        direction = "상승 우세"
    elif score <= -25:
        direction = "하락 우세"
    else:
        direction = "중립 / 혼조"

    return score, confidence, direction


def position_level(score, confidence, basis_z=0.0):
    """
    0/25/50/75/100 레벨.
    하락 방향일수록 인버스/헤지 비중을 높이는 구조.
    Basis가 비정상적으로 벌어지면 한 단계 보수적으로 낮춘다.
    """
    effective = score

    # 현물 대비 선물의 괴리가 지나치게 크면 확신도 할인
    if abs(basis_z) >= 2:
        effective *= 0.65
    elif abs(basis_z) >= 1.5:
        effective *= 0.8

    if effective <= -65 and confidence >= 65:
        return 100
    if effective <= -45 and confidence >= 45:
        return 75
    if effective <= -25 and confidence >= 25:
        return 50
    if effective < 0:
        return 25
    return 0


def basis_filtered_score(score, basis_z, volume_ratio_value, volatility_ratio_value):
    """Basis 극단 구간에서 현물 방향 점수의 신뢰도만 보수적으로 할인한다."""
    if np.isnan(basis_z):
        return float(score), 1.0
    if abs(basis_z) >= 2.0:
        discount = 0.55 if (
            np.isnan(volume_ratio_value)
            or volume_ratio_value < 1.0
            or (not np.isnan(volatility_ratio_value) and volatility_ratio_value > VOLATILITY_RATIO_HOT)
        ) else 0.70
    elif abs(basis_z) >= 1.5:
        discount = 0.85
    else:
        discount = 1.0
    return float(score * discount), discount


def stabilize_mixed_level(proposed_level, score, records):
    """혼조 구간에서 25%p 단위의 잦은 레벨 전환을 완화한다."""
    if not records or abs(score) >= 30:
        return proposed_level
    previous = records[-1].get("current_position")
    if previous not in {0, 25, 50} or abs(int(previous) - proposed_level) != 25:
        return proposed_level
    return int(previous)


def regime_text(score, confidence):
    if confidence < 25:
        return "신호 약함"
    if score <= -60:
        return "강한 하방 위험"
    if score <= -25:
        return "하방 우세"
    if score >= 60:
        return "강한 상승"
    if score >= 25:
        return "상승 우세"
    return "중립"


def market_model_divergence(df, score, current_level):
    """강한 하방 모델 신호와 당일 반등이 충돌할 때 단계적 대응을 제안한다."""
    if len(df) < 70:
        return None

    close = df["Close"]
    today_return = float(close.pct_change().iloc[-1] * 100)
    return_5d = float(close.pct_change(5).iloc[-1] * 100)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    volume = volume_ratio(df, n=20)

    strong_rebound = today_return >= 1.0
    trend_confirmation = close.iloc[-1] > ma20 and return_5d > 0
    volume_confirmation = not np.isnan(volume) and volume >= 1.2

    if score <= -45 and strong_rebound:
        if trend_confirmation and volume_confirmation:
            suggested_level = max(0, current_level - 50)
            status = "강한 반등 확인"
            action = "헤지를 50%p까지 단계적으로 축소하고 추세 확인"
        else:
            suggested_level = max(0, current_level - 25)
            status = "반등 발생, 추세 전환 미확인"
            action = "헤지를 25%p만 축소하고 다음 종가 확인"
        return {
            "status": status,
            "action": action,
            "suggested_level": suggested_level,
            "today_return": today_return,
            "return_5d": return_5d,
            "volume_ratio": volume,
            "trend_confirmation": trend_confirmation,
            "volume_confirmation": volume_confirmation,
        }

    return {
        "status": "모델-시장 괴리 없음",
        "action": "현재 모델 헤지 단계 유지",
        "suggested_level": current_level,
        "today_return": today_return,
        "return_5d": return_5d,
        "volume_ratio": volume,
        "trend_confirmation": trend_confirmation,
        "volume_confirmation": volume_confirmation,
    }


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ 데이터 설정")
approved_adjustments = read_adjustments(MODEL_ADJUSTMENT_PATH)
energy_buffer_multiplier = float(approved_adjustments.get("energy_buffer_multiplier", 1.0))
risk_aversion_delta = float(approved_adjustments.get("risk_aversion_delta", 0.0))

kospi_ticker = st.sidebar.text_input("KOSPI 종합지수 (계산 기준)", DEFAULTS["KOSPI 현물"])
ks200_ticker = st.sidebar.text_input("KOSPI200 현물 (참고용, 계산에는 미반영)", DEFAULTS["KOSPI200 현물"])
futures_ticker = st.sidebar.text_input("KOSPI200 선물 (참고용, 계산에는 미반영)", DEFAULTS["KOSPI200 선물"])

st.sidebar.divider()
st.sidebar.caption("매크로 게이트용 데이터")
krw_ticker = st.sidebar.text_input("USD/KRW 환율", DEFAULTS["USD/KRW"])
vix_ticker = st.sidebar.text_input("VIX 지수", DEFAULTS["VIX"])
global_ticker = st.sidebar.text_input("미국장 프록시 (나스닥 종합)", DEFAULTS["미국장 프록시"])

st.sidebar.divider()
st.sidebar.caption("신규 게이트 (v2)")
es_futures_ticker = st.sidebar.text_input("S&P500 선물 (ES=F)", "ES=F")
nq_futures_ticker = st.sidebar.text_input("나스닥100 선물 (NQ=F)", "NQ=F")
futures_conflict_threshold = st.sidebar.slider("선물 충돌 판정 기준(%)", 0.1, 2.0, 0.5, 0.1)
leader_divergence_threshold = st.sidebar.slider("대장주 이탈 판정 기준(%)", -2.0, 0.0, -0.3, 0.1)
intraday_monitor_ticker = st.sidebar.text_input("1시간 델타 모니터링 티커", DEFAULTS["KOSPI 현물"])

st.sidebar.divider()
st.sidebar.caption("벨만 최적화(다단계 포지션 계획) 파라미터")
risk_aversion_default = float(np.clip(3.0 + risk_aversion_delta, 0.5, 10.0))
risk_aversion = st.sidebar.slider("리스크회피계수", 0.5, 10.0, risk_aversion_default, 0.5)
rebal_cost_pct = st.sidebar.slider("리밸런싱 비용 (%)", 0.0, 5.0, 2.0, 0.5)
rebal_cost = rebal_cost_pct / 100.0

st.sidebar.divider()
st.sidebar.caption("확장형 GJR-GARCH + 하이브리드 밴드 파라미터")
delta1_volume = st.sidebar.slider("δ1 거래량 소진 가중치", 0.0, 1.0, 0.3, 0.05)
delta2_basis = st.sidebar.slider("δ2 괴리(Basis) 극단도 가중치", 0.0, 1.0, 0.2, 0.05)
k1_upper = st.sidebar.slider("k1 상단 승수", 0.5, 6.0, 2.0, 0.5)
k2_lower = st.sidebar.slider("k2 하단 승수", 0.5, 6.0, 2.0, 0.5)
theta_call_pct = st.sidebar.slider("θ_call 상단 압축계수 (스팟가격 대비 %)", 0.0, 3.0, 0.0, 0.1)

period = st.sidebar.selectbox(
    "분석 기간",
    ["6mo", "1y", "2y", "5y"],
    index=1,
)

st.sidebar.caption(
    "※ 무료 데이터 제공처의 티커/지연 여부는 변할 수 있습니다. "
    "실전 매매 전에는 KRX/증권사 데이터로 교차검증하세요. "
    "방향 점수/지지·저항선 등 모든 계산은 KOSPI 종합지수만 사용하며, "
    "KOSPI200과 선물은 티커를 입력해도 화면에 참고용으로만 표시됩니다."
)

# ---------------- Load ----------------
kospi, kospi_source = load_data(kospi_ticker, period, with_source=True)
ks200 = load_data(ks200_ticker, period)
futures = load_data(futures_ticker, period)
krw = load_data(krw_ticker, period)
vix = load_data(vix_ticker, period)
global_market = load_data(global_ticker, period)

st.title("📊 KOSPI Market Decision Engine")
st.caption(
    f"코스피 종합지수 기준 계산 · KOSPI200/선물은 참고용 표시 · "
    f"코스피 데이터 출처: {kospi_source or '없음'} · 기준일: {kospi.index[-1].date() if not kospi.empty else '-'} · "
    f"배포 커밋: {deployment_revision()}"
)

if kospi.empty:
    st.error(
        "KOSPI 종합지수 데이터를 불러오지 못했습니다. "
        "사이드바의 티커(기본값 ^KS11)를 확인하세요."
    )
    st.stop()

# ---------------- Main model ----------------
# 모든 계산(방향 점수/신뢰도/지지·저항 밴드/차트)은 코스피 종합지수(kospi_ticker, 기본 ^KS11) 기준.
spot_levels = make_levels(kospi)
overnight_stats = overnight_gap_stats(kospi)
energy_level = energy_levels(kospi, buffer_multiplier=energy_buffer_multiplier)
energy_backtest = energy_level_backtest(kospi, buffer_multiplier=energy_buffer_multiplier)
price_volume_energy = price_volume_momentum(kospi)
obv_signal = obv_analysis(kospi)
volatility_volume = volatility_volume_energy(kospi)
conditional_gap_stats = conditional_overnight_gap_stats(kospi, global_market)
swing_levels = swing_level_clusters(kospi)
rebound_analysis = rebound_scenario_analysis(kospi, swing_levels["support"])
spot_score, spot_conf, spot_direction = directional_score(kospi)
spot = spot_levels["close"]

# KOSPI200 현물은 참고용으로만 표시 (계산에는 미반영)
ks200_price = float(ks200["Close"].iloc[-1]) if not ks200.empty else np.nan

fut_score = fut_conf = 0.0
fut_direction = "데이터 없음"
if not futures.empty:
    fut_score, fut_conf, fut_direction = directional_score(futures)

future = float(futures["Close"].iloc[-1]) if not futures.empty else np.nan

# 선물-KOSPI200 괴리(Basis)는 내부적으로만 계산하며 화면에는 참고 수치로만 노출한다.
basis = np.nan
basis_z = 0.0
if not np.isnan(future) and not np.isnan(ks200_price) and ks200_price != 0:
    basis = (future - ks200_price) / ks200_price * 100

    if not ks200.empty:
        basis_series = (
            futures["Close"].reindex(ks200.index).ffill() - ks200["Close"]
        ) / ks200["Close"] * 100
        basis_z_series = zscore(basis_series, 60).dropna()
        if not basis_z_series.empty:
            basis_z = float(basis_z_series.iloc[-1])

# ---- 신뢰도 보정 지표 (거래량 / 변동성 / 주봉 교차 / 매크로) ----
vol_ratio_val = volume_ratio(kospi, n=20)
vola_ratio_val = volatility_ratio(kospi, short_n=5, long_n=60)
weekly_ma_val = weekly_ma20(kospi, ma_n=20)
krw_5d_pct, vix_last, macro_risk, macro_multiplier = macro_gate(krw, vix)

# Basis 극단도는 선물 방향을 합성하지 않고, 현물 점수의 노이즈 신뢰도만 할인한다.
composite_score, basis_discount = basis_filtered_score(
    spot_score, basis_z, vol_ratio_val, vola_ratio_val
)
composite_conf = float(np.clip(abs(composite_score), 0, 100))
prediction_records = _read_prediction_log(PREDICTION_LOG_PATH)
regime = regime_text(composite_score, composite_conf)
level = position_level(composite_score, composite_conf, basis_z)
level = stabilize_mixed_level(level, composite_score, prediction_records)

# ---- v2 신규 게이트 1: ES=F/NQ=F 선물 충돌 게이트 ----
es_last, es_pct = load_live_quote(es_futures_ticker)
nq_last, nq_pct = load_live_quote(nq_futures_ticker)
futures_pct_for_gate = es_pct if not np.isnan(es_pct) else nq_pct  # KOSPI 모델이므로 S&P500 우선
futures_gate_result = futures_collision_gate(
    composite_score, futures_pct_for_gate, threshold_pct=futures_conflict_threshold
)

# ---- v2 신규 게이트 2: 대장주(삼성전자·SK하이닉스) 가격 괴리 킬스위치 ----
kospi_last_return_pct = float(pct_change(kospi).iloc[-1])
leader_returns = {}
for name, tkr in LEADER_TICKERS_DEFAULT.items():
    leader_df = load_data(tkr, "6mo")
    if not leader_df.empty and len(leader_df) > 1:
        leader_returns[name] = float(pct_change(leader_df).iloc[-1])
    else:
        leader_returns[name] = np.nan
leader_result = leader_divergence_check(
    kospi_last_return_pct, leader_returns, threshold_pct=leader_divergence_threshold
)

# 매크로 위험 신호(원화 약세 + VIX 급등) + 선물 충돌 + 대장주 이탈 시 진입 강도를 낮춘다.
leader_multiplier = 0.5 if leader_result.get("divergence") else 1.0
level_final = round(level * macro_multiplier * futures_gate_result["multiplier"] * leader_multiplier / 25) * 25
divergence_guidance = market_model_divergence(
    kospi, composite_score, level_final
)
rebound_exit = rebound_exit_guide(
    spot,
    swing_levels,
    energy_level,
    spot_levels["resistance"],
    vol_ratio_val,
    composite_score,
)
overnight_plan = overnight_action_plan(
    level_final,
    overnight_stats,
    conditional_gap_stats,
    rebound_analysis,
    vol_ratio_val,
)
overnight_timeline = overnight_execution_timeline(
    level_final,
    overnight_plan,
    overnight_stats,
)

# ---- GJR-GARCH(1,1,1) 조건부 변동성 및 EWMA fallback ----
garch_res, garch_today_vol_pct, garch_vol_path_pct = fit_gjr_garch(kospi)
volatility_model_source = "GJR-GARCH"
if garch_res is None or garch_vol_path_pct is None or np.isnan(garch_today_vol_pct):
    garch_today_vol_pct, garch_vol_path_pct = ewma_volatility_path(kospi)
    volatility_model_source = "EWMA fallback"

# ---- 확장형 조건부분산 (거래량 소진 + 괴리 극단도 외생변수 반영) ----
h_t_raw, h_t_ext = extended_garch_variance(
    garch_today_vol_pct, vol_ratio_val, basis_z, delta1_volume, delta2_basis
)
ext_factor = (h_t_ext / h_t_raw) if (not np.isnan(h_t_raw) and h_t_raw != 0) else np.nan

# ---- GARCH-M(평균결합) 2단계 근사: 변동성 → 기대수익 피드백 ----
garch_m_mu, garch_m_lambda = fit_garch_m(kospi, garch_res)

# ---------------- Header cards ----------------
c1, c3, c4, c5 = st.columns(4)

c1.metric(
    "KOSPI 종합지수",
    f"{spot:,.2f}",
    f"{pct_change(kospi).iloc[-1]:+.2f}%"
)

c3.metric("방향 점수", f"{composite_score:+.1f}")
c4.metric("신뢰도", f"{composite_conf:.0f}%")
c5.metric(
    "헤지/인버스 레벨",
    f"{level_final}%",
    None if level_final == level else f"매크로 보정 전 {level}%"
)

if not futures.empty:
    st.caption(
        f"선물 (참고용): {future:,.2f} ({pct_change(futures).iloc[-1]:+.2f}%) "
        "— 방향 점수 계산에는 반영되지 않습니다."
    )

st.divider()
st.subheader("⚡ 핵심 시장 에너지 계산")
st.latex(
    r"R_{dynamic} = P_{current} + \left(ATR_{14} \times "
    r"\frac{Volume_{actual}}{Volume_{avg}} \times "
    r"\left(1 - \frac{|\Delta P_{5d}|}{P_{current}}\right)\right)"
)
if energy_level is None:
    st.warning("현재 데이터가 부족해 시장 에너지 저항선을 계산하지 못했습니다.")
else:
    st.write(
        f"`{energy_level['current']:,.2f} + ({energy_level['atr14']:,.2f} × "
        f"{energy_level['volume_ratio']:.2f} × "
        f"(1 - {abs(energy_level['momentum_return_5d']):.4f}))`"
    )
    energy_top = st.columns(4)
    energy_top[0].metric("현재가 P_current", f"{energy_level['current']:,.2f}")
    energy_top[1].metric("ATR14", f"{energy_level['atr14']:,.2f}")
    energy_top[2].metric("거래량 배수", f"{energy_level['volume_ratio']:.2f}x")
    energy_top[3].metric("R_dynamic 동적 저항", f"{energy_level['dynamic_resistance']:,.2f}")
    st.caption(
        f"최종 에너지 폭 {energy_level['energy_width']:,.2f} · "
        f"동적 지지선 {energy_level['dynamic_support']:,.2f} · "
        f"5일 수익률 {energy_level['momentum_return_5d'] * 100:+.2f}% · "
        "스윙 저항선과 거래량 배수를 직접 결합하지 않음"
    )

st.divider()
st.subheader("📊 반등의 질: 가격·거래량·OBV 검증")
st.caption(
    "가격 상승만으로 국면 전환을 확정하지 않습니다. 최근 거래 참여량의 상대적 강도, "
    "OBV 방향, 변동성·거래량 결합 에너지를 함께 확인하고 과거 반등 표본의 확률과 비교합니다."
)
if price_volume_energy is None or obv_signal is None or volatility_volume is None:
    st.info("거래량 기반 지표를 계산할 수 있는 데이터가 부족합니다.")
else:
    quality_columns = st.columns(5)
    quality_columns[0].metric(
        "가격×거래량 에너지",
        f"{price_volume_energy['percentile']:.0f}퍼센타일",
        f"당일 수익률 {price_volume_energy['return_pct']:+.2f}%",
    )
    quality_columns[1].metric("당일 거래량", f"{price_volume_energy['volume_ratio']:.2f}x")
    quality_columns[2].metric("OBV 20일 변화", f"{obv_signal['obv_change_units']:+.2f} 평균거래량")
    quality_columns[3].metric(
        "ATR×거래량 에너지",
        f"{volatility_volume['percentile']:.0f}퍼센타일",
        f"ATR {volatility_volume['atr_pct']:.2f}% × {volatility_volume['volume_ratio']:.2f}x",
    )
    quality_columns[4].metric("OBV 판정", obv_signal["signal"])

    supply_confirmed = (
        price_volume_energy["volume_ratio"] >= 1.2
        and obv_signal["obv_change"] >= 0
        and price_volume_energy["percentile"] >= 50
    )
    if supply_confirmed:
        st.success(
            "실제 수급 확인 쪽에 가점: 거래량과 OBV가 반등을 지지합니다. "
            "다만 저항 돌파와 지지선 유지가 추가로 확인되어야 국면 전환으로 분류합니다."
        )
    elif obv_signal["bullish_divergence"]:
        st.info(
            "강세 OBV 다이버전스: 가격은 약했지만 누적 수급은 개선되었습니다. "
            "국면 전환의 초기 후보이지 확정 신호는 아닙니다."
        )
    else:
        st.warning(
            "수급 확인 부족: 가격 반등이 거래량·OBV로 충분히 뒷받침되지 않습니다. "
            "기술적 반등 경계 시나리오를 우선 유지합니다."
        )

# ---------------- Decision ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("🎯 현재 판단")
    st.markdown(f"### {regime}")
    st.write(f"**방향:** {spot_direction}")
    st.write(f"**종합 점수:** `{composite_score:+.1f} / 100`")
    st.write(f"**신뢰도:** `{composite_conf:.0f}%`")
    st.write(f"**권장 헤지/인버스 단계:** `{level}%`")
    if level_final != level:
        st.write(f"**매크로 보정 후 최종 진입 강도:** `{level_final}%` (위험 신호로 절반 축소)")

    if composite_score <= -45:
        st.warning(
            "하방 위험이 높은 구간입니다. "
            "단일 진입보다 분할 대응을 우선하세요."
        )
    elif composite_score >= 45:
        st.success(
            "상승 추세 우세입니다. "
            "과도한 헤지는 줄이고 추세 추종 여부를 검토하세요."
        )
    else:
        st.info("신호가 혼조입니다. 신규 포지션은 보수적으로 접근하세요.")

with right:
    st.subheader("📐 현·선물 괴리 (참고용)")
    st.caption("※ 아래 수치는 참고 정보이며, 위 방향 점수/신뢰도 계산에는 반영되지 않습니다.")
    if not np.isnan(basis):
        st.metric("Basis", f"{basis:+.3f}%")
        st.write(f"Basis Z-score: `{basis_z:+.2f}`")
        st.write(f"현물 방향점수 할인: `{basis_discount:.0%}`")

        if abs(basis_z) >= 2:
            st.error("괴리 극단: 현물 방향점수에 강한 노이즈 할인 적용")
        elif abs(basis_z) >= 1.5:
            st.warning("괴리 확대: 추격 진입 주의")
        else:
            st.success("괴리 정상 범위")
    else:
        st.info("선물 데이터를 사용하지 않아 Basis는 계산하지 않습니다. (현물 단독 분석 모드)")

# ---------------- Overnight gap risk ----------------
st.divider()
st.subheader("🌙 오버나이트 갭 리스크: 홀딩 vs 개장 후 재진입")
st.caption(
    "현재 분석 기간의 실제 KOSPI 일봉에서 '오늘 종가 → 다음 거래일 시가'를 계산했습니다. "
    "예측 확률이 아니라 관측된 과거 분포이며, 데이터가 없는 당일 갭은 포함하지 않습니다."
)

if overnight_stats is None:
    st.info("익일 시가를 포함한 일봉 표본이 부족해 갭 통계를 계산할 수 없습니다.")
else:
    gap1, gap2, gap3, gap4 = st.columns(4)
    gap1.metric("익일 상승 갭", f"{overnight_stats['up_probability']:.1f}%")
    gap2.metric("익일 하락 갭", f"{overnight_stats['down_probability']:.1f}%")
    gap3.metric("절대 갭 평균", f"{overnight_stats['mean_abs']:.2f}%")
    gap4.metric("절대 갭 90퍼센타일", f"{overnight_stats['p90_abs']:.2f}%")

    gap_table = pd.DataFrame([
        {
            "실제 익일 갭 통계": "표본 수",
            "값": f"{overnight_stats['count']:,}회",
            "해석": "종가와 다음 거래일 시가가 모두 있는 관측치",
        },
        {
            "실제 익일 갭 통계": "평균 / 중앙값",
            "값": f"{overnight_stats['mean']:+.2f}% / {overnight_stats['median']:+.2f}%",
            "해석": "갭 방향의 중심 위치",
        },
        {
            "실제 익일 갭 통계": "10~90퍼센타일",
            "값": f"{overnight_stats['p10']:+.2f}% ~ {overnight_stats['p90']:+.2f}%",
            "해석": "관측 갭의 중앙 80% 범위",
        },
        {
            "실제 익일 갭 통계": "절대 갭 1% 이상",
            "값": f"{overnight_stats['large_gap_probability']:.1f}%",
            "해석": "종가 홀딩 시 큰 시가 변동에 노출된 빈도",
        },
    ]).set_index("실제 익일 갭 통계")
    st.dataframe(gap_table, width="stretch")
    st.info(
        f"이 표본에서 종가 홀딩 시 다음 시가의 평균 절대 변동은 "
        f"{overnight_stats['mean_abs']:.2f}%이고, 1% 이상 갭은 "
        f"{overnight_stats['large_gap_probability']:.1f}%에서 발생했습니다. "
        "개장 후 재진입은 이 갭 노출을 줄이는 대신, 갭 이후 가격으로 진입하게 됩니다."
    )

    if conditional_gap_stats is None:
        st.warning("나스닥 프록시와 KOSPI 날짜를 매칭할 수 없어 미국장 조건부 갭 통계는 표시하지 않습니다.")
    else:
        st.write(
            f"**현재 미국장 조건:** {conditional_gap_stats['latest_condition']} · "
            f"최근 나스닥 수익률 `{conditional_gap_stats['latest_us_return']:+.2f}%` · "
            f"조건부 매칭 표본 `{conditional_gap_stats['sample_count']}건`"
        )
        conditional_table = pd.DataFrame([
            {
                "미국장 조건": row["condition"],
                "표본": f"{row['count']}건",
                "다음날 상승 갭": f"{row['up_probability']:.1f}%",
                "다음날 하락 갭": f"{row['down_probability']:.1f}%",
                "±1% 이상 갭": f"{row['large_gap_probability']:.1f}%",
                "상승 갭 95% CI": f"{row['ci_lower']:.1f}%~{row['ci_upper']:.1f}%",
            }
            for row in conditional_gap_stats["rows"]
        ]).set_index("미국장 조건")
        st.dataframe(conditional_table, width="stretch")

    st.subheader("✅ 오늘의 실행안")
    plan_columns = st.columns(3)
    plan_columns[0].metric("현재 모델 레벨", f"{level_final}%")
    plan_columns[1].metric("오버나이트 권장", f"{overnight_plan['level']}%")
    plan_columns[2].metric("실행 신뢰도", f"{overnight_plan['confidence']}%")
    if overnight_plan["level"] < level_final:
        st.warning(overnight_plan["action"])
    else:
        st.success(overnight_plan["action"])
    if overnight_plan["reasons"]:
        st.write("판단 근거: " + " · ".join(overnight_plan["reasons"]))

    if overnight_timeline is not None:
        st.subheader("🕒 내일 아침 갭 대응 타임라인")
        st.caption(
            f"현재 갭 위험도: {overnight_timeline['gap_risk']} · "
            f"장 마감 전 권장 헤지/인버스: {overnight_timeline['recommended_level']}%. "
            "자동 주문이 아닌 수동 확인용 실행 가이드입니다."
        )
        timeline_table = pd.DataFrame(overnight_timeline["stages"]).set_index("시간")
        st.dataframe(timeline_table, width="stretch")
        if overnight_timeline["recommended_level"] <= 50 and level_final > 50:
            st.warning(
                "갭 위험이 높습니다. 장 마감 전 헤지/인버스 물량을 50% 수준으로 줄이고, "
                "내일 첫 15분 수급 확인 전 추가 자동 주문을 실행하지 마세요."
            )

    st.subheader("📉 내일 갭 시나리오와 실전 실행 가이드")
    if overnight_stats["down_probability"] > overnight_stats["up_probability"]:
        overall_bias = "하락 갭"
    elif overnight_stats["up_probability"] > overnight_stats["down_probability"]:
        overall_bias = "상승 갭"
    else:
        overall_bias = "방향 우세 없음"

    conditional_row = overnight_plan.get("conditional_row")
    if conditional_row is not None:
        condition_text = (
            f"현재 미국장 조건({conditional_gap_stats['latest_condition']})에서 "
            f"상승 갭 {conditional_row['up_probability']:.1f}% · "
            f"하락 갭 {conditional_row['down_probability']:.1f}%"
        )
        conditional_bias = (
            "하락 갭 우세"
            if conditional_row["down_probability"] > conditional_row["up_probability"]
            else "상승 갭 우세"
            if conditional_row["up_probability"] > conditional_row["down_probability"]
            else "방향 우세 없음"
        )
    else:
        condition_text = "미국장 조건부 표본이 없어 전체 KOSPI 갭 분포만 사용합니다."
        conditional_bias = "조건부 판단 불가"

    st.write(
        f"과거 전체 표본에서는 **{overall_bias}**가 우세합니다 "
        f"(상승 {overnight_stats['up_probability']:.1f}% · "
        f"하락 {overnight_stats['down_probability']:.1f}%). {condition_text}"
    )
    st.write(
        f"절대 갭 평균은 **{overnight_stats['mean_abs']:.2f}%**, "
        f"±1% 이상 갭 빈도는 **{overnight_stats['large_gap_probability']:.1f}%**입니다. "
        "이는 방향 예측 확률이 아니라 과거 관측 빈도이므로, 갭 반대 방향 손실 한도를 먼저 정해야 합니다."
    )

    if level_final >= 75 and overnight_plan["level"] < level_final:
        st.warning(
            f"모델 헤지 레벨은 {level_final}%지만 갭 위험 때문에 권장 보유량은 "
            f"{overnight_plan['level']}%입니다. 장 마감 전 일부 축소 후, "
            "개장 갭과 첫 15분 수급을 확인하고 남은 물량을 재조정하는 시나리오입니다."
        )
    elif level_final >= 75:
        st.info(
            f"모델 헤지 레벨 {level_final}%를 유지할 수 있는 조건이지만, "
            f"{overnight_stats['mean_abs']:.2f}% 평균 갭을 감내해야 합니다. "
            "시가 급등 시 추가 손실 한도와 강제 축소 기준을 미리 정하세요."
        )
    else:
        st.info(
            f"현재 모델 헤지 레벨은 {level_final}%로 강한 방향 베팅이 아닙니다. "
            "오버나이트보다 개장 후 방향 확인 뒤 재진입하는 보수적 접근이 적합합니다."
        )
    st.caption(f"미국장 조건부 방향: {conditional_bias} · 자동 주문은 실행하지 않습니다.")
    st.caption(
        "이 실행안은 과거 갭 빈도와 현재 데이터 조건을 결합한 리스크 관리 규칙입니다. "
        "수익 방향을 보장하지 않으며, 실제 주문 전 상품의 손절·증거금·추적오차를 별도로 확인하세요."
    )

# ---------------- Swing support/resistance clusters ----------------
st.divider()
st.subheader("🧭 실제 반응 기반 스윙 지지·저항 클러스터")
st.caption(
    "최근 분석 기간의 국소 스윙 고점·저점을 찾고, 현재 ATR의 0.5배 이상 차이 나는 가격만 "
    "같은 구간으로 묶었습니다. 최소 2회 반응한 클러스터만 표시합니다."
)
st.info(
    "단위 주의: 스윙 저항선은 지수 가격(포인트), 거래량 비율은 무차원 배수입니다. "
    "두 값을 나누거나 곱한 가상 가격은 계산하지 않으며, 거래량은 아래 에너지 폭 공식에서만 "
    "ATR에 곱해 변동 폭을 조정하는 보조 입력으로 사용합니다."
)

support_rows = [
    {
        "구분": "지지",
        "가격": f"{cluster['level']:,.2f}",
        "반응 횟수": cluster["reactions"],
        "최근 반응일": cluster["last_reaction"],
    }
    for cluster in swing_levels["support"][:5]
]
resistance_rows = [
    {
        "구분": "저항",
        "가격": f"{cluster['level']:,.2f}",
        "반응 횟수": cluster["reactions"],
        "최근 반응일": cluster["last_reaction"],
    }
    for cluster in swing_levels["resistance"][:5]
]
cluster_rows = support_rows + resistance_rows
if not cluster_rows:
    st.info("현재가 주변에서 2회 이상 반응한 스윙 지지·저항 클러스터를 찾지 못했습니다.")
else:
    st.dataframe(
        pd.DataFrame(cluster_rows).set_index("구분"),
        width="stretch",
    )
    st.caption(
        f"현재가: `{spot:,.2f}` · 가격 묶음 허용오차: "
        f"±{swing_levels['tolerance']:,.2f} · 가까운 지지부터, 가까운 저항부터 표시"
    )

# ---------------- Rebound exit guide ----------------
st.divider()
st.subheader("🎯 반등 목표가와 3단계 분할 청산 가이드")
st.caption(
    "가장 가까운 실제 저항 또는 동적 저항을 1차 목표 구간으로 삼습니다. "
    "목표가는 예측값이 아니라 과거 반응이 확인된 관찰 기준이며, 전량 매도를 강제하지 않습니다."
)

if rebound_exit is None:
    st.info("현재가 위에서 확인되는 저항 목표가가 부족해 분할 청산 기준을 계산할 수 없습니다.")
else:
    exit_metrics = st.columns(4)
    exit_metrics[0].metric("1차 목표가", f"{rebound_exit['first_target']:,.2f}")
    exit_metrics[1].metric("현재가 대비", f"+{rebound_exit['distance_pct']:.2f}%")
    exit_metrics[2].metric("목표 구간 하단", f"{rebound_exit['target_zone_low']:,.2f}")
    exit_metrics[3].metric("목표 구간 상단", f"{rebound_exit['target_zone_high']:,.2f}")

    exit_table = pd.DataFrame([
        {
            "단계": "1차 청산",
            "물량": "30%",
            "실행 기준": f"{rebound_exit['target_zone_low']:,.2f}~{rebound_exit['target_zone_high']:,.2f} 진입",
            "의미": "첫 저항에서 수익 일부 확정",
        },
        {
            "단계": "2차 관망/청산",
            "물량": "40%",
            "실행 기준": "저항 돌파 여부 확인",
            "의미": "거래량 돌파면 보유, 거부면 순차 청산",
        },
        {
            "단계": "최종 추세 물량",
            "물량": "30%",
            "실행 기준": "추세 점수와 지지선 이탈 확인",
            "의미": "추세가 유지될 때만 가장 늦게 정리",
        },
    ]).set_index("단계")
    st.dataframe(exit_table, width="stretch")

    st.write(
        f"**1차 목표 근거:** {rebound_exit['target_source']} · "
        f"현재가 `{spot:,.2f}` → 목표 `{rebound_exit['first_target']:,.2f}`"
    )
    if rebound_exit["breakout_confirmed"]:
        st.success(
            f"거래량 {vol_ratio_val:.2f}x로 돌파 확인 조건을 충족했습니다. "
            "1차 30%만 확정하고 잔여 물량은 다음 저항까지 추적하는 시나리오를 검토하세요."
        )
    else:
        volume_text = f"{vol_ratio_val:.2f}x" if not np.isnan(vol_ratio_val) else "N/A"
        st.warning(
            f"현재 거래량 {volume_text}는 돌파 확인 기준 {VOLUME_RATIO_STRONG:.1f}x 미만입니다. "
            "목표 구간에서 30%를 우선 확정하고, 저항 거부 시 2차 물량을 줄이는 보수적 대응을 권장합니다."
        )
    if rebound_exit["trend_holds"]:
        st.info(
            f"현재 방향 점수 {composite_score:+.1f}로 하방 추세 신호가 남아 있습니다. "
            "반등 중에는 30%를 먼저 확정하되, 남은 물량은 추세 전환 확인 전까지 단계적으로 관리하세요."
        )
    else:
        st.info(
            f"현재 방향 점수 {composite_score:+.1f}로 하방 추세 신호가 강하지 않습니다. "
            "저항 돌파와 지지선 유지가 함께 확인되면 최종 30%를 서둘러 정리하지 않는 시나리오도 가능합니다."
        )

# ---------------- Rebound scenario validation ----------------
st.divider()
st.subheader("🔬 반등 형태와 국면 전개 시나리오 검증")
st.caption(
    "과거 '직전 5일 -2% 이하 하락 후 당일 +1% 이상 반등' 사례를 찾아, 이후 5영업일의 실제 결과를 분류했습니다. "
    "국면 전환은 이후 수익률 +2% 이상, 기술적 반등 실패는 -1% 이하, 나머지는 추세 진행 중입니다."
)

if rebound_analysis["sample_count"] == 0:
    st.info("현재 분석 기간에서 비교 가능한 반등 사례가 부족합니다.")
else:
    current_setup = rebound_analysis["current"]
    scenario_metrics = st.columns(4)
    scenario_metrics[0].metric("당일 반등", f"{current_setup['today_return']:+.2f}%")
    scenario_metrics[1].metric("직전 5일", f"{current_setup['return_5d']:+.2f}%")
    scenario_metrics[2].metric("셋업 일치도", f"{current_setup['setup_score']}%")
    scenario_metrics[3].metric("검증 표본", f"{rebound_analysis['sample_count']}건")

    setup_flags = []
    setup_flags.append("당일 +1% 이상 반등" if current_setup["is_rebound"] else "당일 반등 조건 미충족")
    setup_flags.append("지지선 근접" if current_setup["support_touch"] else "지지선 근접 조건 미충족")
    setup_flags.append("종가가 MA20 상회" if current_setup["ma20"] < spot else "종가가 MA20 하회")
    st.write("현재 형태: " + " · ".join(setup_flags))

    scenario_table = pd.DataFrame([
        {
            "시나리오": row["scenario"],
            "발생 횟수": f"{row['count']}건",
            "과거 확률": f"{row['probability']:.1f}%",
            "95% 신뢰구간": f"{row['lower']:.1f}% ~ {row['upper']:.1f}%",
        }
        for row in rebound_analysis["scenarios"]
    ]).set_index("시나리오")
    st.dataframe(scenario_table, width="stretch")

    dominant = rebound_analysis["dominant"]
    st.info(
        f"가장 빈번했던 결과는 '{dominant['scenario']}'로 "
        f"{dominant['probability']:.1f}% ({dominant['lower']:.1f}%~{dominant['upper']:.1f}%, 95% CI)입니다. "
        f"현재 셋업의 과거 비교 표본은 {rebound_analysis['sample_label']} {rebound_analysis['sample_count']}건입니다. "
        "신뢰구간이 넓으면 표본 부족으로 확률 신뢰도가 낮다는 뜻이며, 모델 신호와 실제 수급이 충돌할 때는 이 구간을 보수적으로 해석해야 합니다."
    )

# ---------------- Model/market divergence ----------------
st.divider()
st.subheader("🔎 모델-시장 괴리 대응")
st.caption(
    "가격·5일 모멘텀·거래량만으로 모델 신호와 당일 시장 움직임의 충돌을 점검합니다. "
    "외국인·연기금 수급을 직접 사용하지 않으며, 아래 단계는 자동 주문이 아닌 참고안입니다."
)

if divergence_guidance is not None:
    dg1, dg2, dg3 = st.columns(3)
    dg1.metric("당일 수익률", f"{divergence_guidance['today_return']:+.2f}%")
    dg2.metric("5일 수익률", f"{divergence_guidance['return_5d']:+.2f}%")
    volume_text = (
        f"{divergence_guidance['volume_ratio']:.2f}x"
        if not np.isnan(divergence_guidance["volume_ratio"])
        else "N/A"
    )
    dg3.metric("거래량 비율", volume_text)

    if divergence_guidance["suggested_level"] != level_final:
        st.warning(
            f"{divergence_guidance['status']}: {divergence_guidance['action']}. "
            f"현재 {level_final}% → 참고 단계 {divergence_guidance['suggested_level']}%"
        )
    else:
        st.info(
            f"{divergence_guidance['status']}: {divergence_guidance['action']}."
        )

# ---------------- Reliability filters ----------------
st.divider()
st.subheader("🧪 신뢰도 보정 필터")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("**① 거래량 필터**")
    if not np.isnan(vol_ratio_val):
        st.metric("거래량 비율 (당일 / 20일 평균)", f"{vol_ratio_val:.2f}x")
        if vol_ratio_val >= VOLUME_RATIO_STRONG:
            st.success(f"평균 대비 {VOLUME_RATIO_STRONG}배 이상 → 밴드 터치 신뢰도 높음")
        else:
            st.warning("거래량 부족 → 밴드 터치 시 '가짜 이탈' 가능성")
    else:
        st.info("거래량 데이터를 사용할 수 없습니다.")

with r2:
    st.markdown("**② 변동성 비율**")
    if not np.isnan(vola_ratio_val):
        st.metric("변동성 비율 (5일σ / 60일σ)", f"{vola_ratio_val:.2f}")
        if vola_ratio_val > VOLATILITY_RATIO_HOT:
            st.warning(
                f"{VOLATILITY_RATIO_HOT} 초과 → 변동성 과열 구간. "
                "1~2σ 밴드는 무시하고 3σ 이상만 신뢰 권장"
            )
        else:
            st.success("변동성 정상 범위 → 모든 σ 밴드 참고 가능")
    else:
        st.info("변동성 비율을 계산할 데이터가 부족합니다.")

with r3:
    st.markdown("**④ 매크로 게이트**")
    krw_txt = f"{krw_5d_pct:+.2f}%" if not np.isnan(krw_5d_pct) else "N/A"
    vix_txt = f"{vix_last:.1f}" if not np.isnan(vix_last) else "N/A"
    st.write(f"USD/KRW 5일 변화율: `{krw_txt}`")
    st.write(f"VIX: `{vix_txt}`")
    if macro_risk:
        st.error("위험 신호 감지: 원화 약세 + VIX 급등 → 진입 강도 50% 축소 적용됨")
    else:
        st.success("매크로 위험 신호 없음")

st.caption(
    "③ 다중 타임프레임(주봉) 교차 검증 결과는 아래 표준편차 밴드 표의 "
    "'신뢰도' 열에 '★ 주봉 MA20 겹침'으로 표시됩니다."
)

# ---------------- v2: 신규 게이트 3종 ----------------
st.divider()
st.subheader("🆕 v2 신규 게이트: 선물 충돌 · 대장주 이탈 · 1시간 가속도")
st.caption(
    "옛날 버전(kospi_model.py)과 구분하기 위해 새 파일(kospi_engine.py)로 분리했습니다. "
    "대장주 이탈 판정은 실제 순매수 금액(KRW) 데이터가 무료로 없어, 대신 실제 "
    "대장주 가격 등락률로 구현했습니다 — '수급 이탈'이 아니라 '가격 괴리'로 읽어야 합니다."
)

g1, g2 = st.columns(2)
with g1:
    st.markdown("**① ES=F/NQ=F 선물 충돌 게이트**")
    es_txt = f"{es_pct:+.2f}%" if not np.isnan(es_pct) else "N/A"
    nq_txt = f"{nq_pct:+.2f}%" if not np.isnan(nq_pct) else "N/A"
    st.write(f"ES=F: `{es_txt}` · NQ=F: `{nq_txt}`")
    if futures_gate_result["conflict"]:
        st.error(f"충돌 감지 → 진입 강도 추가 50% 축소 적용됨\n\n{futures_gate_result['detail']}")
    elif np.isnan(futures_pct_for_gate):
        st.info("선물 실시간가를 가져오지 못했습니다 (장 마감/데이터 지연 가능).")
    else:
        st.success(f"충돌 없음 — {futures_gate_result['detail']}")

with g2:
    st.markdown("**② 대장주(삼성전자·SK하이닉스) 가격 괴리**")
    if leader_result.get("available"):
        st.write(leader_result["detail"])
        if leader_result["divergence"]:
            st.error("대장주 이탈 감지 → 진입 강도 추가 50% 축소 적용됨")
        else:
            st.success("대장주 동조 — 게이트 미발동")
    else:
        st.info("대장주 가격 데이터를 가져오지 못했습니다.")

st.markdown("**③ 1시간 델타·가속도 모니터 (장중 내내 롤링 계산)**")
intraday_hourly_df = load_data(intraday_monitor_ticker, "5d", interval="60m")
intraday_calc_df = intraday_delta_acceleration(intraday_hourly_df)

if intraday_calc_df is None or intraday_calc_df["delta_pct"].dropna().empty:
    st.info(
        "1시간봉 데이터를 가져오지 못했습니다. 야후파이낸스는 60분봉을 "
        "최근 약 730일까지만 제공하며, 일부 지수는 무료로 제공되지 않을 수 있습니다."
    )
    intraday_latest_dict = None
else:
    valid_rows = intraday_calc_df.dropna(subset=["delta_pct"])
    latest_row = valid_rows.iloc[-1]
    intraday_latest_dict = {"delta_pct": latest_row["delta_pct"], "accel": latest_row.get("accel", np.nan)}

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("최근 1시간 델타", f"{latest_row['delta_pct']:+.3f}%")
    accel_val = latest_row.get("accel", np.nan)
    ic2.metric("가속도 (델타 변화량)", f"{accel_val:+.3f}%p" if not np.isnan(accel_val) else "N/A (당일 첫 봉)")
    if len(valid_rows) >= 2:
        swing = latest_row["delta_pct"] - valid_rows.iloc[-2]["delta_pct"]
        ic3.metric("직전 시간 대비 스윙 폭", f"{abs(swing):.3f}%p")
        if abs(swing) >= 0.5 and np.sign(latest_row["delta_pct"]) != np.sign(valid_rows.iloc[-2]["delta_pct"]):
            st.warning(
                f"1시간 델타 반전 감지: {valid_rows.iloc[-2]['delta_pct']:+.2f}% → "
                f"{latest_row['delta_pct']:+.2f}% (총 {abs(swing):.2f}%p 스윙)"
            )
    st.line_chart(valid_rows[["delta_pct", "accel"]].tail(24).rename(
        columns={"delta_pct": "시간당 델타(%)", "accel": "가속도(%p)"}
    ))

st.markdown("**④ 세션 요약 리포트** (새 판단 로직 아님 — 위 결과를 한 장으로 요약)")
session_report = build_session_report(
    spot_price=float(kospi["Close"].iloc[-1]),
    composite_score=composite_score,
    confidence=composite_conf,
    level_final=level_final,
    leader_result=leader_result,
    futures_result=futures_gate_result,
    intraday_latest=intraday_latest_dict,
    macro_risk=macro_risk,
)
st.markdown(session_report)

# ---------------- Bands ----------------
st.divider()
st.subheader("📏 KOSPI 종합지수 동적 밴드")

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("60일 저점", f"{spot_levels['low60']:,.2f}")
b2.metric("지지선", f"{spot_levels['support']:,.2f}")
b3.metric("현재", f"{spot_levels['close']:,.2f}")
b4.metric("저항선", f"{spot_levels['resistance']:,.2f}")
b5.metric("60일 고점", f"{spot_levels['high60']:,.2f}")

st.write(
    f"MA20 `{spot_levels['ma20']:,.2f}` · "
    f"MA60 `{spot_levels['ma60']:,.2f}` · "
    f"ATR14 `{spot_levels['atr']:,.2f}`"
)

# ---------------- Energy-based dynamic levels ----------------
st.divider()
st.subheader("⚡ 시장 에너지 기반 동적 저항·지지선")
st.caption(
    "에너지 폭 = ATR14 × 거래량 배수 × (1 - |5일 수익률|)입니다. "
    "현재 거래 참여량으로 도달 가능한 상단과 하단을 계산한 참고선이며, 확정적인 매물대는 아닙니다."
)
st.latex(
    r"R_{dynamic} = P_{current} + \left(ATR_{14} \times "
    r"\frac{Volume_{actual}}{Volume_{avg}} \times "
    r"\left(1 - \frac{|\Delta P_{5d}|}{P_{current}}\right)\right)"
)
st.caption(
    "P_current는 현재 지수, ATR14는 기본 변동성 단위, "
    "Volume_actual / Volume_avg는 거래량 에너지 배수, "
    "(1 - |ΔP_5d| / P_current)는 모멘텀 저항 계수입니다. "
    "거래량 배수는 가격과 직접 나누거나 곱하는 값이 아니라 ATR 폭을 조정하는 무차원 계수입니다."
)

if energy_level is None:
    st.info("에너지 레벨 계산에 필요한 일봉 데이터가 부족합니다.")
else:
    momentum_pct = energy_level["momentum_return_5d"] * 100
    momentum_abs_ratio = abs(energy_level["momentum_return_5d"])
    st.markdown("**현재 데이터 대입**")
    st.write(
        f"`R_dynamic = {energy_level['current']:,.2f} + "
        f"({energy_level['atr14']:,.2f} × {energy_level['volume_ratio']:.2f} × "
        f"(1 - {momentum_abs_ratio:.4f}))`"
    )
    st.write(
        f"거래량 조정 ATR 폭: `{energy_level['atr14']:,.2f} × "
        f"{energy_level['volume_ratio']:.2f} = "
        f"{energy_level['atr14'] * energy_level['volume_ratio']:,.2f}` · "
        f"모멘텀 저항 계수: `{energy_level['momentum_factor']:.4f}` · "
        f"5일 수익률: `{momentum_pct:+.2f}%`"
    )
    st.write(
        f"최종 에너지 폭: `{energy_level['energy_width']:,.2f}` · "
        f"동적 저항선: `{energy_level['dynamic_resistance']:,.2f}` · "
        f"동적 지지선: `{energy_level['dynamic_support']:,.2f}`"
    )
    st.info(
        "이 동적 저항선은 현재 거래량 에너지로 도달 가능한 단기 경계입니다. "
        "과거 스윙 저항 클러스터(예: 7,194.07)와 같은 가격대 매물대가 아니며, "
        "두 수치를 서로 나누거나 곱해 새로운 가격을 만들지 않습니다."
    )
    energy_columns = st.columns(5)
    energy_columns[0].metric("ATR14", f"{energy_level['atr14']:,.2f}")
    energy_columns[1].metric("거래량 에너지", f"{energy_level['volume_ratio']:.2f}x")
    energy_columns[2].metric("5일 모멘텀 계수", f"{energy_level['momentum_factor']:.4f}")
    energy_columns[3].metric("동적 저항선", f"{energy_level['dynamic_resistance']:,.2f}")
    energy_columns[4].metric("동적 지지선", f"{energy_level['dynamic_support']:,.2f}")

    energy_table = pd.DataFrame([
        {"구분": "현재가", "값": f"{energy_level['current']:,.2f}", "산출 의미": "계산 기준 종가"},
        {"구분": "에너지 폭", "값": f"{energy_level['energy_width']:,.2f}", "산출 의미": "현재 참여량으로 가중한 ATR14"},
        {"구분": "5일 수익률", "값": f"{energy_level['momentum_return_5d'] * 100:+.2f}%", "산출 의미": "모멘텀 저항 계수의 원자료"},
        {"구분": "거래량 데이터", "값": "실제 거래량" if energy_level["volume_available"] else "거래량 없음 · 1.00x 대체", "산출 의미": "거래량 배수의 신뢰 상태"},
    ]).set_index("구분")
    st.dataframe(energy_table, width="stretch")

    if energy_backtest["resistance"] and energy_backtest["support"]:
        resistance_result = energy_backtest["resistance"]
        support_result = energy_backtest["support"]
        validation_table = pd.DataFrame([
            {
                "검증 대상": "동적 저항선",
                "검증 표본": f"{resistance_result['touch_count']}회 터치",
                "터치 후 거부 확률": f"{resistance_result['hold_probability']:.1f}%",
                "95% 신뢰구간": f"{resistance_result['ci_lower']:.1f}% ~ {resistance_result['ci_upper']:.1f}%",
            },
            {
                "검증 대상": "동적 지지선",
                "검증 표본": f"{support_result['touch_count']}회 터치",
                "터치 후 방어 확률": f"{support_result['hold_probability']:.1f}%",
                "95% 신뢰구간": f"{support_result['ci_lower']:.1f}% ~ {support_result['ci_upper']:.1f}%",
            },
        ]).set_index("검증 대상")
        st.write(
            f"**과거 {energy_backtest['horizon']}영업일 검증:** 전체 이벤트 "
            f"`{energy_backtest['sample_count']}건` · 각 레벨에 실제 도달한 사례만 조건부 집계"
        )
        st.dataframe(validation_table, width="stretch")
        st.caption(
            "저항 거부는 상단을 터치한 뒤 검증 기간 마지막 종가가 상단 아래인 경우, "
            "지지 방어는 하단을 터치한 뒤 마지막 종가가 하단 위인 경우입니다. "
            "신뢰구간이 넓거나 터치 표본이 적으면 해당 레벨의 신뢰도를 낮게 해석하세요."
        )
    else:
        st.info("과거 동적 상·하단 터치 표본이 부족해 신뢰도 검증을 표시할 수 없습니다.")

# ---------------- Standard deviation bands (1σ~6σ) ----------------
st.divider()
st.subheader("📐 표준편차 밴드 (1σ ~ 6σ)")
st.caption("MA20 기준 ± k × 20일 표준편차. k값이 클수록 통계적으로 드문(극단적인) 구간입니다.")

std_ma, std_val, bands = std_bands(kospi, ma_n=20, std_n=20, max_k=6)

band_rows = []
for k in range(1, 7):
    support_val = bands[k]["support"]
    resistance_val = bands[k]["resistance"]
    band_rows.append({
        "σ 배수": f"{k}σ",
        "지지선 (하단)": f"{support_val:,.2f}",
        "지지선 신뢰도": band_reliability_tags(k, support_val, vol_ratio_val, vola_ratio_val, weekly_ma_val),
        "저항선 (상단)": f"{resistance_val:,.2f}",
        "저항선 신뢰도": band_reliability_tags(k, resistance_val, vol_ratio_val, vola_ratio_val, weekly_ma_val),
    })

st.dataframe(
    pd.DataFrame(band_rows).set_index("σ 배수"),
    width="stretch",
)
st.caption(
    f"기준 MA20: `{std_ma:,.2f}` · 20일 표준편차: `{std_val:,.2f}` · "
    f"주봉 MA20: `{weekly_ma_val:,.2f}`" if not np.isnan(weekly_ma_val)
    else f"기준 MA20: `{std_ma:,.2f}` · 20일 표준편차: `{std_val:,.2f}` · 주봉 MA20: 데이터 부족"
)

# ---------------- Taylor series (local polynomial) curvature analysis ----------------
st.divider()
st.subheader("📈 테일러 급수 기반 곡률 분석 (실험적)")
st.caption(
    "⚠️ 실제 주가는 매끄러운(무한 미분 가능한) 함수가 아니므로, 이 섹션은 "
    "엄밀한 테일러 급수가 아니라 '최근 20일 다항식 국소 피팅' 기반 근사입니다. "
    "예측값이 아니라 현재 추세의 휘어짐(모멘텀·가속도)을 보조적으로 참고하는 용도로만 사용하세요."
)

taylor_window = 20
taylor_degree = 3
tfit = taylor_fit(kospi, window=taylor_window, degree=taylor_degree)

if tfit is None:
    st.info(f"다항식 피팅에 필요한 최근 {taylor_window}일 데이터가 부족합니다.")
else:
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("f(a) 기준값", f"{tfit['f_a']:,.2f}")
    t2.metric("f'(a) 1차미분(모멘텀)", f"{tfit['f1_a']:+.2f} / 일")
    t3.metric("f''(a) 2차미분(곡률/가속도)", f"{tfit['f2_a']:+.3f}")
    t4.metric("f'''(a) 3차미분(곡률 변화율)", f"{tfit['f3_a']:+.4f}")

    if tfit["f2_a"] > 0:
        st.success("곡률 양(+): 하락 속도 둔화 또는 상승 가속 국면 (바닥권/상승 전환 신호 가능)")
    elif tfit["f2_a"] < 0:
        st.warning("곡률 음(-): 상승 속도 둔화 또는 하락 가속 국면 (고점권/하락 전환 신호 가능)")
    else:
        st.info("곡률 거의 없음: 선형적인 추세 구간")

    st.write("**단기 경로 근사 (h영업일 뒤, 테일러 전개 기반)**")
    proj_rows = taylor_projection(tfit, horizons=(1, 2, 3, 4, 5))
    proj_table = pd.DataFrame([
        {
            "h (영업일 뒤)": r["h"],
            "근사 예상값": f"{r['proj']:,.2f}",
            "오차밴드 상단": f"{r['upper']:,.2f}",
            "오차밴드 하단": f"{r['lower']:,.2f}",
        }
        for r in proj_rows
    ]).set_index("h (영업일 뒤)")
    st.dataframe(proj_table, width="stretch")
    st.caption(
        "오차밴드는 최근 20일 피팅 잔차(RMSE)를 기준으로 h가 커질수록(기준점 a에서 "
        "멀어질수록) 넓어지도록 근사한 값입니다. h=1에 가까울수록 신뢰도가 높습니다."
    )

    tp = turning_point_signal(kospi, short_window=10, long_window=taylor_window, degree=2)
    if tp is not None:
        if tp["sign_flip"]:
            st.error(
                f"⚠️ 변곡점 신호: 단기(10일) 곡률({tp['short_curv']:+.3f})과 "
                f"장기(20일) 곡률({tp['long_curv']:+.3f})의 부호가 반대입니다. "
                "추세 전환이 임박했을 가능성 — 위 σ 밴드 터치 시 특히 주의 깊게 확인하세요."
            )
        else:
            st.success("단기/장기 곡률 방향 일치 — 추세 전환 신호 없음")

# ---------------- GJR-GARCH conditional volatility ----------------
st.divider()
st.subheader("📉 GJR-GARCH(1,1,1) 조건부 변동성 모형")

if volatility_model_source == "EWMA fallback":
    st.info(
        "GJR-GARCH를 사용할 수 없어 EWMA(λ=0.94) 변동성으로 대체했습니다. "
        f"현재 일간 변동성은 {garch_today_vol_pct:.3f}%이며 Bellman과 밴드 계산에 사용됩니다."
    )
elif not ARCH_AVAILABLE:
    st.warning(
        "`arch` 패키지가 설치되어 있지 않습니다. 터미널에서 "
        "`pip install arch` 실행 후 앱을 다시 시작하세요."
    )
elif garch_res is None:
    st.info("GJR-GARCH 적합에 필요한 데이터(최소 100일 이상의 일간수익률)가 부족합니다.")
else:
    params = garch_res.params
    omega = float(params.get("omega", np.nan))
    alpha1 = float(params.get("alpha[1]", np.nan))
    gamma1 = float(params.get("gamma[1]", np.nan))
    beta1 = float(params.get("beta[1]", np.nan))

    g1, g2, g3 = st.columns(3)
    g1.metric("오늘 조건부 일간변동성", f"{garch_today_vol_pct:.3f}%")
    g2.metric("연율화 변동성(참고)", f"{garch_today_vol_pct * np.sqrt(252):.2f}%")
    g3.metric("비대칭계수 γ (레버리지 효과)", f"{gamma1:+.4f}" if not np.isnan(gamma1) else "N/A")

    if not np.isnan(gamma1) and gamma1 > 0:
        st.error(
            f"γ = {gamma1:+.4f} > 0 → 레버리지 효과 확인: 음(-)의 충격(하락)이 "
            "양(+)의 충격(상승)보다 변동성을 더 크게 증폭시키는 구조입니다."
        )
    elif not np.isnan(gamma1):
        st.info(f"γ = {gamma1:+.4f} ≤ 0 → 이 구간에서는 뚜렷한 레버리지 효과가 관측되지 않습니다.")

    st.caption(
        f"ω(상수항)=`{omega:.4f}` · α(충격계수)=`{alpha1:.4f}` · "
        f"γ(비대칭계수)=`{gamma1:.4f}` · β(지속성계수)=`{beta1:.4f}` "
        "(모두 % 수익률 기준 GJR-GARCH(1,1,1) 적합 파라미터)"
    )

    st.write(f"**향후 {BELLMAN_HORIZON}영업일 조건부 변동성 예측 경로**")
    garch_rows = []
    for h, vol_pct in enumerate(garch_vol_path_pct, start=1):
        band_1s = spot * (vol_pct / 100.0)
        garch_rows.append({
            "h (영업일 뒤)": h,
            "예측 일간변동성(%)": f"{vol_pct:.3f}%",
            "±1σ 가격폭(참고)": f"±{band_1s:,.2f}",
        })
    st.dataframe(pd.DataFrame(garch_rows).set_index("h (영업일 뒤)"), width="stretch")
    st.caption(
        "이 GARCH 변동성은 아래 '확장형 분산' 및 '벨만 최적화' 모듈의 "
        "리스크(위험) 항으로 이어져 사용됩니다."
    )

    # ---------------- Extended variance + GARCH-M + Hybrid bands ----------------
    st.divider()
    st.subheader("🧬 확장형 GJR-GARCH + GARCH-M + 하이브리드 밴드")
    st.caption(
        "⚠️ 실제 미결제약정/옵션프리미엄 데이터는 무료로 구할 수 없어, "
        "거래량비율(δ1)과 현·선물 괴리 Z-score(δ2)를 파생 수급 소진의 대리변수로 사용합니다. "
        "또한 arch 패키지가 분산방정식에 외생변수를 직접 MLE로 넣는 것을 지원하지 않으므로, "
        "이미 적합된 GJR-GARCH 분산에 사후 보정을 가하는 근사 방식입니다."
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("h_t (원 GJR-GARCH 분산)", f"{h_t_raw:.4f} %²" if not np.isnan(h_t_raw) else "N/A")
    e2.metric("h_t,ext (외생변수 반영 후)", f"{h_t_ext:.4f} %²" if not np.isnan(h_t_ext) else "N/A")
    e3.metric("확장 배수", f"{ext_factor:.2f}x" if not np.isnan(ext_factor) else "N/A")

    if not np.isnan(ext_factor) and ext_factor > 1.3:
        st.warning("거래량 소진 또는 괴리 극단도가 높아 실질 리스크가 원 GARCH 추정치보다 크게 확대되었습니다.")

    st.write("**② GARCH-M 피드백 (변동성 → 기대수익 연동, 2단계 근사)**")
    gm1, gm2 = st.columns(2)
    gm1.metric("μ̂ (상수항, %)", f"{garch_m_mu:+.4f}%" if not np.isnan(garch_m_mu) else "N/A")
    gm2.metric("λ̂ (위험프리미엄 계수)", f"{garch_m_lambda:+.4f}" if not np.isnan(garch_m_lambda) else "N/A")
    if not np.isnan(garch_m_lambda):
        if garch_m_lambda > 0:
            st.success("λ̂ > 0: 변동성이 커질수록 반등 기대수익도 함께 커지는 구조(변동성-수익 양의 피드백)")
        else:
            st.info("λ̂ ≤ 0: 이 구간에서는 변동성 확대가 기대수익 상승으로 이어지지 않습니다.")

    st.write("**③ 최종 하이브리드 밴드**")
    daily_vol_ext_price = spot * (np.sqrt(h_t_ext) / 100.0) if not np.isnan(h_t_ext) else np.nan
    last_ret = float(kospi["Close"].pct_change().iloc[-1])
    neg_shock_indicator = 1 if last_ret < 0 else 0
    theta_call_price = spot * (theta_call_pct / 100.0)

    if not np.isnan(daily_vol_ext_price):
        hybrid_upper, hybrid_lower = hybrid_bands(
            ma_t=spot_levels["ma20"],
            daily_vol_ext_price=daily_vol_ext_price,
            k1=k1_upper,
            k2=k2_lower,
            gamma=gamma1,
            neg_shock_indicator=neg_shock_indicator,
            theta_call_price=theta_call_price,
        )
        hb1, hb2 = st.columns(2)
        hb1.metric("Upper Band (상단, 압축)", f"{hybrid_upper:,.2f}")
        hb2.metric("Lower Band (하단, 확장)", f"{hybrid_lower:,.2f}")
        st.caption(
            f"어제 충격 부호 I(ε<0) = {neg_shock_indicator} · γ(비대칭계수) = {gamma1:+.4f} · "
            f"θ_call = {theta_call_price:,.2f} · k1 = {k1_upper} · k2 = {k2_lower}. "
            "음의 충격이 있었던 날 다음에는 하단이 자동으로 더 넓게 열립니다."
        )
    else:
        st.info("확장분산을 계산할 수 없어 하이브리드 밴드를 표시할 수 없습니다.")

# ---------------- Dynamic volatility band forecast ----------------
st.divider()
st.subheader("🔭 비대칭 변동성·동적 밴드 전망")
st.caption(
    "오늘 종가를 기준으로 테일러 중심 경로와 GJR-GARCH 조건부 변동성 경로를 결합합니다. "
    "음(-)의 마지막 충격에는 γ를 하단 폭에 추가하고, 양(+)의 반등에는 해당 레버리지 확대를 적용하지 않습니다. "
    "계산 결과는 확정 예측이 아닌 다음 5영업일의 위험 범위 참고값입니다."
)

forecast_gamma = np.nan
if garch_res is not None:
    forecast_gamma = float(garch_res.params.get("gamma[1]", np.nan))

forecast_taylor_rows = (
    taylor_projection(tfit, horizons=tuple(range(1, BELLMAN_HORIZON + 1)))
    if tfit is not None else []
)
forecast_last_return = float(kospi["Close"].pct_change().iloc[-1])
taylor_error_multiplier = 1.0
if tfit is not None and not np.isnan(spot_levels["atr"]) and spot_levels["atr"] > 0:
    taylor_error_multiplier = float(np.clip(tfit["rmse"] / spot_levels["atr"], 1.0, 3.0))
dynamic_rows = dynamic_band_forecast(
    spot=spot,
    taylor_rows=forecast_taylor_rows,
    volatility_path_pct=garch_vol_path_pct,
    atr_value=spot_levels["atr"],
    gamma=forecast_gamma,
    last_return=forecast_last_return,
    taylor_error_multiplier=taylor_error_multiplier,
)

if not dynamic_rows:
    st.info("GJR-GARCH 변동성 경로가 없어 동적 밴드 전망을 계산할 수 없습니다.")
else:
    shock_label = "하락 충격 · γ 하단 확대" if forecast_last_return < 0 else "상승/반등 · γ 하단 확대 없음"
    d1, d2, d3 = st.columns(3)
    d1.metric("최근 일간 수익률", f"{forecast_last_return * 100:+.2f}%")
    d2.metric("비대칭계수 γ", f"{forecast_gamma:+.4f}" if not np.isnan(forecast_gamma) else "N/A")
    d3.metric("충격 상태", shock_label)
    st.caption(
        f"변동성 모델: {volatility_model_source} · Taylor RMSE/ATR 오차 확대 배수: "
        f"{taylor_error_multiplier:.2f}x"
    )

    dynamic_table = pd.DataFrame([
        {
            "시점": f"t+{row['h']}일",
            "중심값(테일러)": f"{row['center']:,.2f}",
            "조건부 변동성": f"{row['volatility_pct']:.3f}%",
            "동적 상단": f"{row['upper']:,.2f}",
            "동적 하단": f"{row['lower']:,.2f}",
            "오차폭(RMSE)": f"±{row['taylor_error']:,.2f}",
        }
        for row in dynamic_rows
    ]).set_index("시점")
    st.dataframe(dynamic_table, width="stretch")
    st.caption(
        f"폭 계산 기준: GARCH 일간 변동성 가격폭과 ATR14({spot_levels['atr']:,.2f}) 중 큰 값 + "
        "테일러 피팅 RMSE 오차폭. 현재 20일 표준편차 밴드는 아래 표에서 별도로 확인합니다."
    )

# ---------------- Bellman optimal multi-step position path ----------------
st.divider()
st.subheader("🧮 벨만 최적화 기반 다단계 포지션 계획 (실험적)")
st.caption(
    "벨만의 최적성 원리(동적계획법)로 향후 "
    f"{BELLMAN_HORIZON}영업일에 걸친 최적 포지션 경로를 계산합니다. "
    "기대수익은 방향점수 + GARCH-M 피드백에서, 리스크는 확장형 GJR-GARCH "
    "변동성(거래량 소진·괴리 극단도 반영)에서 가져오며, "
    "포지션을 바꿀 때마다 리밸런싱 비용을 반영합니다. "
    "⚠️ 실제 확률분포가 아닌 단순화된 추정이므로 참고용 계획이지 확정 예측이 아닙니다."
)

# 기대수익률(하루, 소수) 추정: 방향점수 기반 + GARCH-M(변동성→기대수익 피드백) 항 결합
daily_ret_std = float(kospi["Close"].pct_change().dropna().std())
mu_daily_score = (composite_score / 100.0) * daily_ret_std

if not np.isnan(garch_m_lambda) and not np.isnan(h_t_ext):
    mu_daily_garch_m = (garch_m_lambda * np.sqrt(h_t_ext)) / 100.0  # % → 소수
else:
    mu_daily_garch_m = 0.0

mu_daily = mu_daily_score + mu_daily_garch_m
effective_rebal_cost = rebal_cost * (1.5 if abs(composite_score) < 30 else 1.0)

# 리스크(일간 변동성, 소수) 경로: 확장형 GJR-GARCH 분산을 우선 사용, 없으면 표준편차로 대체
if garch_res is not None and garch_vol_path_pct is not None and not np.isnan(ext_factor):
    sigma_daily_path = (garch_vol_path_pct * np.sqrt(ext_factor)) / 100.0
elif garch_res is not None and garch_vol_path_pct is not None:
    sigma_daily_path = (garch_vol_path_pct / 100.0)
else:
    fallback_sigma = std_val / spot if spot else daily_ret_std
    sigma_daily_path = np.full(BELLMAN_HORIZON, fallback_sigma)

v0, optimal_path = bellman_optimal_path(
    current_position=level_final,
    mu_daily=mu_daily,
    sigma_daily_path=sigma_daily_path,
    states=POSITION_STATES,
    risk_aversion=risk_aversion,
    rebal_cost=effective_rebal_cost,
)

predicted_return_pct = mu_daily * 100
predicted_direction = (
    "상승" if predicted_return_pct > 0 else
    "하락" if predicted_return_pct < 0 else "보합"
)
prediction_record = {
    "prediction_date": str(kospi.index[-1].date()),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "data_source": kospi_source,
    "spot_close": float(spot),
    "direction_score": float(composite_score),
    "confidence": float(composite_conf),
    "basis_z": None if np.isnan(basis_z) else float(basis_z),
    "basis_discount": float(basis_discount),
    "volatility_model": volatility_model_source,
    "taylor_error_multiplier": float(taylor_error_multiplier),
    "regime": regime,
    "predicted_next_close": float(spot * (1.0 + mu_daily)),
    "predicted_return_pct": float(predicted_return_pct),
    "predicted_direction": predicted_direction,
    "current_position": int(level_final),
    "next_position": int(optimal_path[1]),
    "position_path": [int(position) for position in optimal_path],
    "rebalancing_cost_used": float(effective_rebal_cost),
    "volume_ratio": None if np.isnan(vol_ratio_val) else float(vol_ratio_val),
    "volatility_ratio": None if np.isnan(vola_ratio_val) else float(vola_ratio_val),
    "energy_dynamic_resistance": None if energy_level is None else float(energy_level["dynamic_resistance"]),
    "energy_dynamic_support": None if energy_level is None else float(energy_level["dynamic_support"]),
    "energy_width": None if energy_level is None else float(energy_level["energy_width"]),
    "energy_backtest_samples": int(energy_backtest["sample_count"]),
    "energy_resistance_hold_probability": (
        None if not energy_backtest["resistance"] else float(energy_backtest["resistance"]["hold_probability"])
    ),
    "energy_support_hold_probability": (
        None if not energy_backtest["support"] else float(energy_backtest["support"]["hold_probability"])
    ),
    "price_volume_energy_percentile": (
        None if price_volume_energy is None else float(price_volume_energy["percentile"])
    ),
    "obv_signal": None if obv_signal is None else obv_signal["signal"],
    "obv_bullish_divergence": (
        None if obv_signal is None else bool(obv_signal["bullish_divergence"])
    ),
    "volatility_volume_energy_percentile": (
        None if volatility_volume is None else float(volatility_volume["percentile"])
    ),
    "actual_high": float(kospi["High"].iloc[-1]),
    "actual_low": float(kospi["Low"].iloc[-1]),
    "krw_5d_pct": None if np.isnan(krw_5d_pct) else float(krw_5d_pct),
    "vix": None if np.isnan(vix_last) else float(vix_last),
    "macro_risk": bool(macro_risk),
}
prediction_comparison = log_prediction_and_compare(prediction_record)

st.divider()
st.subheader("📋 어제 예측 vs 오늘 실제")
if prediction_comparison is None:
    st.info("비교할 이전 거래일 예측 로그가 없습니다. 오늘 예측을 저장했으며 다음 실행부터 비교합니다.")
else:
    comparison = prediction_comparison
    result_text = "적중" if comparison["direction_hit"] else "불일치"
    comparison_columns = st.columns(4)
    comparison_columns[0].metric(
        "예측 종가",
        f"{comparison['predicted_close']:,.2f}",
        f"{comparison['predicted_return_pct']:+.2f}%",
    )
    comparison_columns[1].metric(
        "오늘 실제 종가",
        f"{comparison['actual_close']:,.2f}",
        f"{comparison['actual_return_pct']:+.2f}%",
    )
    comparison_columns[2].metric("종가 오차", f"{comparison['close_error_pct']:+.2f}%")
    comparison_columns[3].metric(
        "방향 적중",
        result_text,
        f"예측 {comparison['predicted_direction']} / 실제 {comparison['actual_direction']}",
    )
    st.caption(
        f"예측 기준일 {comparison['prediction_date']} → 실제 기준일 {comparison['actual_date']} · "
        f"로그 파일: {PREDICTION_LOG_PATH}"
    )
    band_columns = st.columns(3)
    band_columns[0].metric(
        "실제 고가",
        f"{comparison['actual_high']:,.2f}" if comparison.get("actual_high") is not None else "N/A",
    )
    band_columns[1].metric(
        "동적 저항 도달",
        "도달" if comparison.get("resistance_touched") else "미도달"
        if comparison.get("resistance_touched") is not None else "N/A",
    )
    band_columns[2].metric(
        "동적 지지 이탈",
        "이탈" if comparison.get("support_broken") else "방어"
        if comparison.get("support_broken") is not None else "N/A",
    )

st.divider()
st.subheader("📈 신뢰도 추이 및 사후 검증")
st.caption(
    "예측 시점의 신뢰도와 이후 실제 방향을 거래일별로 누적합니다. "
    "기존 로그에 신뢰도가 없으면 빈 값으로 표시되며, 새 실행부터 기록됩니다."
)
history_rows = prediction_history_rows(_read_prediction_log(PREDICTION_LOG_PATH))
if not history_rows:
    st.info("아직 저장된 예측 이력이 없습니다.")
else:
    history_df = pd.DataFrame(history_rows).set_index("예측일")
    confirmed = history_df[history_df["방향 적중"].isin(["적중", "불일치"])]
    confidence_values = pd.to_numeric(history_df["신뢰도(%)"], errors="coerce").dropna()
    h1, h2, h3 = st.columns(3)
    h1.metric("누적 예측 수", f"{len(history_df)}건")
    h2.metric(
        "확인된 방향 적중률",
        f"{(confirmed['방향 적중'] == '적중').mean() * 100:.1f}%"
        if not confirmed.empty else "N/A",
    )
    h3.metric(
        "평균 신뢰도",
        f"{confidence_values.mean():.1f}%"
        if not confidence_values.empty else "N/A",
    )
    st.dataframe(history_df, width="stretch")

records = _read_prediction_log(PREDICTION_LOG_PATH)
proposal = correction_proposal(records)
st.subheader("🛠️ 사후 보정 제안 (반자동 승인)")
st.caption("평가 결과를 바탕으로 다음 실행의 밴드 버퍼와 리스크회피계수 변경안을 제안합니다. 자동 주문이나 자동 적용은 하지 않습니다.")
if proposal is None:
    st.info("평가 완료 기록이 없어 보정안을 만들 수 없습니다. 최소 3건의 장 마감 후 평가가 필요합니다.")
else:
    proposal_columns = st.columns(4)
    proposal_columns[0].metric("최근 평가 표본", f"{proposal['sample_count']}건")
    proposal_columns[1].metric("방향 적중률", f"{proposal['direction_hit_rate']:.1f}%")
    proposal_columns[2].metric("평균 종가 오차", f"{proposal['mean_abs_close_error_pct']:.2f}%")
    proposal_columns[3].metric("지지선 이탈", f"{proposal['support_break_count']}건")
    st.write(f"**제안 근거:** {proposal['reason']}")
    changes = proposal["changes"]
    if not changes:
        st.info("현재 표본 기준으로 변경 제안이 없습니다.")
    else:
        next_buffer = changes["energy_buffer_multiplier"]
        next_risk_delta = changes["risk_aversion_delta"]
        st.write(
            f"제안값: 에너지 밴드 버퍼 `{next_buffer:.2f}x`, "
            f"리스크회피계수 보정 `{next_risk_delta:+.2f}`"
        )
        if st.button("제안 승인 및 다음 실행부터 적용", type="primary"):
            write_adjustments(
                MODEL_ADJUSTMENT_PATH,
                {
                    "energy_buffer_multiplier": next_buffer,
                    "risk_aversion_delta": next_risk_delta,
                    "approved_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            st.success("보정안을 저장했습니다. 다음 Streamlit 실행부터 적용됩니다.")

bp1, bp2 = st.columns(2)
bp1.metric("현재 포지션 (t=0)", f"{level_final}%")
bp2.metric(
    f"{BELLMAN_HORIZON}일 뒤 권장 포지션",
    f"{optimal_path[-1]}%",
    f"{optimal_path[-1] - level_final:+d}%p"
)

path_table = pd.DataFrame({
    "시점": ["오늘(t=0)"] + [f"t+{i}일" for i in range(1, len(optimal_path))],
    "권장 포지션(%)": optimal_path,
})
st.dataframe(path_table.set_index("시점"), width="stretch")

if optimal_path[1] != level_final:
    direction_word = "확대" if optimal_path[1] > level_final else "축소"
    st.warning(
        f"다음 거래일 최적 행동: 포지션을 {level_final}% → {optimal_path[1]}%로 "
        f"{direction_word}하는 것이 (현재 가정 하에) 기대효용을 극대화합니다."
    )
else:
    st.success("다음 거래일 최적 행동: 현재 포지션 유지")

with st.expander("벨만 모형 가정 상세"):
    st.write(f"- 방향점수 기반 기대수익: `{mu_daily_score * 100:+.4f}%` (방향점수 {composite_score:+.1f} 반영)")
    st.write(f"- GARCH-M 피드백 기대수익: `{mu_daily_garch_m * 100:+.4f}%` (λ̂={garch_m_lambda:+.4f})" if not np.isnan(garch_m_lambda) else "- GARCH-M 피드백: 계산 불가")
    st.write(f"- 합산 기대수익(mu): `{mu_daily * 100:+.4f}%`")
    st.write("- 헤지/인버스 포지션 수익은 부호가 반대이므로, 하락신호(mu<0)일수록 헤지 비중 확대가 유리하게 계산됩니다.")
    st.write(f"- 리스크회피계수: `{risk_aversion}` · 리밸런싱 비용: `{rebal_cost * 100:.1f}%`")
    st.write(f"- 상태공간: `{POSITION_STATES}` · 계획 기간: `{BELLMAN_HORIZON}`영업일")
    st.write(f"- t=0 가치함수 V(현재 포지션 {level_final}%): `{v0:.6f}`")

# ---------------- Chart ----------------
chart_df = kospi[["Close"]].copy()
chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
chart_df["MA60"] = chart_df["Close"].rolling(60).mean()
chart_df = chart_df.tail(180)

st.line_chart(chart_df)

# ---------------- Detailed data ----------------
with st.expander("🔎 상세 분석"):
    rows = {
        "현물 방향 점수": spot_score,
        "현물 신뢰도": spot_conf,
        "선물 방향 점수": fut_score,
        "선물 신뢰도": fut_conf,
        "Basis %": basis,
        "Basis Z": basis_z,
        "권장 헤지/인버스(보정 전)": level,
        "권장 헤지/인버스(매크로 보정 후)": level_final,
        "거래량 비율": vol_ratio_val,
        "변동성 비율 (5일/60일)": vola_ratio_val,
        "주봉 MA20": weekly_ma_val,
        "USD/KRW 5일 변화율 %": krw_5d_pct,
        "VIX": vix_last,
        "매크로 위험 신호": macro_risk,
    }
    if tfit is not None:
        rows.update({
            "Taylor f'(a) 모멘텀": tfit["f1_a"],
            "Taylor f''(a) 곡률": tfit["f2_a"],
            "Taylor f'''(a) 곡률변화율": tfit["f3_a"],
            "Taylor 피팅 RMSE": tfit["rmse"],
        })
    if garch_res is not None:
        rows.update({
            "GARCH 오늘 조건부변동성(%)": garch_today_vol_pct,
            "GARCH 비대칭계수 γ": float(garch_res.params.get("gamma[1]", np.nan)),
            "h_t (원 분산)": h_t_raw,
            "h_t,ext (확장 분산)": h_t_ext,
            "확장 배수": ext_factor,
            "GARCH-M λ̂": garch_m_lambda,
            "GARCH-M μ̂(%)": garch_m_mu,
        })
    rows.update({
        "벨만 t=0 가치함수": v0,
        f"벨만 {BELLMAN_HORIZON}일뒤 권장 포지션": optimal_path[-1],
    })
    st.dataframe(
        pd.DataFrame(rows, index=["값"]).T,
        width="stretch",
    )

st.divider()
st.caption(
    "⚠️ 본 프로그램은 투자 판단 보조용입니다. "
    "수익을 보장하지 않으며 실시간 주문/매매 기능은 포함하지 않습니다."
)
