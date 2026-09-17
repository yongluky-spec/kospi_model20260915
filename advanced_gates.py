"""
advanced_gates.py
------------------
kospi_engine.py (신버전)에서 사용하는 4가지 신규 규칙/유틸리티.

옛날 버전(kospi_model.py)과 구분하기 위해 새 파일로 분리했다.
전부 실제 데이터로 계산되며, 검증 불가능한 서술(예: "청산 세탁기",
"유동성 훅")이나 근거 없는 정밀 계수(예: "델타 12.0배")는 사용하지 않는다.

포함된 4가지:
1. leader_divergence_check   - 대장주(삼성전자·SK하이닉스) 가격 괴리 킬스위치
2. futures_collision_gate    - ES=F/NQ=F 선물-신호 충돌 게이트
3. intraday_delta_acceleration - 1시간 델타·가속도 모니터
4. build_session_report      - 위 계산 결과를 한 장으로 요약하는 리포트 렌더러
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1) 대장주 가격 괴리 킬스위치
# ----------------------------------------------------------------------
LEADER_TICKERS_DEFAULT = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
}


def leader_divergence_check(kospi_return_pct, leader_returns_pct, threshold_pct=-0.3):
    """
    "지수는 오르는데 대장주는 빠진다"는 가짜 반등 신호를 실제 가격으로 확인한다.

    ⚠️ 원래 아이디어였던 '외국인/기관 순매수 금액(KRW)'은 무료로 실시간
    제공되지 않아 그대로는 구현할 수 없다. 대신 실제로 받아올 수 있는
    대장주 종가 등락률(가격 기반)로 같은 목적을 다른 방식으로 구현했다.
    "수급이 빠졌다"가 아니라 "대장주 가격이 지수와 반대로 갔다"로 읽어야 한다.

    kospi_return_pct: 코스피 당일 수익률(%)
    leader_returns_pct: {"삼성전자": 1.2, "SK하이닉스": -0.5, ...} 형태의 dict
    threshold_pct: 대장주 평균 수익률이 이 값보다 낮으면 이탈로 판정
    """
    valid = {k: v for k, v in leader_returns_pct.items() if v is not None and not np.isnan(v)}
    if not valid or np.isnan(kospi_return_pct):
        return {
            "available": False,
            "leader_avg_return_pct": np.nan,
            "divergence": False,
            "detail": "대장주 가격 데이터 부족",
        }

    leader_avg = float(np.mean(list(valid.values())))
    divergence = kospi_return_pct > 0.1 and leader_avg < threshold_pct

    return {
        "available": True,
        "leader_avg_return_pct": leader_avg,
        "kospi_return_pct": float(kospi_return_pct),
        "divergence": bool(divergence),
        "per_ticker": valid,
        "detail": (
            f"코스피 {kospi_return_pct:+.2f}% vs 대장주 평균 {leader_avg:+.2f}% "
            f"({'이탈' if divergence else '동조'})"
        ),
    }


# ----------------------------------------------------------------------
# 2) ES=F/NQ=F 선물 충돌 게이트
# ----------------------------------------------------------------------
def futures_collision_gate(composite_score, futures_pct, threshold_pct=0.5):
    """
    선물(ES=F 또는 NQ=F) 변동률이 현재 방향점수와 반대로 threshold_pct(%) 이상
    움직이면 '충돌'로 보고, 기존 매크로 게이트와 동일한 '절반 축소' 규칙을
    그대로 적용한다. 임의의 민감도 계수(예: ×12.0)는 쓰지 않는다.
    """
    if np.isnan(futures_pct) or np.isnan(composite_score) or composite_score == 0:
        return {"conflict": False, "multiplier": 1.0, "detail": "데이터 부족 또는 방향점수 0"}

    score_dir = 1 if composite_score > 0 else -1
    fut_dir = 1 if futures_pct > 0 else (-1 if futures_pct < 0 else 0)
    conflict = abs(futures_pct) >= threshold_pct and fut_dir != 0 and fut_dir != score_dir

    return {
        "conflict": bool(conflict),
        "multiplier": 0.5 if conflict else 1.0,
        "futures_pct": float(futures_pct),
        "detail": (
            f"방향점수 {composite_score:+.1f} vs 선물 {futures_pct:+.2f}% "
            f"({'충돌' if conflict else '충돌 없음'})"
        ),
    }


# ----------------------------------------------------------------------
# 3) 1시간 델타·가속도 모니터
# ----------------------------------------------------------------------
def intraday_delta_acceleration(hourly_df):
    """
    1시간봉 OHLC 데이터(hourly_df, 'Close' 컬럼 필수)로 시간당 변동률(델타)과
    그 변화량(가속도 = 델타의 1차 차분)을 계산한다.

    장중 내내(항상) 롤링으로 갱신되는 값이며, 오버나이트 갭(전일 마감→당일
    첫 봉)은 실제 장중 움직임이 아니므로 델타·가속도 계산에서 제외한다.
    "청산 세탁기" 같은 해석은 검증할 방법이 없어 포함하지 않고, 계산된
    수치만 반환한다.
    """
    if hourly_df is None or hourly_df.empty or "Close" not in hourly_df.columns:
        return None

    df = hourly_df.copy()
    df["date_only"] = df.index.date
    df["delta_pct"] = df["Close"].pct_change() * 100
    df["same_day"] = df["date_only"] == df["date_only"].shift(1)
    df.loc[~df["same_day"], "delta_pct"] = np.nan  # 오버나이트 갭 제외

    df["accel"] = df["delta_pct"].diff()
    same_day_shifted = np.concatenate(([False], df["same_day"].to_numpy()[:-1]))
    df.loc[~df["same_day"], "accel"] = np.nan
    df.loc[df["same_day"].to_numpy() & ~same_day_shifted, "accel"] = np.nan

    return df


# ----------------------------------------------------------------------
# 4) 세션 리포트 렌더러 (새 판단 로직 아님 — 기존 결과를 한 장으로 요약)
# ----------------------------------------------------------------------
def build_session_report(
    spot_price,
    composite_score,
    confidence,
    level_final,
    leader_result,
    futures_result,
    intraday_latest,
    macro_risk,
):
    """
    위 3가지 게이트 + 기존 모델 결과를 사람이 읽기 쉬운 텍스트 블록 하나로
    합친다. 이 함수는 새로운 판단을 만들지 않고, 이미 계산된 값을 요약해서
    보여주는 '출력 계층'일 뿐이다.
    """
    lines = []
    lines.append(f"**현재가**: {spot_price:,.2f}")
    lines.append(f"**방향점수/신뢰도**: {composite_score:+.1f} / {confidence:.0f}%")
    lines.append(f"**권장 헤지/인버스 레벨**: {level_final}%")

    if leader_result and leader_result.get("available"):
        flag = "⚠️ 대장주 이탈" if leader_result["divergence"] else "✅ 대장주 동조"
        lines.append(f"**대장주 확인**: {flag} — {leader_result['detail']}")
    else:
        lines.append("**대장주 확인**: 데이터 부족")

    if futures_result:
        flag = "⚠️ 선물 충돌" if futures_result["conflict"] else "✅ 선물 충돌 없음"
        lines.append(f"**선물 게이트**: {flag} — {futures_result['detail']}")

    if intraday_latest is not None:
        d = intraday_latest.get("delta_pct")
        a = intraday_latest.get("accel")
        d_txt = f"{d:+.3f}%" if d is not None and not np.isnan(d) else "N/A"
        a_txt = f"{a:+.3f}%p" if a is not None and not np.isnan(a) else "N/A"
        lines.append(f"**최근 1시간 델타/가속도**: {d_txt} / {a_txt}")

    lines.append(f"**매크로 위험 신호**: {'⚠️ 있음' if macro_risk else '✅ 없음'}")

    return "\n\n".join(lines)
