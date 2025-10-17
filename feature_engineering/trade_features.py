import pandas as pd
import numpy as np
from typing import Dict

# ==============================================================
# 🔧 Trade-based Feature Functions
# ==============================================================

def calc_trade_intensity(df: pd.DataFrame, window_s: float = 1.0) -> float:
    """
    거래 빈도(초당 체결 횟수)
    """
    if df.empty:
        return 0.0
    duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    if duration == 0:
        return 0.0
    return len(df) / duration


def calc_buy_sell_ratio(df: pd.DataFrame) -> float:
    """
    체결강도 (buy volume / total volume)
    """
    if df.empty:
        return 0.5
    buy_vol = df.loc[df["side"] == "buy", "qty"].sum()
    sell_vol = df.loc[df["side"] == "sell", "qty"].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.5
    return buy_vol / total


def calc_volume_delta(df: pd.DataFrame) -> float:
    """
    Volume Delta = buy volume - sell volume
    """
    if df.empty:
        return 0.0
    buy_vol = df.loc[df["side"] == "buy", "qty"].sum()
    sell_vol = df.loc[df["side"] == "sell", "qty"].sum()
    return buy_vol - sell_vol


def calc_vwap(df: pd.DataFrame) -> float:
    """
    VWAP (체결가중 평균가격)
    """
    if df.empty:
        return np.nan
    return np.sum(df["price"] * df["qty"]) / np.sum(df["qty"])


def calc_volume_spike(df: pd.DataFrame, prev_avg_vol: float, factor: float = 2.0) -> float:
    """
    거래량 급증 여부 (이전 평균 대비 몇 배인지)
    """
    cur_vol = df["qty"].sum()
    if prev_avg_vol == 0:
        return 1.0
    ratio = cur_vol / prev_avg_vol
    return ratio if ratio > factor else 1.0


def calc_trade_pressure(df: pd.DataFrame) -> float:
    """
    매수 vs 매도 체결 강도의 상대적 힘
    - 값 > 0 → 매수 우위
    - 값 < 0 → 매도 우위
    """
    if df.empty:
        return 0.0
    buy_vol = df.loc[df["side"] == "buy", "qty"].sum()
    sell_vol = df.loc[df["side"] == "sell", "qty"].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return (buy_vol - sell_vol) / total


# ==============================================================
# 🧠 메인 Feature Extractor
# ==============================================================

def extract_trade_features(trades_df: pd.DataFrame, prev_avg_vol: float = 0.0) -> Dict:
    """
    단일 구간(예: 1초) 내 체결 데이터에서 피처 추출

    | Feature           | 설명                 | 의미         |
    | ----------------- | ------------------ | ---------- |
    | `trade_count`     | 구간 내 체결 수          | 시장 활동도     |
    | `trade_intensity` | 초당 거래 횟수           | 체결 속도      |
    | `buy_sell_ratio`  | 매수 체결량 / 전체 체결량    | 매수세 강도     |
    | `volume_delta`    | 매수 체결량 - 매도 체결량    | 매수/매도 순압력  |
    | `vwap`            | 거래량 가중평균가          | 공정가 수준     |
    | `trade_pressure`  | (매수-매도)/(총체결량)     | 순간 체결세력 지표 |
    | `volume_spike`    | 이전 평균 대비 거래량 폭증 비율 | 이벤트성 거래 감지 |

    """
    feats = {
        "trade_count": len(trades_df),
        "trade_intensity": calc_trade_intensity(trades_df),
        "buy_sell_ratio": calc_buy_sell_ratio(trades_df),
        "volume_delta": calc_volume_delta(trades_df),
        "vwap": calc_vwap(trades_df),
        "trade_pressure": calc_trade_pressure(trades_df),
        "volume_spike": calc_volume_spike(trades_df, prev_avg_vol)
    }
    return feats


# ==============================================================
# 🔬 테스트용 메인
# ==============================================================

if __name__ == "__main__":
    import datetime

    # 테스트용 가짜 데이터 생성
    sample_data = {
        "timestamp": pd.to_datetime([
            "2025-10-17 09:00:00.100",
            "2025-10-17 09:00:00.300",
            "2025-10-17 09:00:00.700",
            "2025-10-17 09:00:00.800"
        ]),
        "price": [54000.0, 54000.2, 54000.5, 54000.4],
        "qty": [0.3, 0.5, 0.7, 0.4],
        "side": ["buy", "buy", "sell", "sell"]
    }

    df = pd.DataFrame(sample_data)
    feats = extract_trade_features(df, prev_avg_vol=1.0)

    print("🧩 Trade Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")
