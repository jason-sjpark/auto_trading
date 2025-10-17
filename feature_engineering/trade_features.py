import pandas as pd
import numpy as np
from typing import Dict

# ==============================================================
# 📈 Trade Feature Engineering (체결 데이터 기반)
#  - 필수 컬럼: ['timestamp', 'price', 'qty', 'side']
#  - trade_intensity = 초당 체결 '횟수'(Trades Per Second)
# ==============================================================

def extract_trade_features(trades_df: pd.DataFrame, prev_avg_vol: float = 0.0) -> Dict[str, float]:
    """
    체결 데이터 구간(윈도우)에서 다음 피처를 계산:
      - trade_count       : 체결 횟수
      - trade_intensity   : 초당 체결 횟수 (Trades Per Second)  ✅
      - buy_sell_ratio    : 매수량 / (매수+매도)
      - volume_delta      : 매수량 - 매도량
      - vwap              : 체결가중 평균가
      - trade_pressure    : (매수량-매도량) / (매수+매도)
      - volume_spike      : 현재 체결량 / 직전 평균 체결량 (상한 10.0)

    prev_avg_vol:
      - 직전 구간의 총 체결량(선택). 0이면 현재 구간의 총 체결량으로 대체.
    """
    # 데이터 없음 처리
    if trades_df is None or len(trades_df) == 0:
        return {
            "trade_count": 0.0,
            "trade_intensity": 0.0,     # TPS
            "buy_sell_ratio": 0.5,
            "volume_delta": 0.0,
            "vwap": 0.0,
            "trade_pressure": 0.0,
            "volume_spike": 0.0,
        }

    df = trades_df.copy()

    # 필수 컬럼 보정
    for col in ["timestamp", "price", "qty", "side"]:
        if col not in df.columns:
            # 누락되면 안전 기본값
            if col == "timestamp":
                df[col] = pd.Timestamp.utcnow()
            elif col == "price" or col == "qty":
                df[col] = 0.0
            elif col == "side":
                df[col] = "buy"

    # 타입/결측 보정
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "qty"]).reset_index(drop=True)
    if len(df) == 0:
        return {
            "trade_count": 0.0,
            "trade_intensity": 0.0,
            "buy_sell_ratio": 0.5,
            "volume_delta": 0.0,
            "vwap": 0.0,
            "trade_pressure": 0.0,
            "volume_spike": 0.0,
        }

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce").fillna(0.0)
    df["side"]  = df["side"].astype(str).str.lower()

    # 기본 통계
    total_trades = int(len(df))
    total_volume = float(df["qty"].sum())
    df["amount"] = df["price"] * df["qty"]

    buy_vol  = float(df.loc[df["side"] == "buy",  "qty"].sum())
    sell_vol = float(df.loc[df["side"] == "sell", "qty"].sum())

    # ⭐ 초당 체결 횟수(TPS)
    # 구간 길이(초) = max(끝-시작, 아주 작은 값)
    duration_sec = float(
        max((df["timestamp"].max() - df["timestamp"].min()).total_seconds(), 1e-3)
    )
    trade_intensity = total_trades / duration_sec  # ✅ TPS (핵심 수정)

    # 피처 계산
    trade_count     = float(total_trades)
    buy_sell_ratio  = buy_vol / (buy_vol + sell_vol + 1e-9)
    volume_delta    = buy_vol - sell_vol
    vwap            = float(df["amount"].sum()) / (total_volume + 1e-9)
    trade_pressure  = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)

    # 거래량 급증(직전 대비) — prev_avg_vol=0이면 현재로 대체
    base_vol = total_volume if prev_avg_vol == 0 else prev_avg_vol
    volume_spike = (total_volume / (base_vol + 1e-9)) if base_vol > 0 else 1.0
    volume_spike = float(min(volume_spike, 10.0))  # 상한

    feats = {
        "trade_count": trade_count,
        "trade_intensity": float(trade_intensity),  # ✅ TPS
        "buy_sell_ratio": float(buy_sell_ratio),
        "volume_delta": float(volume_delta),
        "vwap": float(vwap),
        "trade_pressure": float(trade_pressure),
        "volume_spike": float(volume_spike),
    }

    # NaN → 0 보정
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0

    # --- 런타임 검증(경고 로그) : 비정상치 감지 ---
    # trade_intensity는 일반적으로 0~수백 TPS 범위
    if feats["trade_intensity"] < 0 or feats["trade_intensity"] > 1e5:
        print(f"[WARN] trade_intensity out of bounds: {feats['trade_intensity']:.3f} (duration={duration_sec:.4f}s, trades={total_trades})")

    return feats


# ==============================================================
# 🔬 단독 테스트
# ==============================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    ts = [now + timedelta(milliseconds=i*100) for i in range(30)]  # 3초 동안 30건 → 10 TPS 기대
    df = pd.DataFrame({
        "timestamp": ts,
        "price": np.linspace(100, 101, len(ts)),
        "qty":   np.random.rand(len(ts)) * 0.5,
        "side":  np.random.choice(["buy", "sell"], size=len(ts))
    })
    feats = extract_trade_features(df)
    print("✅ Trade Features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    # 기대: trade_count≈30, trade_intensity≈10.0 (±)
