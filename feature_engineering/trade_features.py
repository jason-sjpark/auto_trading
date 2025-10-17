import pandas as pd
import numpy as np
from typing import Dict

def extract_trade_features(trades_df: pd.DataFrame, prev_avg_vol: float = 0.0) -> Dict[str, float]:
    """
    체결 구간 피처:
      - trade_count        : 체결 횟수
      - trade_intensity    : 초당 체결 횟수(TPS) = total_trades / duration_sec  ✅
      - buy_sell_ratio     : buy_qty / (buy_qty + sell_qty)
      - volume_delta       : buy_qty - sell_qty
      - vwap               : sum(price*qty) / sum(qty)
      - trade_pressure     : (buy_qty - sell_qty) / (buy_qty + sell_qty)
      - volume_spike       : total_volume / prev_avg_vol  (clip ≤ 10)
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            "trade_count": 0.0, "trade_intensity": 0.0, "buy_sell_ratio": 0.5,
            "volume_delta": 0.0, "vwap": 0.0, "trade_pressure": 0.0, "volume_spike": 0.0
        }

    df = trades_df.copy()
    for col in ["timestamp","price","qty","side"]:
        if col not in df.columns:
            if col == "timestamp": df[col] = pd.Timestamp.utcnow()
            elif col in ("price","qty"): df[col] = 0.0
            elif col == "side": df[col] = "buy"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp","price","qty"]).reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce").fillna(0.0)
    df["side"]  = df["side"].astype(str).str.lower()
    if len(df) == 0:
        return {
            "trade_count": 0.0, "trade_intensity": 0.0, "buy_sell_ratio": 0.5,
            "volume_delta": 0.0, "vwap": 0.0, "trade_pressure": 0.0, "volume_spike": 0.0
        }

    total_trades = int(len(df))
    total_volume = float(df["qty"].sum())
    df["amount"] = df["price"] * df["qty"]
    buy_vol  = float(df.loc[df["side"] == "buy",  "qty"].sum())
    sell_vol = float(df.loc[df["side"] == "sell", "qty"].sum())

    duration_sec = float(max((df["timestamp"].max() - df["timestamp"].min()).total_seconds(), 1e-3))
    trade_intensity = total_trades / duration_sec  # ✅ TPS

    trade_count    = float(total_trades)
    buy_sell_ratio = buy_vol / (buy_vol + sell_vol + 1e-9)
    volume_delta   = buy_vol - sell_vol
    vwap           = float(df["amount"].sum()) / (total_volume + 1e-9)
    trade_pressure = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)
    base_vol       = total_volume if prev_avg_vol == 0 else prev_avg_vol
    volume_spike   = (total_volume / (base_vol + 1e-9)) if base_vol > 0 else 1.0
    volume_spike   = float(min(volume_spike, 10.0))

    feats = {
        "trade_count": trade_count, "trade_intensity": float(trade_intensity),
        "buy_sell_ratio": float(buy_sell_ratio), "volume_delta": float(volume_delta),
        "vwap": float(vwap), "trade_pressure": float(trade_pressure), "volume_spike": float(volume_spike)
    }
    for k,v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0
    return feats
