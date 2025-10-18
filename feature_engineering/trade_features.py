# feature_engineering/trade_features.py
import pandas as pd
import numpy as np
from typing import Dict, Iterable

EPS = 1e-12

def _prep_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=["timestamp","price","qty","side"])\
               .set_index(pd.DatetimeIndex([], name="timestamp"))

    df = trades_df.copy()
    # UTC-naive 정규화
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # side 유도
    if "side" not in df.columns:
        ibm = df.get("is_buyer_maker")
        if ibm is not None:
            df["side"] = (~ibm.astype(bool)).map({True:"buy", False:"sell"})
        else:
            df["side"] = "buy"

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce")
    df = df.dropna(subset=["price","qty"])
    df = df[df["qty"] > 0]
    return df.set_index("timestamp")


def _roll_features(idx_df: pd.DataFrame, window: str) -> pd.DataFrame:
    df = idx_df

    ones  = pd.Series(1.0, index=df.index)
    count = ones.rolling(window, closed="right", min_periods=1).sum()

    qty = df["qty"]
    vol   = qty.rolling(window, closed="right", min_periods=1).sum()

    is_buy   = (df["side"] == "buy").astype(float)
    buy_vol  = (qty * is_buy).rolling(window, closed="right", min_periods=1).sum()
    sell_vol = (qty * (1.0 - is_buy)).rolling(window, closed="right", min_periods=1).sum()

    price = df["price"]
    vwap  = (price*qty).rolling(window, closed="right", min_periods=1).sum() / (vol + EPS)

    win_sec = max(pd.to_timedelta(window).total_seconds(), EPS)
    trade_intensity = count / win_sec  # TPS

    buy_sell_ratio = buy_vol / (sell_vol + EPS)
    volume_delta   = buy_vol - sell_vol
    trade_pressure = (buy_vol - sell_vol) / (buy_vol + sell_vol + EPS)

    vol_mean = vol.rolling(window, closed="right", min_periods=1).mean()
    vol_std  = vol.rolling(window, closed="right", min_periods=1).std()
    volume_spike = (vol - vol_mean) / (vol_std + EPS)

    out = pd.DataFrame(index=df.index)
    out[f"trade_count@{window}"]      = count
    out[f"trade_intensity@{window}"]  = trade_intensity
    out[f"buy_sell_ratio@{window}"]   = buy_sell_ratio
    out[f"volume_delta@{window}"]     = volume_delta
    out[f"vwap@{window}"]             = vwap
    out[f"trade_pressure@{window}"]   = trade_pressure
    out[f"volume_spike@{window}"]     = volume_spike
    return out


def extract_trade_features(trades_df: pd.DataFrame,
                           windows: Iterable[str] = ("0.5s","1s","5s")) -> pd.DataFrame:
    """
    배치용: 각 윈도우 롤링 → 최종적으로 1초 그리드(resample('1s').last())에 고정.
    컬럼 라벨은 사람이 읽기 쉬운 @0.5s, @1s, @5s 유지.
    """
    df = _prep_trades(trades_df)
    if df.empty:
        cols = []
        for w in windows:
            cols += [f"trade_count@{w}", f"trade_intensity@{w}", f"buy_sell_ratio@{w}",
                     f"volume_delta@{w}", f"vwap@{w}", f"trade_pressure@{w}", f"volume_spike@{w}"]
        return pd.DataFrame(columns=["timestamp"] + cols)

    # 내부 계산 창 매핑(0.5s → 500ms)
    win_map = {"0.5s":"500ms", "1s":"1s", "5s":"5s"}

    parts = []

    for w in windows:
        internal = win_map.get(w, w)
        f = _roll_features(df, internal)
        if w == "0.5s":
            f_1s = f.resample("1s").mean()
        else:
            f_1s = f.resample("1s").last()
        if internal != w:
            f_1s = f_1s.rename(columns=lambda c: c.replace(f"@{internal}", f"@{w}"))
        parts.append(f_1s)

    merged = pd.concat(parts, axis=1)
    merged = merged.reset_index().rename(columns={"index":"timestamp"})
    merged = merged.sort_values("timestamp").ffill().bfill()
    return merged


def extract_trade_features_snapshot(trades_df: pd.DataFrame, prev_avg_vol: float = 0.0) -> Dict[str, float]:
    """
    실시간 스냅샷용: 단일 구간 집계(feature dict)
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
    trade_intensity = total_trades / duration_sec  # TPS

    feats = {
        "trade_count": float(total_trades),
        "trade_intensity": float(trade_intensity),
        "buy_sell_ratio": float(buy_vol / (buy_vol + sell_vol + 1e-9)),
        "volume_delta": float(buy_vol - sell_vol),
        "vwap": float(df["amount"].sum() / (total_volume + 1e-9)),
        "trade_pressure": float((buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)),
        "volume_spike": float(min((total_volume / (total_volume if prev_avg_vol == 0 else prev_avg_vol + 1e-9)) if (total_volume if prev_avg_vol == 0 else prev_avg_vol) > 0 else 1.0, 10.0))
    }
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0
    return feats
