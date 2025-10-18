import pandas as pd
import numpy as np
from typing import Dict, Iterable

EPS = 1e-12


# -----------------------------
# 공통 전처리
# -----------------------------
def _prep_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    기대 컬럼:
      - timestamp (datetime/string/epoch → UTC-naive)
      - price (float)
      - qty (float)
      - side ('buy'/'sell') or is_buyer_maker(bool)
    """
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "side"])\
                 .set_index(pd.DatetimeIndex([], name="timestamp"))

    df = trades_df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # side 유도
    if "side" not in df.columns:
        ibm = df.get("is_buyer_maker")
        if ibm is not None:
            # taker가 buy면 is_buyer_maker == False → side='buy'
            df["side"] = (~ibm.astype(bool)).map({True: "buy", False: "sell"})
        else:
            df["side"] = "buy"

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"]   = pd.to_numeric(df["qty"],   errors="coerce")
    df = df.dropna(subset=["price", "qty"])
    df = df[df["qty"] > 0]

    return df.set_index("timestamp")


# -----------------------------
# (A) 배치/백테스트용: 고정 버킷 집계
# -----------------------------
def _agg_window_bins(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    고정 간격(freq 기준)으로 버킷팅 후 각 버킷에서 필요한 집계를 계산.
    반환 인덱스: DatetimeIndex(버킷 끝시각)
    """
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))

    # buy/sell 마스크
    is_buy = (df["side"] == "buy").astype(float)

    g = df.resample(freq, label="right", closed="right")

    # 기본 집계
    count = g["qty"].count().astype(float)  # trade count
    vol   = g["qty"].sum().astype(float)    # total volume
    buy_v = g.apply(lambda x: float((x["qty"] * (x["side"] == "buy")).sum()))
    sell_v= g.apply(lambda x: float((x["qty"] * (x["side"] == "sell")).sum()))
    # VWAP = sum(price*qty)/sum(qty)
    vwap  = g.apply(lambda x: float((x["price"] * x["qty"]).sum()) / (float(x["qty"].sum()) + EPS))

    out = pd.DataFrame({
        "trade_count": count,
        "total_volume": vol,
        "buy_vol": buy_v,
        "sell_vol": sell_v,
        "vwap": vwap,
    })

    # 파생
    denom = (out["buy_vol"] + out["sell_vol"]).replace(0, np.nan)
    out["buy_sell_ratio"] = (out["buy_vol"] / denom).fillna(0.5)  # 완전 무체결이면 중립 0.5

    out["volume_delta"]   = out["buy_vol"] - out["sell_vol"]
    out["trade_pressure"] = (out["buy_vol"] - out["sell_vol"]) / (out["buy_vol"] + out["sell_vol"] + EPS)

    return out


def _to_1s_grid(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """
    1초 그리드로 변환.
      - how == "mean"  → 구간 평균 (0.5s처럼 1초 안에 여러 버킷이 있는 경우)
      - how == "last"  → 구간 마지막 값 유지 (1s/5s 같은 저빈도)
      - how == "ffill" → 저빈도 값을 1초 단위로 보간(5s 등)
    """
    if df.empty:
        return df

    if how == "mean":
        return df.resample("1s").mean()
    elif how == "last":
        return df.resample("1s").last()
    elif how == "ffill":
        return df.resample("1s").ffill()
    else:
        return df.resample("1s").last()


def extract_trade_features(trades_df: pd.DataFrame,
                           windows: Iterable[str] = ("0.5s", "1s", "5s")) -> pd.DataFrame:
    """
    고정 버킷 집계 기반:
      - 0.5s: 500ms 버킷 집계 → 1초로 평균(mean) 앵커
      - 1s  : 1s 버킷 집계 → 1초 그리드 동일(last)
      - 5s  : 5s 버킷 집계 → 1초로 ffill(해당 5초 구간 동안 값 유지)
    """
    df = _prep_trades(trades_df)
    if df.empty:
        cols = []
        for w in windows:
            cols += [f"trade_count@{w}", f"trade_intensity@{w}", f"buy_sell_ratio@{w}",
                     f"volume_delta@{w}", f"vwap@{w}", f"trade_pressure@{w}", f"volume_spike@{w}"]
        return pd.DataFrame(columns=["timestamp"] + cols)

    parts = []

    for w in windows:
        if w == "0.5s":
            # 500ms 집계 → 1초 평균
            agg = _agg_window_bins(df, "500ms")
            # trade_intensity(TPS) = count / 0.5
            agg["trade_intensity"] = agg["trade_count"] / 0.5
            # volume_spike: 60초(=120 bins) 기준 z-score, 500ms 기준으로 rolling
            # 1초 평균으로 내릴 것이므로 일단 500ms 상에서 계산
            vol = agg["total_volume"]
            win = 120  # 120 * 0.5s = 60s
            mean = vol.rolling(win, min_periods=1).mean()
            std  = vol.rolling(win, min_periods=1).std().fillna(0.0)
            agg["volume_spike"] = (vol - mean) / (std + EPS)

            f_1s = _to_1s_grid(agg, "mean")
            f_1s = f_1s.rename(columns=lambda c: f"{c}@0.5s" if c not in ["total_volume"] else c)

            # total_volume@0.5s는 의미 애매하니 드롭(이미 @1s/5s가 있음)
            f_1s = f_1s.drop(columns=["total_volume"], errors="ignore")
            parts.append(f_1s)

        elif w == "1s":
            agg = _agg_window_bins(df, "1s")
            agg["trade_intensity"] = agg["trade_count"] / 1.0
            # volume_spike: 60초 기준 z
            vol = agg["total_volume"]
            mean = vol.rolling(60, min_periods=1).mean()
            std  = vol.rolling(60, min_periods=1).std().fillna(0.0)
            agg["volume_spike"] = (vol - mean) / (std + EPS)

            f_1s = _to_1s_grid(agg, "last")
            f_1s = f_1s.rename(columns=lambda c: f"{c}@1s" if c not in ["total_volume"] else c)
            f_1s = f_1s.drop(columns=["total_volume"], errors="ignore")
            parts.append(f_1s)

        elif w == "5s":
            agg = _agg_window_bins(df, "5s")
            agg["trade_intensity"] = agg["trade_count"] / 5.0
            # volume_spike: 5분(=60개 5s 버킷) 기준 z
            vol = agg["total_volume"]
            mean = vol.rolling(60, min_periods=1).mean()
            std  = vol.rolling(60, min_periods=1).std().fillna(0.0)
            agg["volume_spike"] = (vol - mean) / (std + EPS)

            f_1s = _to_1s_grid(agg, "ffill")
            f_1s = f_1s.rename(columns=lambda c: f"{c}@5s" if c not in ["total_volume"] else c)
            f_1s = f_1s.drop(columns=["total_volume"], errors="ignore")
            parts.append(f_1s)

        else:
            # 혹시 다른 창이 들어오면 안전하게 1s mean으로 다운샘플
            agg = _agg_window_bins(df, w)
            # 초 길이
            seconds = max(pd.to_timedelta(w).total_seconds(), 1.0)
            agg["trade_intensity"] = agg["trade_count"] / seconds
            vol = agg["total_volume"]
            mean = vol.rolling(int(60/seconds), min_periods=1).mean()
            std  = vol.rolling(int(60/seconds), min_periods=1).std().fillna(0.0)
            agg["volume_spike"] = (vol - mean) / (std + EPS)
            f_1s = _to_1s_grid(agg, "mean")
            f_1s = f_1s.rename(columns=lambda c: f"{c}@{w}" if c not in ["total_volume"] else c)
            f_1s = f_1s.drop(columns=["total_volume"], errors="ignore")
            parts.append(f_1s)

    merged = pd.concat(parts, axis=1).sort_index()
    # 혹시 일부 초가 비면 보간
    merged = merged.ffill().bfill()
    merged = merged.reset_index().rename(columns={"index": "timestamp"})

    return merged


# -----------------------------
# (B) 실시간 스냅샷용 (Dict 반환) — 기존 API 유지
# -----------------------------
def extract_trade_features_snapshot(trades_df: pd.DataFrame, prev_avg_vol: float = 0.0) -> Dict[str, float]:
    """
    단일 구간 스냅샷 피처:
      - trade_count
      - trade_intensity (TPS) = total_trades / duration_sec
      - buy_sell_ratio = buy_qty / (buy_qty + sell_qty)
      - volume_delta   = buy_qty - sell_qty
      - vwap           = sum(price*qty) / sum(qty)
      - trade_pressure = (buy_qty - sell_qty) / (buy_qty + sell_qty)
      - volume_spike   = total_volume / prev_avg_vol (clip ≤ 10)
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

    base_vol = total_volume if prev_avg_vol == 0 else prev_avg_vol
    volume_spike = (total_volume / (base_vol + EPS)) if base_vol > 0 else 1.0
    volume_spike = float(min(volume_spike, 10.0))

    feats = {
        "trade_count": float(total_trades),
        "trade_intensity": float(trade_intensity),
        "buy_sell_ratio": float(buy_vol / (buy_vol + sell_vol + EPS)),
        "volume_delta": float(buy_vol - sell_vol),
        "vwap": float(df["amount"].sum() / (total_volume + EPS)),
        "trade_pressure": float((buy_vol - sell_vol) / (buy_vol + sell_vol + EPS)),
        "volume_spike": volume_spike
    }
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0
    return feats
