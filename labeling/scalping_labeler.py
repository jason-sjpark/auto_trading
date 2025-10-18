import pandas as pd
import numpy as np
from typing import Tuple

EPS = 1e-12

def _to_dt_utc_naive(s) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

def make_scalping_labels(
    ohlcv_df: pd.DataFrame,
    horizon_s: int = 5,
    threshold_bp: float = 10.0,   # 10 bp = 0.10%
    neutral_band: bool = True,    # True → {-1,0,+1} / False → { -1,+1 }
    price_col: str = "close",
) -> pd.DataFrame:
    """
    future_return(t+h, t) = close_{t+h} / close_t - 1
    label:
      +1 if r >= +thr, -1 if r <= -thr, else 0(중립, neutral_band=True인 경우)
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.DataFrame(columns=["timestamp", "label", "future_return"])

    df = ohlcv_df.copy()
    df["timestamp"] = _to_dt_utc_naive(df["timestamp"])
    df = df.dropna(subset=["timestamp", price_col]).sort_values("timestamp").reset_index(drop=True)

    # horizon 만큼 미래 가격
    # 시간격자 1s 가정. 로버스트하게 asfreq로 정렬 후 shift.
    base = df.set_index("timestamp").sort_index()
    base = base.asfreq("1s", method="pad")
    fwd = base[price_col].shift(-int(horizon_s)).rename("fwd_price")

    cur = base[price_col]
    ret = (fwd / (cur + EPS)) - 1.0
    thr = threshold_bp / 10000.0  # bp → ratio

    if neutral_band:
        lab = np.where(ret >= thr, 1, np.where(ret <= -thr, -1, 0))
    else:
        lab = np.where(ret >= thr, 1, -1)

    out = pd.DataFrame({
        "timestamp": base.index,
        "label": lab,
        "future_return": ret.fillna(0.0),
    })
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out
