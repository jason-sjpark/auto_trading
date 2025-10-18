import pandas as pd
import numpy as np

def resample_ohlcv(trades_or_ticks: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    입력: 최소 ['timestamp','price','qty']  (ticks/aggTrades)
    출력: ['timestamp','open','high','low','close','volume']
    """
    df = trades_or_ticks.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp","price"]).sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    o = df["price"].resample(freq).first()
    h = df["price"].resample(freq).max()
    l = df["price"].resample(freq).min()
    c = df["price"].resample(freq).last()
    v = df.get("qty", pd.Series(dtype=float)).resample(freq).sum() if "qty" in df.columns else None
    vol = v if v is not None else pd.Series(0.0, index=c.index)

    ohlcv = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":vol}).reset_index()
    ohlcv.rename(columns={"timestamp":"timestamp"}, inplace=True)
    return ohlcv.dropna(subset=["open","high","low","close"])
