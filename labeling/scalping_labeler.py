import pandas as pd
import numpy as np

def make_scalping_labels(ohlcv_df: pd.DataFrame, horizon: int = 5, threshold_pct: float = 0.05) -> pd.DataFrame:
    """
    horizon 이후 수익률 기반 3-class:
      y = +1 if future_return >= +threshold_pct
      y = -1 if future_return <= -threshold_pct
      y =  0 otherwise
    future_return = (future_close - current_close) / current_close
    """
    df = ohlcv_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["future_close"]  = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / (df["close"] + 1e-9)
    cond_up   = df["future_return"] >= threshold_pct
    cond_down = df["future_return"] <= -threshold_pct
    df["label"] = np.where(cond_up, 1, np.where(cond_down, -1, 0))
    return df[["timestamp","label","future_return"]].dropna()
