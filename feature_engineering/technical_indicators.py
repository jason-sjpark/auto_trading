import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력: ['timestamp','open','high','low','close','volume']
    출력: timestamp + 아래 지표들
      - EMA(9,20,50)
      - RSI(14): Wilder-style EWM
      - ATR(14): TR rolling mean
      - Bollinger(20, 2σ): bb_ma, bb_upper, bb_lower, bb_width
      - VWAP(cum): sum(price*vol)/sum(vol)
      - Momentum_5: pct_change(5)
      - Volatility_5: 1-step pct_change rolling std(5)
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        if c not in df: df[c] = np.nan

    # EMA
    df["ema_9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Bollinger(20)
    df["bb_ma"]   = df["close"].rolling(window=20, min_periods=1).mean()
    df["bb_std"]  = df["close"].rolling(window=20, min_periods=1).std()
    df["bb_upper"]= df["bb_ma"] + 2*df["bb_std"]
    df["bb_lower"]= df["bb_ma"] - 2*df["bb_std"]
    df["bb_width"]= (df["bb_upper"] - df["bb_lower"]) / df["bb_ma"].replace(0, np.nan)

    # VWAP(cum)
    df["vwap"] = (df["close"]*df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    # Momentum/Volatility
    df["momentum_5"]   = df["close"].pct_change(periods=5)
    df["volatility_5"] = df["close"].pct_change().rolling(window=5, min_periods=1).std()

    df = df.dropna(subset=["timestamp"]).fillna(method="ffill").fillna(method="bfill")
    cols = ["timestamp","ema_9","ema_20","ema_50","rsi_14","vwap","atr_14",
            "bb_ma","bb_upper","bb_lower","bb_width","momentum_5","volatility_5"]
    return df[cols]
