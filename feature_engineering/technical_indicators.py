import pandas as pd
import numpy as np

# ==============================================================
# 📊 Technical Indicators
# ==============================================================

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 OHLCV 데이터에서 대표적인 기술 지표들을 계산해 반환.
    최소 컬럼: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # 안전장치: 필수 컬럼 검사
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"⚠️ [compute_technical_indicators] 필수 컬럼 누락: {required - set(df.columns)}")
        for c in required:
            if c not in df:
                df[c] = np.nan

    # ---- EMA ----
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ---- RSI ----
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ---- ATR ----
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # ---- Bollinger Bands ----
    df["bb_ma"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["bb_std"] = df["close"].rolling(window=20, min_periods=1).std()
    df["bb_upper"] = df["bb_ma"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_ma"] - (df["bb_std"] * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_ma"].replace(0, np.nan)

    # ---- VWAP ----
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    # ---- Momentum & Volatility ----
    df["momentum_5"] = df["close"].pct_change(periods=5)
    df["volatility_5"] = df["close"].pct_change().rolling(window=5, min_periods=1).std()

    # ---- 정리 ----
    df = df.dropna(subset=["timestamp"])
    df = df.fillna(method="ffill").fillna(method="bfill")

    # 필요한 컬럼만 반환
    cols = [
        "timestamp",
        "ema_9", "ema_20", "ema_50",
        "rsi_14",
        "vwap",
        "atr_14",
        "bb_ma", "bb_upper", "bb_lower", "bb_width",
        "momentum_5", "volatility_5"
    ]
    df = df[cols]
    print(f"✅ [compute_technical_indicators] 완료: shape={df.shape}")
    return df


# ==============================================================
# 🔬 테스트
# ==============================================================

if __name__ == "__main__":
    import datetime
    now = pd.Timestamp.utcnow().floor("s")

    data = {
        "timestamp": pd.date_range(now, periods=30, freq="S"),
        "open": np.random.rand(30) * 100,
        "high": np.random.rand(30) * 100,
        "low": np.random.rand(30) * 100,
        "close": np.random.rand(30) * 100,
        "volume": np.random.rand(30) * 50
    }
    df = pd.DataFrame(data)
    tech = compute_technical_indicators(df)
    print(tech.head())
