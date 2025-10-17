import pandas as pd
import numpy as np
from typing import Dict

# ==============================================================
# ⚙️ Technical Indicators (OHLCV 기반)
# ==============================================================

def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index, name="rsi")


def calc_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_bollinger_bands(df: pd.DataFrame, period: int = 20, std_factor: float = 2.0):
    ma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    width = (upper - lower) / ma
    return ma, upper, lower, width


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    cum_vol = df["volume"].cumsum()
    cum_vol_price = (df["close"] * df["volume"]).cumsum()
    return (cum_vol_price / cum_vol).rename("vwap")


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.rename("atr")


# ==============================================================
# 🧠 통합 Feature Extractor
# ==============================================================

def extract_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력: OHLCV DataFrame
    출력: 기술적 지표가 추가된 DataFrame

    | 지표              | 의미              |
    | --------------- | --------------- |
    | `ema_9, 20, 50` | 단기/중기/장기 지수이동평균 |
    | `rsi_14`        | 과매수·과매도 구간 식별   |
    | `bb_width`      | 볼린저 밴드 폭 (변동성)  |
    | `atr_14`        | 절대적 변동성         |
    | `vwap`          | 거래량 가중 평균가      |
    | `momentum_5`    | 단기 추세 변화량       |
    | `volatility_5`  | 5틱 변동성 표준편차     |

    """
    df = df.copy()

    df["ema_9"] = calc_ema(df, 9)
    df["ema_20"] = calc_ema(df, 20)
    df["ema_50"] = calc_ema(df, 50)
    df["rsi_14"] = calc_rsi(df, 14)
    df["vwap"] = calc_vwap(df)
    df["atr_14"] = calc_atr(df, 14)

    ma, upper, lower, width = calc_bollinger_bands(df, 20)
    df["bb_ma"] = ma
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = width

    # 모멘텀 / 변동성 보조 피처
    df["momentum_5"] = df["close"].diff(5)
    df["volatility_5"] = df["close"].pct_change().rolling(5).std()

    # 결측치 제거
    df = df.dropna().reset_index(drop=True)
    return df


# ==============================================================
# 🔬 테스트용 메인
# ==============================================================

if __name__ == "__main__":
    # 테스트용 데이터
    np.random.seed(42)
    data = {
        "timestamp": pd.date_range("2025-10-17 09:00:00", periods=100, freq="S"),
        "open": np.random.normal(54000, 5, 100),
        "high": np.random.normal(54010, 5, 100),
        "low": np.random.normal(53990, 5, 100),
        "close": np.random.normal(54000, 5, 100),
        "volume": np.random.uniform(1, 5, 100),
    }
    df = pd.DataFrame(data)
    tech = extract_technical_indicators(df)
    print("🧩 Technical Indicators Sample:")
    print(tech.tail(3).T)
