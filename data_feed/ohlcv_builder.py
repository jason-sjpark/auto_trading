import pandas as pd
from pathlib import Path
from datetime import datetime

# ==============================================================
# 설정
# ==============================================================
RAW_PATH = Path("./data/raw")
PROCESSED_PATH = Path("./data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# 지원 타임프레임 (초 단위)
TIMEFRAMES = {
    "0.5s": "500ms",
    "1s": "1s",
    "5s": "5s",
}

# ==============================================================
# 함수 정의
# ==============================================================

def load_trades(date: str = None):
    """수집된 tick(aggTrades) parquet 파일 불러오기"""
    if date is None:
        date = datetime.utcnow().strftime("%Y%m%d")

    file_path = RAW_PATH / f"aggTrades_{date}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"❌ No aggTrades file found for {date}.")

    df = pd.read_parquet(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def build_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Tick 데이터(aggTrades) → OHLCV 변환
    timeframe 예시: '500ms', '1s', '5s'
    """

    ohlcv = (
        df.resample(timeframe, on="timestamp")
        .agg({
            "price": ["first", "max", "min", "last"],
            "qty": "sum",
        })
        .dropna()
    )
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv.reset_index(inplace=True)
    return ohlcv


def save_ohlcv(ohlcv: pd.DataFrame, timeframe_label: str, date: str = None):
    """변환된 OHLCV 데이터를 parquet으로 저장"""
    if date is None:
        date = datetime.utcnow().strftime("%Y%m%d")

    out_path = PROCESSED_PATH / f"ohlcv_{timeframe_label}_{date}.parquet"
    ohlcv.to_parquet(out_path, index=False)
    print(f"💾 Saved {len(ohlcv)} rows → {out_path}")


def run_ohlcv_builder():
    """전체 파이프라인 실행"""
    date = datetime.utcnow().strftime("%Y%m%d")
    print(f"🚀 Loading trades for {date}")
    df = load_trades(date)

    print(f"📊 Source data: {len(df)} ticks from {df['timestamp'].min()} to {df['timestamp'].max()}")

    for label, tf in TIMEFRAMES.items():
        print(f"⏱ Building OHLCV ({label}) ...")
        ohlcv = build_ohlcv(df, tf)
        save_ohlcv(ohlcv, label, date)


if __name__ == "__main__":
    run_ohlcv_builder()
