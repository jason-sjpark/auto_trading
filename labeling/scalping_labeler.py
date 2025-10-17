import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ==============================================================
# 🎯 핵심 원칙
# ==============================================================
# - 입력: OHLCV DataFrame (timestamp, open, high, low, close)
# - 출력: (label, future_return) 컬럼 추가된 DataFrame
# - 라벨 규칙 (기본 기준 ±0.05%)
#   • 상승 (1): future_return >= threshold
#   • 하락 (-1): future_return <= -threshold
#   • 횡보 (0): |future_return| < threshold

# ==============================================================
# ⚙️ 라벨링 함수
# ==============================================================

def make_scalping_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold_pct: float = 0.05,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    스캘핑용 3-Class 라벨 생성기
    Args:
        df: OHLCV DataFrame (timestamp, close 필수)
        horizon: N틱(초) 뒤의 미래 가격을 비교
        threshold_pct: ±0.05% 이상을 상승/하락으로 구분
        price_col: 기준가격 (기본: close)
    Returns:
        df: 'future_return', 'label' 컬럼이 추가된 DataFrame

    | 항목                | 설명                         |
    | ----------------- | -------------------------- |
    | **기준가격**          | 현재 시점의 close               |
    | **비교대상**          | `horizon`초 뒤의 close        |
    | **future_return** | (미래가격 - 현재가격) / 현재가격 * 100 |
    | **라벨 기준**         | ±threshold_pct (기본 ±0.05%) |
    | **결과 라벨**         | 상승: `1`, 횡보: `0`, 하락: `-1` |

    | 파라미터            | 의미                 | 기본값   |
    | --------------- | ------------------ | ----- |
    | `horizon`       | 미래 몇 초 뒤의 변화를 비교할지 | 5     |
    | `threshold_pct` | 라벨 구분 임계값 (%)      | 0.05  |
    | `price_col`     | 라벨링 기준 가격 컬럼명      | close |


    """

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # future return 계산 (% 단위)
    df["future_price"] = df[price_col].shift(-horizon)
    df["future_return"] = (df["future_price"] - df[price_col]) / df[price_col] * 100  # %
    
    # 라벨 생성
    df["label"] = 0
    df.loc[df["future_return"] >= threshold_pct, "label"] = 1
    df.loc[df["future_return"] <= -threshold_pct, "label"] = -1

    # 마지막 horizon 구간은 미래가격 없음 → 제거
    df = df.dropna(subset=["future_price"]).reset_index(drop=True)

    return df


# ==============================================================
# 🧠 보조 함수
# ==============================================================

def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    라벨 분포 요약
    """
    counts = df["label"].value_counts().sort_index()
    total = counts.sum()
    summary = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "ratio(%)": (counts.values / total * 100).round(2)
    })
    mapping = {-1: "하락", 0: "횡보", 1: "상승"}
    summary["meaning"] = summary["label"].map(mapping)
    return summary


# ==============================================================
# 🔬 테스트용 메인
# ==============================================================

if __name__ == "__main__":
    np.random.seed(42)
    timestamps = pd.date_range("2025-10-17 09:00:00", periods=1000, freq="S")
    prices = np.cumsum(np.random.normal(0, 0.05, 1000)) + 54000  # 작은 랜덤 변동
    df = pd.DataFrame({"timestamp": timestamps, "close": prices})

    labeled = make_scalping_labels(df, horizon=5, threshold_pct=0.05)
    print("✅ 라벨링 완료")
    print(labeled[["timestamp", "close", "future_return", "label"]].head(10))
    print("\n📊 라벨 분포:")
    print(label_summary(labeled))
