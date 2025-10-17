import pandas as pd
from feature_engineering.feature_pipeline import BatchFeaturePipeline

# 1) 저장된 parquet 불러오기 (예시 경로는 여러분 환경에 맞게)
ohlcv = pd.read_parquet("data/processed/ohlcv_1s_20251017.parquet")
trades = pd.read_parquet("data/raw/aggTrades_20251017.parquet")
depth  = pd.read_parquet("data/raw/depth_20251017.parquet")

pipe = BatchFeaturePipeline(timeframes=["0.5s","1s","5s","1min"])
features_df = pipe.build_features(ohlcv_df=ohlcv, trades_df=trades, depth_df=depth)

print(features_df.head())
# → 이 결과 DataFrame이 모델 학습/백테스트 입력의 표준 테이블!
