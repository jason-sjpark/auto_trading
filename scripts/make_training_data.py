from feature_engineering.feature_pipeline import BatchFeaturePipeline
from labeling.scalping_labeler import make_scalping_labels
import pandas as pd, os

ohlcv_path = "data/processed/ohlcv_1s_20251017.parquet"
trades_path = "data/raw/aggTrades_20251017.parquet"
depth_path  = "data/raw/depth_20251017.parquet"
out_path    = "data/features/features_1s.parquet"

pipe = BatchFeaturePipeline(timeframes=["0.5s", "1s", "5s"])
ohlcv = pd.read_parquet(ohlcv_path)
trades = pd.read_parquet(trades_path)
depth  = pd.read_parquet(depth_path)

features_df = pipe.build_features(ohlcv, trades, depth)

# 라벨 생성
labeled = make_scalping_labels(ohlcv, horizon=5, threshold_pct=0.05)

# timestamp 기준 근접 병합
final_dataset = pd.merge_asof(
    features_df.sort_values("timestamp"),
    labeled[["timestamp","label","future_return"]],
    on="timestamp", direction="nearest", tolerance=pd.Timedelta(milliseconds=500)
)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
final_dataset.to_parquet(out_path, index=False)

print("✅ 데이터셋 저장 완료:", out_path)
