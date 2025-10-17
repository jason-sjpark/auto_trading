# scripts/make_training_data.py
import os
import pandas as pd
import numpy as np

from feature_engineering.feature_pipeline import BatchFeaturePipeline
from labeling.scalping_labeler import make_scalping_labels

# -------------------------
# 경로 설정 (필요시 수정)
# -------------------------
OHLCV_PATH = "data/processed/ohlcv_1s_20251017.parquet"
TRADES_PATH = "data/raw/aggTrades_20251017.parquet"
DEPTH_PATH  = "data/raw/depth_20251017.parquet"
OUT_PATH    = "data/features/features_1s.parquet"

# -------------------------
# 유틸: 타임스탬프 표준화
# -------------------------
def _tz_normalize(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """
    - 모든 ts를 tz-naive UTC 기준으로 통일
    - 문자열/epoch 혼재도 흡수
    - 정렬/중복 제거까지 수행
    """
    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    # tz-aware → UTC로 맞춘 뒤 tz 제거(naive)
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df[col] = s
    # 정렬 + 중복 제거
    df = df.sort_values(col)
    df = df[~df[col].isna()].drop_duplicates(subset=[col])
    return df

def _print_range(name: str, df: pd.DataFrame, col: str = "timestamp"):
    if df.empty:
        print(f"⚠️ {name} EMPTY")
    else:
        print(f"⏱ {name} range: {df[col].min()}  →  {df[col].max()}  ({len(df)} rows)")

# -------------------------
# 1) 로드
# -------------------------
print("📥 Loading source data...")
ohlcv  = pd.read_parquet(OHLCV_PATH)
trades = pd.read_parquet(TRADES_PATH)
depth  = pd.read_parquet(DEPTH_PATH)

# 표준화(타임존/정렬/중복)
ohlcv  = _tz_normalize(ohlcv,  "timestamp")
trades = _tz_normalize(trades, "timestamp")
depth  = _tz_normalize(depth,  "timestamp")

_print_range("OHLCV", ohlcv)
_print_range("TRADES", trades)
_print_range("DEPTH", depth)

# -------------------------
# 2) 피처 생성
# -------------------------
pipe = BatchFeaturePipeline(timeframes=["0.5s", "1s", "5s"])
features_df = pipe.build_features(
    ohlcv_df=ohlcv,
    trades_df=trades,
    depth_df=depth
)


# 표준화 + 안전벨트
features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]
features_df = _tz_normalize(features_df, "timestamp")

_print_range("FEATURES", features_df)

# -------------------------
# 3) 라벨 생성 (OHLCV 기반)
# -------------------------
labeled = make_scalping_labels(ohlcv, horizon=5, threshold_pct=0.05)
labeled = labeled[["timestamp", "label", "future_return"]]
labeled = labeled.loc[:, ~labeled.columns.duplicated(keep="first")]
labeled = _tz_normalize(labeled, "timestamp")

_print_range("LABELS", labeled)

# -------------------------
# 4) 병합 로직 (단계적 시도)
# -------------------------
def try_merge(tol):
    return pd.merge_asof(
        features_df.sort_values("timestamp"),
        labeled.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol
    )

# 4-1) tolerance 사다리: 0.5s → 2s → 5s → 10s
tolerances = [pd.Timedelta(milliseconds=500),
              pd.Timedelta(seconds=2),
              pd.Timedelta(seconds=5),
              pd.Timedelta(seconds=10)]

final_dataset = None
for tol in tolerances:
    tmp = try_merge(tol)
    print(f"🧪 merge_asof with tolerance={tol}: result rows = {len(tmp)}")
    if len(tmp) > 0:
        final_dataset = tmp
        print(f"✅ merged with tolerance={tol}")
        break

# 4-2) 그래도 0행이면: 라벨을 특성 타임스탬프에 보간(가장 가까운 이웃)
if final_dataset is None or len(final_dataset) == 0:
    print("⚠️ asof merge still empty. Falling back to nearest-neighbor labeling...")

    # 두 축을 epoch(ms)로 바꿔서 최근접 인덱스 매칭
    f_ms = features_df["timestamp"].view("int64") // 10**6
    l_ms = labeled["timestamp"].view("int64") // 10**6
    lf   = labeled.set_index(labeled["timestamp"])

    # 각 feature ts에 대해 가장 가까운 label ts 찾기
    # (효율적으로는 searchsorted를 사용)
    lab_ts = labeled["timestamp"].to_numpy()
    feat_ts = features_df["timestamp"].to_numpy()

    pos = np.searchsorted(lab_ts, feat_ts)
    pos = np.clip(pos, 1, len(lab_ts)-1)
    prev = lab_ts[pos-1]
    next = lab_ts[pos]
    # 어느 쪽이 더 가까운지
    pick = np.where((feat_ts - prev) <= (next - feat_ts), prev, next)

    nn_lab = labeled.set_index("timestamp").loc[pick, ["label", "future_return"]].reset_index(drop=True)
    final_dataset = pd.concat([features_df.reset_index(drop=True), nn_lab], axis=1)
    print(f"✅ fallback merged rows = {len(final_dataset)}")

# -------------------------
# 5) 최종 저장
# -------------------------
final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated(keep="first")]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
final_dataset.to_parquet(OUT_PATH, index=False)

print("🎉 DONE: saved →", OUT_PATH)
print("shape:", final_dataset.shape)
print("sample:")
print(final_dataset.head(3))
