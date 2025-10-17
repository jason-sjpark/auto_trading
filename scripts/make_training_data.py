# scripts/make_training_data.py
import os
import glob
import pandas as pd
import numpy as np

from feature_engineering.feature_pipeline import BatchFeaturePipeline
from feature_engineering.external_features import build_external_features
from labeling.scalping_labeler import make_scalping_labels

# ============================
# 설정 (필요시 수정)
# ============================
OHLCV_PATH = "data/processed/ohlcv_1s_20251017.parquet"   # 샘플 경로
TRADES_PATH = "data/raw/aggTrades_20251017.parquet"
DEPTH_PATH  = "data/raw/depth_20251017.parquet"
OUT_PATH    = "data/features/features_1s.parquet"

# 라벨링 설정
LABEL_HORIZON = 5           # N초 뒤
LABEL_THRESHOLD_PCT = 0.05  # 5% 기준 (예: 스케일 확인 후 조정)

# asof 병합 허용 오차 사다리
ASOF_TOLERANCES = [
    pd.Timedelta(milliseconds=500),
    pd.Timedelta(seconds=2),
    pd.Timedelta(seconds=5),
    pd.Timedelta(seconds=10),
]


# ============================
# 유틸
# ============================
def _tz_normalize(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """
    - 모든 ts를 tz-naive UTC 기준으로 통일
    - 문자열/epoch/ms 혼재도 흡수 (상위에서 ms->datetime으로 이미 처리된 경우도 안전)
    - 정렬/중복 제거까지 수행
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])

    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    # tz-aware → UTC로 맞춘 뒤 tz 제거(naive)
    if hasattr(s, "dt"):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # 단일 Timestamp일 경우
        s = s.tz_convert("UTC").tz_localize(None)

    out = df.copy()
    out[col] = s
    out = out.dropna(subset=[col]).drop_duplicates(subset=[col]).sort_values(col)
    out.reset_index(drop=True, inplace=True)
    return out


def _print_range(name: str, df: pd.DataFrame, col: str = "timestamp"):
    if df is None or df.empty or col not in df.columns:
        print(f"⚠️ {name} EMPTY")
        return
    print(f"⏱ {name} range: {df[col].min()}  →  {df[col].max()}  ({len(df)} rows)")


def _latest(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _safe_read_parquet(path: str, expected_cols=None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=expected_cols or ["timestamp"])
    try:
        df = pd.read_parquet(path)
        if expected_cols:
            # 스키마 최소한 보정
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = np.nan
        return df
    except Exception as e:
        print(f"⚠️ read_parquet fail: {path} ({e})")
        return pd.DataFrame(columns=expected_cols or ["timestamp"])


def _try_asof_merge(left: pd.DataFrame, right: pd.DataFrame, tolerances) -> pd.DataFrame:
    """
    asof merge를 tolerance 사다리로 시도. 하나라도 성공하면 반환.
    둘 중 하나가 비면 left 그대로 반환.
    """
    if left is None or left.empty:
        return left
    if right is None or right.empty:
        return left

    for tol in tolerances:
        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=tol,
        )
        rows = len(merged.dropna(how="all"))
        print(f"🧪 merge_asof with tolerance={tol}: result rows = {len(merged)}")
        # 이 조건은 단순히 병합이 수행됐는지 확인용. 필요시 조건 강화 가능.
        if len(merged) > 0:
            print(f"✅ merged with tolerance={tol}")
            return merged
    print("⚠️ asof merge did not produce rows with given tolerances; returning last attempt")
    return merged  # 마지막 시도 결과 반환


# ============================
# 메인
# ============================
if __name__ == "__main__":
    print("📥 Loading source data...")

    # 1) 원천 데이터 로드
    ohlcv  = _safe_read_parquet(OHLCV_PATH, expected_cols=["timestamp","open","high","low","close","volume"])
    trades = _safe_read_parquet(TRADES_PATH, expected_cols=["timestamp","price","qty","side","trade_id","is_buyer_maker"])
    depth  = _safe_read_parquet(DEPTH_PATH,  expected_cols=["timestamp","bids","asks"])

    # 표준화(타임존/정렬/중복)
    ohlcv  = _tz_normalize(ohlcv,  "timestamp")
    trades = _tz_normalize(trades, "timestamp")
    depth  = _tz_normalize(depth,  "timestamp")

    _print_range("OHLCV", ohlcv)
    _print_range("TRADES", trades)
    _print_range("DEPTH", depth)

    # 2) 외부 지표 로드 → 외부 피처 생성
    ext_inputs = {
        "funding":       _safe_read_parquet(_latest("data/external/funding_rates_*.parquet")),
        "open_interest": _safe_read_parquet(_latest("data/external/open_interest_*.parquet")),
        "long_short_ratio": _safe_read_parquet(_latest("data/external/long_short_ratio_*.parquet")),
        "index_mark":    _safe_read_parquet(_latest("data/external/index_mark_*.parquet")),
        "liquidations":  _safe_read_parquet(_latest("data/external/liquidations_*.parquet")),
        "arbitrage":     _safe_read_parquet(_latest("data/external/arbitrage_spreads_*.parquet")),
    }
    external_df = build_external_features(ext_inputs)
    if external_df is None:
        external_df = pd.DataFrame(columns=["timestamp"])
    external_df = _tz_normalize(external_df, "timestamp")
    print("⏱ EXTERNAL range:",
          (external_df["timestamp"].min() if not external_df.empty else None),
          "→",
          (external_df["timestamp"].max() if not external_df.empty else None),
          (f"(rows {len(external_df)})" if not external_df.empty else "(empty)"))

    # 3) 피처 생성 (호가/체결/기술지표 + 외부피처 병합)
    pipe = BatchFeaturePipeline(timeframes=["0.5s", "1s", "5s"])
    features_df = pipe.build_features(
        ohlcv_df=ohlcv,
        trades_df=trades,
        depth_df=depth,
        external_df=external_df,   # ✅ 외부 피처 주입
    )

    # 안전 처리: 중복 컬럼 제거 + ts 표준화
    features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]
    features_df = _tz_normalize(features_df, "timestamp")
    _print_range("FEATURES", features_df)
    if features_df is None or features_df.empty:
        print("⚠️ FEATURES EMPTY")

    # 4) 라벨 생성 (OHLCV 기반)
    labeled = make_scalping_labels(ohlcv, horizon=LABEL_HORIZON, threshold_pct=LABEL_THRESHOLD_PCT)
    labeled = labeled[["timestamp", "label", "future_return"]]
    labeled = labeled.loc[:, ~labeled.columns.duplicated(keep="first")]
    labeled = _tz_normalize(labeled, "timestamp")
    _print_range("LABELS", labeled)

    # 5) 피처-라벨 병합 (asof 사다리 → 최근접 fallback)
    final_dataset = None
    if not features_df.empty and not labeled.empty:
        # 1차: asof 사다리
        final_dataset = _try_asof_merge(features_df, labeled, ASOF_TOLERANCES)

        # 2차: 여전히 0행이면 최근접 이웃 매칭으로 강제 결합
        if final_dataset is None or len(final_dataset) == 0 or final_dataset[["label","future_return"]].isna().all().all():
            print("⚠️ asof merge still empty or labels missing. Falling back to nearest-neighbor labeling...")

            feat_ts = features_df["timestamp"].to_numpy()
            lab_ts = labeled["timestamp"].to_numpy()

            if len(feat_ts) > 0 and len(lab_ts) > 0:
                pos = np.searchsorted(lab_ts, feat_ts)
                pos = np.clip(pos, 1, len(lab_ts) - 1)
                prev = lab_ts[pos - 1]
                nxt = lab_ts[pos]
                # 어느 쪽이 더 가까운지 선택
                pick = np.where((feat_ts - prev) <= (nxt - feat_ts), prev, nxt)
                nn_lab = labeled.set_index("timestamp").loc[pick, ["label", "future_return"]].reset_index(drop=True)
                final_dataset = pd.concat([features_df.reset_index(drop=True), nn_lab], axis=1)
                print(f"✅ fallback merged rows = {len(final_dataset)}")
            else:
                final_dataset = features_df.copy()
                final_dataset["label"] = np.nan
                final_dataset["future_return"] = np.nan
                print("⚠️ not enough timestamps for NN fallback; labels set to NaN.")
    else:
        # 피처 또는 라벨이 비어도 스키마를 유지하도록 처리
        final_dataset = features_df.copy()
        if "label" not in final_dataset.columns:
            final_dataset["label"] = np.nan
        if "future_return" not in final_dataset.columns:
            final_dataset["future_return"] = np.nan
        print("⚠️ features or labels empty; produced dataset with NaN labels.")

    # 6) 최종 저장
    final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated(keep="first")]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_dataset.to_parquet(OUT_PATH, index=False)

    print("🎉 DONE: saved →", OUT_PATH)
    print("shape:", final_dataset.shape)
    print("sample:")
    print(final_dataset.head(3))
