###필독!!!
###실제로 돌릴때는 safemode를 false로 바꿔야함.
###연산이 오래걸려서 임시로 해놓은거임

# scripts/make_training_data.py
import os, glob, re, gc, sys, time
import pandas as pd
import numpy as np
from datetime import timedelta

from feature_engineering.feature_pipeline import BatchFeaturePipeline
from feature_engineering.external_features import build_external_features
from labeling.scalping_labeler import make_scalping_labels

# ================== 설정 ==================
OUT_PATH = "data/features/features_1s.parquet"

# 라벨
LABEL_HORIZON = 5
LABEL_THRESHOLD_PCT = 0.05  # (회의 사항: 스캘핑이면 bps 기준 권장)

# asof 병합 허용 오차 사다리
ASOF_TOLERANCES = [
    pd.Timedelta(milliseconds=500),
    pd.Timedelta(seconds=2),
    pd.Timedelta(seconds=5),
    pd.Timedelta(seconds=10),
]

# 🔸 세이프 모드: 처음엔 15분만 돌려서 빠르게 완주 → 이후 창을 늘려가며 확인
SAFE_MODE = True
SAFE_WINDOW_MIN = 15  # 15분

# 외부지표 윈도우 패딩(라벨/피처 창 주위로 여유)
EXTERNAL_PAD_HOURS = 12

# ================== 유틸 ==================
def _tz_normalize(df: pd.DataFrame, col="timestamp"):
    if df is None or df.empty:
        return pd.DataFrame(columns=[col])
    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    out = df.copy()
    out[col] = s
    out = out.dropna(subset=[col]).drop_duplicates(subset=[col]).sort_values(col)
    out.reset_index(drop=True, inplace=True)
    return out

def _print_range(name, df, col="timestamp"):
    if df is None or df.empty or col not in df.columns:
        print(f"⚠️ {name} EMPTY"); return
    print(f"⏱ {name} range: {df[col].min()} → {df[col].max()} ({len(df)} rows)")

def _glob_all(patterns):
    if isinstance(patterns, str): patterns = [patterns]
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    return sorted(files)

def _extract_day_from_path(path: str):
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", os.path.basename(path))
    if not m: return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"

def _latest_nonempty(patterns, min_rows: int = 1):
    files = _glob_all(patterns)
    for p in reversed(files):
        try:
            df = pd.read_parquet(p, columns=["timestamp"])
            if len(df) >= min_rows:
                return p
        except Exception:
            continue
    return files[-1] if files else None

def _pick_same_day_or_nearest(patterns, target_day: str, min_rows: int = 1):
    files = _glob_all(patterns)
    if not files: return None, "no_files"

    same = [p for p in files if _extract_day_from_path(p) == target_day]
    for p in reversed(same):
        try:
            if len(pd.read_parquet(p, columns=["timestamp"])) >= min_rows:
                return p, "same_day"
        except: pass

    dated = []
    for p in files:
        day = _extract_day_from_path(p)
        if day:
            dated.append((p, day))
    dated = sorted(dated, key=lambda x: x[1])
    if not dated: return None, "no_dated"

    target = pd.to_datetime(target_day)
    best_p, best_abs = None, None
    for p, day in dated:
        try:
            d = pd.to_datetime(day)
            diff = abs((d - target).days)
            rows = len(pd.read_parquet(p, columns=["timestamp"]))
            if rows < min_rows: continue
            if best_abs is None or diff < best_abs:
                best_p, best_abs = p, diff
        except:
            continue
    if best_p:
        return best_p, "nearest_day"
    return None, "no_match"

def _clip_by_window(df, start, end, col="timestamp"):
    if df is None or df.empty or col not in df.columns:
        return df
    m = (df[col] >= start) & (df[col] <= end)
    return df.loc[m].reset_index(drop=True)

def _try_asof_merge(left, right, tolerances):
    if left is None or left.empty: return left
    if right is None or right.empty: return left
    merged = left
    for tol in tolerances:
        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp", direction="nearest", tolerance=tol,
        )
        print(f"🧪 merge_asof with tolerance={tol}: result rows = {len(merged)}")
        if len(merged) > 0:
            print(f"✅ merged with tolerance={tol}")
            return merged
    print("⚠️ asof merge did not produce rows with given tolerances; returning last attempt")
    return merged

def _log_step(name, t0):
    t1 = time.perf_counter()
    print(f"⏳ {name} done in {t1 - t0:.2f}s")
    return t1

# ================== 메인 ==================
if __name__ == "__main__":
    T0 = time.perf_counter()
    print("📥 Loading source data...")

    # 1) 기준 OHLCV
    t = time.perf_counter()
    OHLCV_PATH = _latest_nonempty("data/processed/ohlcv_1s_*.parquet", min_rows=10)
    if not OHLCV_PATH:
        raise SystemExit("❌ no OHLCV found. Run backfill/make-ohlcv first.")
    ohlcv_day = _extract_day_from_path(OHLCV_PATH)
    ohlcv = _tz_normalize(pd.read_parquet(OHLCV_PATH))
    _print_range("OHLCV", ohlcv)
    t = _log_step("load OHLCV", t)

    # 2) 같은 날 TRADES/DEPTH
    TRADES_PATH, t_sel = _pick_same_day_or_nearest(
        ["data/raw/aggTrades/**/*.parquet", "data/realtime/aggTrades/**/*.parquet"],
        ohlcv_day, min_rows=10
    )
    DEPTH_PATH, d_sel  = _pick_same_day_or_nearest(
        ["data/raw/depth/**/*.parquet",      "data/realtime/depth/**/*.parquet"],
        ohlcv_day, min_rows=10
    )
    trades = _tz_normalize(pd.read_parquet(TRADES_PATH)) if TRADES_PATH else pd.DataFrame(columns=["timestamp","price","qty","side","trade_id","is_buyer_maker"])
    depth  = _tz_normalize(pd.read_parquet(DEPTH_PATH )) if DEPTH_PATH  else pd.DataFrame(columns=["timestamp","bids","asks"])
    _print_range("TRADES", trades)
    _print_range("DEPTH", depth)
    t = _log_step("load TRADES/DEPTH", t)

    # 3) 공통 시간창(교집합) + SAFE_WINDOW 적용
    t_min = max([x["timestamp"].min() for x in [ohlcv, trades, depth] if not x.empty])
    t_max = min([x["timestamp"].max() for x in [ohlcv, trades, depth] if not x.empty])
    if pd.isna(t_min) or pd.isna(t_max) or t_min >= t_max:
        t_min = ohlcv["timestamp"].min()
        t_max = ohlcv["timestamp"].max()
        print(f"⚠️ no overlap; use OHLCV window: {t_min} → {t_max}")

    if SAFE_MODE:
        t_max_safe = t_min + pd.Timedelta(minutes=SAFE_WINDOW_MIN)
        if t_max_safe < t_max:
            print(f"🔒 SAFE MODE: restricting window to {SAFE_WINDOW_MIN} minutes → {t_min} → {t_max_safe}")
            t_max = t_max_safe

    pad = timedelta(hours=EXTERNAL_PAD_HOURS)
    ext_min, ext_max = t_min - pad, t_max + pad

    ohlcv  = _clip_by_window(ohlcv,  t_min, t_max)
    trades = _clip_by_window(trades, t_min, t_max)
    depth  = _clip_by_window(depth,  t_min, t_max)
    _print_range("OHLCV(clipped)", ohlcv)
    _print_range("TRADES(clipped)", trades)
    _print_range("DEPTH(clipped)",  depth)
    t = _log_step("clip windows", t)

    # 4) 외부지표: 최신 파일 읽고 강제 슬라이스
    def _latest(pat):
        fs = _glob_all(pat); return fs[-1] if fs else None
    ext_inputs = {
        "funding":         pd.read_parquet(_latest("data/external/funding_rates_*.parquet")) if _latest("data/external/funding_rates_*.parquet") else pd.DataFrame(),
        "open_interest":   pd.read_parquet(_latest("data/external/open_interest_*.parquet")) if _latest("data/external/open_interest_*.parquet") else pd.DataFrame(),
        "long_short_ratio":pd.read_parquet(_latest("data/external/long_short_ratio_*.parquet")) if _latest("data/external/long_short_ratio_*.parquet") else pd.DataFrame(),
        "index_mark":      pd.read_parquet(_latest("data/external/index_mark_*.parquet")) if _latest("data/external/index_mark_*.parquet") else pd.DataFrame(),
        "liquidations":    pd.read_parquet(_latest("data/external/liquidations_*.parquet")) if _latest("data/external/liquidations_*.parquet") else pd.DataFrame(),
        "arbitrage":       pd.read_parquet(_latest("data/external/arbitrage_spreads_*.parquet")) if _latest("data/external/arbitrage_spreads_*.parquet") else pd.DataFrame(),
    }
    for k, v in list(ext_inputs.items()):
        if v is None or v.empty:
            ext_inputs[k] = pd.DataFrame(columns=["timestamp"])
        else:
            v = _tz_normalize(v)
            v = _clip_by_window(v, ext_min, ext_max)
            ext_inputs[k] = v
    t = _log_step("load external inputs", t)

    external_df = build_external_features(ext_inputs)
    if external_df is None or external_df.empty:
        external_df = pd.DataFrame(columns=["timestamp"])
    else:
        external_df = _tz_normalize(external_df)
        external_df = _clip_by_window(external_df, ext_min, ext_max)
    _print_range("EXTERNAL", external_df)
    t = _log_step("build external_df", t)

    # 5) 피처 생성 (여기가 무거우면 시간이 걸림 → 타이머 출력)
    pipe = BatchFeaturePipeline(timeframes=["0.5s","1s","5s"])
    t_build = time.perf_counter()
    features_df = pipe.build_features(ohlcv, trades, depth, external_df=external_df)
    t = _log_step("pipe.build_features", t_build)

    features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]
    features_df = _tz_normalize(features_df)
    _print_range("FEATURES", features_df)
    if features_df.empty: print("⚠️ FEATURES EMPTY")
    t = _log_step("post features", t)

    # 6) 라벨 생성
    t_lab = time.perf_counter()
    labeled = make_scalping_labels(ohlcv, horizon=LABEL_HORIZON, threshold_pct=LABEL_THRESHOLD_PCT)
    labeled = labeled[["timestamp","label","future_return"]]
    labeled = labeled.loc[:, ~labeled.columns.duplicated(keep="first")]
    labeled = _tz_normalize(labeled)
    labeled = _clip_by_window(labeled, ohlcv["timestamp"].min(), ohlcv["timestamp"].max())
    _print_range("LABELS", labeled)
    t = _log_step("make labels", t_lab)

    # 7) 병합
    t_merge = time.perf_counter()
    final_dataset = features_df.copy()
    if not features_df.empty and not labeled.empty:
        final_dataset = _try_asof_merge(features_df, labeled, ASOF_TOLERANCES)
        if final_dataset[["label","future_return"]].isna().all().all():
            print("⚠️ asof merge still empty or labels missing. Falling back to nearest-neighbor labeling...")
            feat_ts = features_df["timestamp"].to_numpy()
            lab_ts = labeled["timestamp"].to_numpy()
            if len(feat_ts) and len(lab_ts):
                pos = np.searchsorted(lab_ts, feat_ts)
                pos = np.clip(pos, 1, len(lab_ts)-1)
                prev, nxt = lab_ts[pos-1], lab_ts[pos]
                pick = np.where((feat_ts - prev) <= (nxt - feat_ts), prev, nxt)
                nn_lab = labeled.set_index("timestamp").loc[pick, ["label","future_return"]].reset_index(drop=True)
                final_dataset = pd.concat([features_df.reset_index(drop=True), nn_lab], axis=1)
                print(f"✅ fallback merged rows = {len(final_dataset)}")
            else:
                final_dataset["label"] = np.nan
                final_dataset["future_return"] = np.nan
    else:
        for c in ["label","future_return"]:
            if c not in final_dataset.columns: final_dataset[c] = np.nan
    t = _log_step("merge labels", t_merge)

    # 8) 저장
    final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated(keep="first")]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_dataset.to_parquet(OUT_PATH, index=False)
    print("🎉 DONE:", OUT_PATH)
    print("shape:", final_dataset.shape)
    print(final_dataset.head(3))

    T1 = time.perf_counter()
    print(f"✅ ALL DONE in {T1 - T0:.2f}s (SAFE_MODE={SAFE_MODE}, window≈{SAFE_WINDOW_MIN}min)")
