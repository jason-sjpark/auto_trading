# scripts/make_training_data.py
"""
통합 학습 데이터 생성 파이프라인 (전역 safemode 스위치)

- 입력:
  * OHLCV(1s): data/processed/ohlcv_1s_YYYYMMDD.parquet
  * Trades raw: data/raw/aggTrades/{SYMBOL}/YYYY-MM-DD.parquet
  * Depth raw : data/raw/depth/{SYMBOL}/YYYY-MM-DD.parquet  (집계형 bookDepth 피벗 완료본)
  * External  : data/external/*.parquet (funding, oi, lsr, index/mark, arb)
  * Liquidations(옵션): data/realtime/liquidations/{SYMBOL}/**/*.parquet

- 출력:
  * data/features/features_1s.parquet

- 라벨:
  * horizon_s=5, threshold_bp=10 (±0.10%), neutral 포함

- safemode (전역 변수):
  * True  → 15분 클립으로 빠른 검증
  * False → 전체 구간
"""

import os
import sys
import glob
import time
import argparse
from datetime import timedelta

import pandas as pd
import numpy as np

# ===== 전역 스위치 =====
safemode: bool = False   # ← 여기만 True/False 로 바꿔서 사용

# --- 로컬 모듈 ---
from feature_engineering.feature_pipeline import BatchFeaturePipeline
from feature_engineering.external_features import build_external_features
from labeling.scalping_labeler import make_scalping_labels


# -------------------------
# 유틸
# -------------------------
def _to_dt_utc_naive(s) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

def _print_range(name: str, df: pd.DataFrame, col: str = "timestamp"):
    if df is None or df.empty or col not in df.columns:
        print(f"⚠️ {name} EMPTY")
        return
    print(f"⏱ {name} range: {df[col].min()} → {df[col].max()} ({len(df)} rows)")

def _latest_path(pattern: str) -> str:
    paths = glob.glob(pattern)
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]

def _detect_ohlcv(date: str = "", base="data/processed") -> str:
    if date:
        return os.path.join(base, f"ohlcv_1s_{date.replace('-', '')}.parquet")
    return _latest_path(os.path.join(base, "ohlcv_1s_*.parquet"))

def _detect_trades(symbol: str, date: str, base="data/raw/aggTrades") -> str:
    return os.path.join(base, symbol, f"{date}.parquet")

def _detect_depth(symbol: str, date: str, base="data/raw/depth") -> str:
    return os.path.join(base, symbol, f"{date}.parquet")

def _load_parquet_safe(path: str, required_cols=None) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if required_cols:
            for c in required_cols:
                if c not in df.columns:
                    df[c] = np.nan
        return df
    except Exception as e:
        print(f"⚠️ read parquet failed: {path} → {e}")
        return pd.DataFrame()

def _load_external_inputs() -> dict:
    base = "data/external"
    out = {}
    out["funding"]          = _load_parquet_safe(_latest_path(os.path.join(base, "funding_rates_*.parquet")))
    out["open_interest"]    = _load_parquet_safe(_latest_path(os.path.join(base, "open_interest_*.parquet")))
    out["long_short_ratio"] = _load_parquet_safe(_latest_path(os.path.join(base, "long_short_ratio_*.parquet")))
    out["index_mark"]       = _load_parquet_safe(_latest_path(os.path.join(base, "index_mark_*.parquet")))
    out["arbitrage"]        = _load_parquet_safe(_latest_path(os.path.join(base, "arbitrage_spreads_*.parquet")))
    return out

def _load_liquidations_between(symbol="BTCUSDT", start=None, end=None) -> pd.DataFrame:
    """실시간 청산 스냅샷(WS 저장분) 로딩"""
    root = os.path.join("data", "realtime", "liquidations", symbol)
    frames = []
    for path in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True):
        try:
            df = pd.read_parquet(path)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = _to_dt_utc_naive(df["timestamp"])
            if start is not None:
                df = df[df["timestamp"] >= start]
            if end is not None:
                df = df[df["timestamp"] <= end]
            if not df.empty:
                cols = [c for c in ["timestamp", "side", "price", "qty", "notional"] if c in df.columns]
                frames.append(df[cols])
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["timestamp", "side", "price", "qty", "notional"])
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return out

def _detect_rt_best(symbol: str, date: str, base="data/processed/realtime_depth_best") -> str:
    # date: "YYYY-MM-DD"
    return os.path.join(base, symbol, f"{date}.parquet")

def _load_rt_best(symbol: str, date: str):
    p = _detect_rt_best(symbol, date)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = _to_dt_utc_naive(df["timestamp"])
    keep = [c for c in [
        "timestamp","best_bid","best_ask","best_bid_sz","best_ask_sz","spread","mid_price","ob_imbalance"
    ] if c in df.columns]
    return df[keep].dropna(subset=["timestamp"]).sort_values("timestamp")


# -------------------------
# 메인
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (미지정 시 최신 OHLCV 날짜로 추정)")
    ap.add_argument("--out", default="data/features/features_1s.parquet")
    ap.add_argument("--horizon_s", type=int, default=5)
    ap.add_argument("--threshold_bp", type=float, default=10.0)
    args = ap.parse_args()

    symbol = args.symbol
    horizon_s = int(args.horizon_s)
    threshold_bp = float(args.threshold_bp)

    # --- 1) 입력 경로 탐색/로드 ---
    t0 = time.time()
    ohlcv_path = _detect_ohlcv(args.date)
    if not ohlcv_path:
        print("❌ OHLCV 파일을 찾지 못했습니다. 먼저 backfill 또는 실시간 수집으로 1s OHLCV를 생성하세요.")
        sys.exit(1)

    # date 추출
    if args.date:
        date_str = args.date
    else:
        base = os.path.basename(ohlcv_path)  # ohlcv_1s_YYYYMMDD.parquet
        date_str = base.replace("ohlcv_1s_", "").replace(".parquet", "")
        if len(date_str) == 8:
            date_str = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"

    trades_path = _detect_trades(symbol, date_str)
    depth_path  = _detect_depth(symbol, date_str)

    print("📥 Loading source data...")
    ohlcv = _load_parquet_safe(ohlcv_path, required_cols=["timestamp", "open", "high", "low", "close", "volume"])
    ohlcv["timestamp"] = _to_dt_utc_naive(ohlcv["timestamp"])
    ohlcv = ohlcv.dropna(subset=["timestamp"]).sort_values("timestamp")
    _print_range("OHLCV", ohlcv)
    print(f"⏳ load OHLCV done in {time.time()-t0:.2f}s")

    t1 = time.time()
    trades = _load_parquet_safe(trades_path)
    if not trades.empty:
        # 표준화: timestamp/price/qty/side
        ts_col = "timestamp" if "timestamp" in trades.columns else ("time" if "time" in trades.columns else None)
        if ts_col:
            trades["timestamp"] = trades[ts_col]
        trades["timestamp"] = _to_dt_utc_naive(trades["timestamp"])
        trades["price"] = pd.to_numeric(trades.get("price", 0), errors="coerce")
        trades["qty"] = pd.to_numeric(trades.get("qty", 0), errors="coerce")
        if "side" not in trades.columns and "is_buyer_maker" in trades.columns:
            trades["side"] = np.where(trades["is_buyer_maker"], "sell", "buy")
        trades["side"] = trades.get("side", "buy").astype(str).str.lower()
        trades = trades.dropna(subset=["timestamp"]).sort_values("timestamp")
    _print_range("TRADES", trades)

    # --- DEPTH: ① 집계 depth 로드 → ② 실시간 BEST 있으면 교체 → ③ 둘 다 없으면 placeholder ---
    depth = _load_parquet_safe(depth_path)
    if not depth.empty:
        if "timestamp" in depth.columns:
            depth["timestamp"] = _to_dt_utc_naive(depth["timestamp"])
            depth = depth.dropna(subset=["timestamp"]).sort_values("timestamp")

    rt_best = _load_rt_best(symbol, date_str)
    if not rt_best.empty:
        depth = rt_best  # 진짜 spread/mid 확보용
        print("🔁 Using realtime BEST depth instead of aggregated bookDepth")

    if depth is None or depth.empty or "timestamp" not in depth.columns:
        # placeholder: OHLCV 타임스탬프에 정렬 (피처 파이프라인이 기대하는 컬럼 뼈대 제공)
        placeholder = pd.DataFrame({"timestamp": ohlcv["timestamp"].copy()})
        for c in ["best_bid","best_ask","best_bid_sz","best_ask_sz","spread","mid_price","ob_imbalance"]:
            placeholder[c] = np.nan
        depth = placeholder
        print("⚠️ No depth sources found. Created placeholder depth aligned to OHLCV.")

    _print_range("DEPTH(final)", depth)
    print(f"⏳ load TRADES/DEPTH done in {time.time()-t1:.2f}s")

    # --- 1.5) safemode: 15분 클립 ---
    # 공통 구간 탐색 시, 각 DF가 비어있을 수 있으므로 안전하게 계산
    cstart_candidates = []
    cend_candidates = []

    if not ohlcv.empty:
        cstart_candidates.append(ohlcv["timestamp"].min())
        cend_candidates.append(ohlcv["timestamp"].max())
    if not trades.empty:
        cstart_candidates.append(trades["timestamp"].min())
        cend_candidates.append(trades["timestamp"].max())
    if not depth.empty:
        cstart_candidates.append(depth["timestamp"].min())
        cend_candidates.append(depth["timestamp"].max())

    if cstart_candidates and cend_candidates:
        clip_start = max(cstart_candidates)
        clip_end   = min(cend_candidates)
    else:
        print("❌ 입력 데이터가 모두 비어 있습니다.")
        sys.exit(1)

    if pd.isna(clip_start) or pd.isna(clip_end) or clip_end <= clip_start:
        if not ohlcv.empty:
            clip_start = ohlcv["timestamp"].min()
            clip_end   = clip_start + timedelta(minutes=15)
        else:
            print("❌ 겹치는 시간 구간을 찾지 못했습니다.")
            sys.exit(1)

    if safemode:
        _clip_end = min(clip_end, clip_start + timedelta(minutes=15))
        print(f"🔒 SAFE MODE: restricting window to 15 minutes → {clip_start} → {_clip_end}")
        ohlcv  = ohlcv[(ohlcv["timestamp"] >= clip_start) & (ohlcv["timestamp"] <= _clip_end)]
        trades = trades[(trades["timestamp"] >= clip_start) & (trades["timestamp"] <= _clip_end)]
        depth  = depth[(depth["timestamp"]  >= clip_start) & (depth["timestamp"]  <= _clip_end)]
        _print_range("OHLCV(clipped)", ohlcv)
        _print_range("TRADES(clipped)", trades)
        _print_range("DEPTH(clipped)", depth)
        print("⏳ clip windows done in {:.2f}s".format(time.time()-t1))

    # --- 2) 외부지표 + 리퀴데이션 통합 ---
    t2 = time.time()
    ext_inputs = _load_external_inputs()
    liq_df = _load_liquidations_between(symbol=symbol, start=ohlcv["timestamp"].min(), end=ohlcv["timestamp"].max())
    if not liq_df.empty:
        ext_inputs["liquidations"] = liq_df

    print("⏳ load external inputs done in {:.2f}s".format(time.time()-t2))
    ext_all = []
    for k, v in ext_inputs.items():
        if v is not None and not v.empty and "timestamp" in v.columns:
            ext_all.append(v[["timestamp"]])
    if ext_all:
        tmp = pd.concat(ext_all, ignore_index=True)
        tmp["timestamp"] = _to_dt_utc_naive(tmp["timestamp"])
        if not tmp.empty:
            print("⏱ EXTERNAL range: {} → {} ({} rows)".format(tmp["timestamp"].min(), tmp["timestamp"].max(), len(tmp)))
    external_df = build_external_features(
        ext_inputs,
        liq_windows_s=(30, 60),
        big_liq_notional=100000.0
    )
    if external_df is None:
        external_df = pd.DataFrame(columns=["timestamp"])
    print("⏳ build external_df done in {:.2f}s".format(time.time()-t2))

    # --- 3) 피처 생성 ---
    t3 = time.time()
    pipe = BatchFeaturePipeline(timeframes=["0.5s", "1s", "5s"])
    try:
        features_df = pipe.build_features(
            ohlcv_df=ohlcv,
            trades_df=trades,
            depth_df=depth,
            external_df=external_df
        )
    except Exception as e:
        print(f"❌ pipe.build_features failed: {e}")
        raise

    if features_df is None or features_df.empty:
        print("⚠️ [build_features] 결과 DataFrame이 비어 있습니다.")
        print("ohlcv_df:", ohlcv.shape if ohlcv is not None else None)
        print("trades_df:", trades.shape if trades is not None else None)
        print("depth_df:", depth.shape if depth is not None else None)
    else:
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]
        if "timestamp" in features_df.columns:
            features_df["timestamp"] = _to_dt_utc_naive(features_df["timestamp"])
            features_df = features_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        features_df = features_df.ffill().bfill()

    print("✅ [build_features] shape={}".format(features_df.shape if features_df is not None else None))
    print("⏳ pipe.build_features done in {:.2f}s".format(time.time()-t3))
    _print_range("FEATURES", features_df)
    print("⏳ post features done in {:.2f}s".format(time.time()-t3))

    # --- 4) 라벨 생성 ---
    t4 = time.time()
    labeled = make_scalping_labels(
        ohlcv_df=ohlcv,
        horizon_s=horizon_s,
        threshold_bp=threshold_bp,
        neutral_band=True,
        price_col="close",
    )
    labeled = labeled[["timestamp", "label", "future_return"]]
    labeled["timestamp"] = _to_dt_utc_naive(labeled["timestamp"])
    labeled = labeled.dropna(subset=["timestamp"]).sort_values("timestamp")
    _print_range("LABELS", labeled)
    print("⏳ make labels done in {:.2f}s".format(time.time()-t4))

    # --- 5) 병합 ---
    t5 = time.time()
    if features_df is None or features_df.empty:
        final_dataset = pd.DataFrame(columns=["timestamp", "label", "future_return"])
    else:
        f = features_df.sort_values("timestamp")
        l = labeled.sort_values("timestamp")
        final_dataset = pd.merge_asof(
            f, l, on="timestamp", direction="nearest",
            tolerance=pd.Timedelta(milliseconds=500)
        )
        if final_dataset is None or final_dataset.empty:
            print("⚠️ merge_asof 결과가 비어 fallback nearest-neighbor labeling...")
            lab_ts = l["timestamp"].to_numpy()
            feat_ts = f["timestamp"].to_numpy()
            if len(lab_ts) == 0 or len(feat_ts) == 0:
                final_dataset = f.copy()
                final_dataset["label"] = 0
                final_dataset["future_return"] = 0.0
            else:
                pos = np.searchsorted(lab_ts, feat_ts)
                pos = np.clip(pos, 1, len(lab_ts) - 1)
                prev = lab_ts[pos - 1]
                next = lab_ts[pos]
                pick = np.where((feat_ts - prev) <= (next - feat_ts), prev, next)
                nn_lab = l.set_index("timestamp").loc[pick, ["label", "future_return"]].reset_index(drop=True)
                final_dataset = pd.concat([f.reset_index(drop=True), nn_lab], axis=1)

    final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated(keep="first")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    final_dataset.to_parquet(args.out, index=False)


    # --- Quick quality report (optional) ---
    critical_cols = ["spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]
    null_rates = final_dataset[critical_cols].isna().mean().sort_values(ascending=False)
    print("🧪 null rates (critical):")
    print((null_rates*100).round(2).astype(str) + "%")

    print("⏳ merge labels done in {:.2f}s".format(time.time()-t5))
    print(f"🎉 DONE: {args.out}")
    print("shape:", final_dataset.shape)
    print(final_dataset.head(3))
    print("✅ ALL DONE in {:.2f}s (safemode={})".format(time.time()-t0, safemode))


if __name__ == "__main__":
    main()
