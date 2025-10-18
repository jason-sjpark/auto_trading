# scripts/compact_realtime_depth.py
#실행 방법
#1. WS 수집을 먼저 돌려서 리얼타임 depth가 쌓이게 함:
#python -m data_feed.binance_ws
# (원래 돌리던대로 그대로 실행)

#2. 하루가 쌓였거나 특정 날짜 구간이 있으면 압축 실행:
#python -m scripts.compact_realtime_depth --symbol BTCUSDT --date 2025-10-17

#3. 생성물 확인:
#python -m scripts.make_training_data --date 2025-10-17

#생성물 :
#data/processed/realtime_depth_best/BTCUSDT/2025-10-17.parquet

# scripts/compact_realtime_depth.py

# scripts/compact_realtime_depth.py
# L2 실시간 스냅샷 → 고정폭 LOB 피처 추출(스프레드/미드/마이크로프라이스/불균형/벽/갭)
# + 추가 피처: 깊이가중 중앙가(depth_wmid_LN), LOB slope, queue position proxy

import os, glob, argparse, ast
import numpy as np
import pandas as pd

def _to_dt_utc_naive(x):
    s = pd.to_datetime(x, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def _safe_parse_levels(arr):
    """arr: [[price, size], ...] or string → [[float, float], ...]"""
    if arr is None:
        return []
    if isinstance(arr, (str, bytes)):
        try:
            arr = ast.literal_eval(arr)
        except Exception:
            return []
    out = []
    try:
        for it in arr:
            if isinstance(it, (list, tuple)) and len(it) >= 1:
                p = float(it[0])
                q = float(it[1]) if len(it) > 1 else float("nan")
                out.append([p, q])
    except Exception:
        return []
    return out

def _sort_levels(bids, asks):
    """bids: desc by price, asks: asc by price"""
    bids = sorted(bids, key=lambda x: x[0], reverse=True)
    asks = sorted(asks, key=lambda x: x[0])
    return bids, asks

def _microprice(bb, ba, bb_sz, ba_sz):
    den = (bb_sz + ba_sz)
    if den <= 0 or not np.isfinite(den):
        return (bb + ba) / 2.0
    return (ba * bb_sz + bb * ba_sz) / den

def _level_gaps_in_ticks(levels, tick_size, side):
    """gap between best and 2nd level (in ticks)"""
    if len(levels) < 2 or tick_size <= 0:
        return np.nan
    if side == "bid":
        return (levels[0][0] - levels[1][0]) / tick_size
    else:
        return (levels[1][0] - levels[0][0]) / tick_size

def _cum_stats(levels, n):
    """상위 n레벨 누적 size, notional"""
    lv = levels[:n]
    if not lv:
        return 0.0, 0.0
    sizes = np.array([x[1] for x in lv], dtype=float)
    prices= np.array([x[0] for x in lv], dtype=float)
    sizes[np.isnan(sizes)] = 0.0
    notional = sizes * prices
    return float(np.sum(sizes)), float(np.sum(notional))

def _wall_in_levels(levels, n):
    """상위 n레벨 중 notional 최대 레벨 인덱스/값/총합"""
    lv = levels[:n]
    if not lv:
        return -1, 0.0, 0.0
    sizes = np.array([x[1] for x in lv], dtype=float)
    prices= np.array([x[0] for x in lv], dtype=float)
    sizes[np.isnan(sizes)] = 0.0
    notionals = sizes * prices
    idx = int(np.argmax(notionals))
    return idx, float(notionals[idx]), float(np.sum(notionals))

def _depth_weighted_mid(bids, asks, n):
    """양측 상위 n레벨 price*size 합을 전체 size로 나눈 깊이가중 mid"""
    b = bids[:n]; a = asks[:n]
    prices = []
    sizes  = []
    for p,q in b + a:
        if np.isfinite(p) and np.isfinite(q) and q > 0:
            prices.append(p); sizes.append(q)
    if not sizes:
        return np.nan
    sizes = np.asarray(sizes, dtype=float)
    prices= np.asarray(prices, dtype=float)
    return float(np.sum(prices * sizes) / (np.sum(sizes) + 1e-9))

def _lob_slope(levels, mid, tick_size, n):
    """
    각 측 상위 n레벨에 대해:
      x = mid로부터의 틱거리(양수), y = 누적 사이즈(레벨 1..i 합)
    선형회귀 y = a*x + b 의 a(기울기) 반환.
    """
    lv = levels[:n]
    if len(lv) < 2 or not np.isfinite(mid) or tick_size <= 0:
        return np.nan
    # 거리(틱)
    # bids: mid - price, asks: price - mid (호출 측에서 보장)
    dists = []
    cums  = []
    csum = 0.0
    for p, q in lv:
        q = 0.0 if not np.isfinite(q) else max(0.0, q)
        csum += q
        dist = abs(p - mid) / tick_size
        dists.append(dist)
        cums.append(csum)
    x = np.asarray(dists, dtype=float)
    y = np.asarray(cums,  dtype=float)
    if np.all(~np.isfinite(x)) or np.all(~np.isfinite(y)) or len(x) < 2:
        return np.nan
    # 결측 제거
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    try:
        a, b = np.polyfit(x[m], y[m], 1)
        return float(a)
    except Exception:
        return np.nan

def extract_row_features(bids_in, asks_in, n_levels=10, tick_size=0.1):
    bids = _safe_parse_levels(bids_in)
    asks = _safe_parse_levels(asks_in)
    if not bids and not asks:
        return {
            "best_bid": np.nan, "best_ask": np.nan,
            "best_bid_sz": np.nan, "best_ask_sz": np.nan,
            "spread": np.nan, "mid_price": np.nan, "microprice": np.nan,
            "l1_imbalance": np.nan,
            "bid_gap_ticks": np.nan, "ask_gap_ticks": np.nan, "liq_gap_ticks": np.nan,
            "cum_bid_sz_LN": 0.0, "cum_ask_sz_LN": 0.0,
            "cum_imbalance_LN": np.nan,
            "wall_level": -1, "wall_side": "none", "wall_notional": 0.0, "wall_strength": 0.0,
            "depth_wmid_LN": np.nan,
            "slope_bid_LN": np.nan, "slope_ask_LN": np.nan, "slope_imbalance_LN": np.nan,
            "best_bid_queue_share": np.nan, "best_ask_queue_share": np.nan,
            "queue_pressure_LN": np.nan,
        }

    bids, asks = _sort_levels(bids, asks)

    bb, bb_sz = (bids[0][0], bids[0][1]) if bids else (np.nan, np.nan)
    ba, ba_sz = (asks[0][0], asks[0][1]) if asks else (np.nan, np.nan)

    spread = (ba - bb) if np.isfinite(bb) and np.isfinite(ba) else np.nan
    mid    = (ba + bb)/2.0 if np.isfinite(bb) and np.isfinite(ba) else np.nan
    mprice = _microprice(bb, ba, bb_sz, ba_sz) if np.isfinite(bb) and np.isfinite(ba) else np.nan

    den1 = bb_sz + ba_sz
    l1_imb = (bb_sz - ba_sz)/den1 if den1 > 0 else np.nan

    bid_gap = _level_gaps_in_ticks(bids, tick_size, "bid")
    ask_gap = _level_gaps_in_ticks(asks, tick_size, "ask")
    liq_gap = np.nanmin([x for x in [bid_gap, ask_gap] if np.isfinite(x)]) if (np.isfinite(bid_gap) or np.isfinite(ask_gap)) else np.nan

    # 누적 N레벨
    cum_b_sz, cum_b_not = _cum_stats(bids, n_levels)
    cum_a_sz, cum_a_not = _cum_stats(asks, n_levels)
    denN = (cum_b_sz + cum_a_sz)
    cum_imb = (cum_b_sz - cum_a_sz)/denN if denN > 0 else np.nan

    # 벽 탐지
    bid_idx, bid_wall_not, bid_tot_not = _wall_in_levels(bids, n_levels)
    ask_idx, ask_wall_not, ask_tot_not = _wall_in_levels(asks, n_levels)
    if bid_wall_not >= ask_wall_not:
        wall_side, wall_level, wall_notional = "bid", bid_idx, bid_wall_not
    else:
        wall_side, wall_level, wall_notional = "ask", ask_idx, ask_wall_not
    tot_not = bid_tot_not + ask_tot_not
    if tot_not <= 0:
        tot_not = bid_tot_not if bid_tot_not > 0 else (ask_tot_not if ask_tot_not > 0 else 1.0)
    wall_strength = wall_notional / tot_not if tot_not > 0 else 0.0

    # 깊이가중 중앙가
    depth_wmid = _depth_weighted_mid(bids, asks, n_levels)

    # LOB slope (측별)
    # bids: x = (mid - price)/tick, asks: x = (price - mid)/tick
    slope_bid = _lob_slope([[p, q] for p, q in bids], mid, tick_size, n_levels) if np.isfinite(mid) else np.nan
    slope_ask = _lob_slope([[p, q] for p, q in asks], mid, tick_size, n_levels) if np.isfinite(mid) else np.nan
    slope_imb = (slope_bid - slope_ask) if (np.isfinite(slope_bid) and np.isfinite(slope_ask)) else np.nan

    # Queue position proxy
    best_bid_queue_share = (bb_sz / (cum_b_sz + 1e-9)) if cum_b_sz > 0 else np.nan
    best_ask_queue_share = (ba_sz / (cum_a_sz + 1e-9)) if cum_a_sz > 0 else np.nan
    queue_pressure = (bb_sz - ba_sz) / (denN + 1e-9) if denN > 0 else np.nan

    return {
        "best_bid": bb, "best_ask": ba,
        "best_bid_sz": bb_sz, "best_ask_sz": ba_sz,
        "spread": spread, "mid_price": mid, "microprice": mprice,
        "l1_imbalance": l1_imb,
        "bid_gap_ticks": bid_gap, "ask_gap_ticks": ask_gap, "liq_gap_ticks": liq_gap,
        "cum_bid_sz_LN": cum_b_sz, "cum_ask_sz_LN": cum_a_sz,
        "cum_imbalance_LN": cum_imb,
        "wall_level": wall_level, "wall_side": wall_side, "wall_notional": wall_notional, "wall_strength": wall_strength,
        "depth_wmid_LN": depth_wmid,
        "slope_bid_LN": slope_bid, "slope_ask_LN": slope_ask, "slope_imbalance_LN": slope_imb,
        "best_bid_queue_share": best_bid_queue_share, "best_ask_queue_share": best_ask_queue_share,
        "queue_pressure_LN": queue_pressure,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--in_root", default="data/realtime/depth")
    ap.add_argument("--out_root", default="data/processed/realtime_depth_best")  # 호환 경로 유지
    ap.add_argument("--n_levels", type=int, default=10, help="상위 레벨 수")
    ap.add_argument("--tick_size", type=float, default=0.1, help="심볼 틱사이즈(BTCUSDT=0.1 권장)")
    args = ap.parse_args()

    in_dir = os.path.join(args.in_root, args.symbol, args.date)
    files = sorted(glob.glob(os.path.join(in_dir, "*.parquet")))
    if not files:
        print(f"⚠️ no input realtime depth files: {in_dir}")
        return

    frames = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if "timestamp" not in df.columns:
                continue
            keep = [c for c in ["timestamp", "bids", "asks"] if c in df.columns]
            frames.append(df[keep].copy())
        except Exception as e:
            print(f"read fail: {p} → {e}")

    if not frames:
        print("⚠️ empty inputs")
        return

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = _to_dt_utc_naive(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")

    feats = raw.apply(
        lambda r: pd.Series(
            extract_row_features(r.get("bids"), r.get("asks"), n_levels=args.n_levels, tick_size=args.tick_size)
        ),
        axis=1
    )

    out = pd.concat([ raw[["timestamp"]].reset_index(drop=True), feats ], axis=1)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.join(args.out_root, args.symbol), exist_ok=True)
    out_path = os.path.join(args.out_root, args.symbol, f"{args.date}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"✅ saved: {out_path}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    main()


