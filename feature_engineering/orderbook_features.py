import pandas as pd
import numpy as np
from typing import List, Dict, Any

EPS = 1e-12

def _to_dt_utc_naive(s) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

def _is_l2_schema(df: pd.DataFrame) -> bool:
    return {"bids", "asks"}.issubset(df.columns)

def _is_agg_schema(df: pd.DataFrame) -> bool:
    cols = df.columns.astype(str).tolist()
    return any(c.startswith("depth_pct_") for c in cols) or any(c.startswith("notional_pct_") for c in cols)

# ---------------- 공통 계산 유틸 (L2 한 행/스냅샷 공용) ----------------
def _best_prices_and_sizes(bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
    """
    bids/asks: [[price, size], ...] (정렬 가정이 틀려도 동작하도록 robust)
    - 베스트 호가 가격 파악 후, 그 가격대의 size를 합산해 top-of-book size로 사용
    """
    bb = max([float(b[0]) for b in bids]) if bids else np.nan
    ba = min([float(a[0]) for a in asks]) if asks else np.nan

    if np.isfinite(bb):
        top_bid_sz = sum(float(q) for p, q in bids if float(p) == bb)
    else:
        top_bid_sz = 0.0

    if np.isfinite(ba):
        top_ask_sz = sum(float(q) for p, q in asks if float(p) == ba)
    else:
        top_ask_sz = 0.0

    spread = float(ba - bb) if (np.isfinite(bb) and np.isfinite(ba)) else np.nan
    mid    = float((ba + bb) / 2.0) if (np.isfinite(bb) and np.isfinite(ba)) else np.nan

    return {
        "best_bid": bb, "best_ask": ba,
        "top_bid_sz": top_bid_sz, "top_ask_sz": top_ask_sz,
        "spread": spread, "mid_price": mid
    }

def _top5_wall_and_balance(bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
    """
    상위 5개 레벨 기준 벽 강도 및 밸런스
    - wall_strength: top5 중 최대 수량
    - depth_balance: top5 합 (bid - ask) / (bid + ask)
    """
    b_top5 = [float(q) for _, q in (bids[:5] if bids else [])]
    a_top5 = [float(q) for _, q in (asks[:5] if asks else [])]

    wall_b = max(b_top5) if b_top5 else 0.0
    wall_a = max(a_top5) if a_top5 else 0.0
    wall_strength = max(wall_b, wall_a)

    sum_b = sum(b_top5)
    sum_a = sum(a_top5)
    depth_balance = (sum_b - sum_a) / (abs(sum_b) + abs(sum_a) + EPS)

    return {"wall_strength": wall_strength, "depth_balance": depth_balance}

def _orderbook_imbalance_total(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    전체 볼륨 기반 불균형 (레벨 수 제한 없이)
    """
    bid_vol = sum(float(b[1]) for b in (bids or []))
    ask_vol = sum(float(a[1]) for a in (asks or []))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + EPS)

# ---------------- L2 (bids/asks: [[price,qty], ...]) ----------------
def _extract_l2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "timestamp","spread","mid_price","orderbook_imbalance",
            "liquidity_gap","wall_strength","depth_balance"
        ])

    d = df.copy()
    d["timestamp"] = _to_dt_utc_naive(d["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

    def _row(row):
        bids = row.get("bids", []) or []
        asks = row.get("asks", []) or []

        # 베스트 가격/사이즈
        best = _best_prices_and_sizes(bids, asks)

        # top-of-book 수량 차(유동성 갭)
        liq_gap = best["top_bid_sz"] - best["top_ask_sz"]

        # 상위5 벽/밸런스
        wall = _top5_wall_and_balance(bids, asks)

        ob_imb = _orderbook_imbalance_total(bids, asks)

        return pd.Series([
            best["spread"], best["mid_price"], ob_imb,
            liq_gap, wall["wall_strength"], wall["depth_balance"]
        ])

    out = d[["timestamp"]].copy()
    out[["spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]] = \
        d.apply(_row, axis=1)

    out = out.sort_values("timestamp").set_index("timestamp").resample("1s").last().reset_index()
    return out

# ---------------- 집계형 (depth_pct_±k, notional_pct_±k) ----------------
def _extract_agg(df: pd.DataFrame, K: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "timestamp","spread","mid_price","orderbook_imbalance",
            "liquidity_gap","wall_strength","depth_balance"
        ])

    d = df.copy()
    d["timestamp"] = _to_dt_utc_naive(d["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

    out = d[["timestamp"]].copy()

    cols = d.columns.astype(str).tolist()
    dep_pos = [c for c in cols if c.startswith("depth_pct_") and not c.startswith("depth_pct_-")]
    dep_neg = [c for c in cols if c.startswith("depth_pct_-")]
    not_pos = [c for c in cols if c.startswith("notional_pct_") and not c.startswith("notional_pct_-")]
    not_neg = [c for c in cols if c.startswith("notional_pct_-")]

    def _pick(seq: List[str], sign: str, kind: str) -> List[str]:
        base = "depth_pct_" if kind == "depth" else "notional_pct_"
        out2 = []
        for k in range(1, K+1):
            name = f"{base}{sign}{k}"
            if name in d.columns:
                out2.append(name)
        return out2

    dep_posK = _pick(dep_pos or ["depth_pct_1"], "+", "depth")
    dep_negK = _pick(dep_neg or ["depth_pct_-1"], "-", "depth")
    not_posK = _pick(not_pos or ["notional_pct_1"], "+", "notional")
    not_negK = _pick(not_neg or ["notional_pct_-1"], "-", "notional")

    buy_sum  = d[dep_posK].sum(axis=1) if dep_posK else pd.Series(0.0, index=d.index)
    sell_sum = d[dep_negK].sum(axis=1) if dep_negK else pd.Series(0.0, index=d.index)
    out["depth_balance"] = (buy_sum - sell_sum) / (buy_sum.abs() + sell_sum.abs() + EPS)

    pos_max = d[not_posK].abs().max(axis=1) if not_posK else pd.Series(0.0, index=d.index)
    neg_max = d[not_negK].abs().max(axis=1) if not_negK else pd.Series(0.0, index=d.index)
    out["wall_strength"] = np.maximum(pos_max, neg_max)

    b1 = d.get("depth_pct_1"); s1 = d.get("depth_pct_-1")
    out["liquidity_gap"] = (b1.fillna(0.0) - s1.fillna(0.0)) if (b1 is not None and s1 is not None) else 0.0

    out["spread"] = np.nan
    out["mid_price"] = np.nan
    out["orderbook_imbalance"] = out["depth_balance"]

    out = out.sort_values("timestamp").set_index("timestamp").resample("1s").last().reset_index()
    return out

# ---------------- 공개 API ----------------
def extract_orderbook_features(depth_df: pd.DataFrame) -> pd.DataFrame:
    """
    입력:
      - L2 스냅샷: columns=[timestamp, bids, asks], 각 행당 상위호가 배열
      - 집계형:    columns include depth_pct_±k, notional_pct_±k
    출력(1s 그리드):
      [timestamp, spread, mid_price, orderbook_imbalance, liquidity_gap, wall_strength, depth_balance]
    """
    if depth_df is None or getattr(depth_df, "empty", True):
        return pd.DataFrame(columns=[
            "timestamp","spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"
        ])

    if _is_l2_schema(depth_df):
        return _extract_l2(depth_df)
    if _is_agg_schema(depth_df):
        return _extract_agg(depth_df)

    # 알 수 없는 스키마: timestamp만 통과
    out = depth_df[["timestamp"]].copy()
    for c in ["spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]:
        out[c] = np.nan
    out["timestamp"] = _to_dt_utc_naive(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")\
             .set_index("timestamp").resample("1s").last().reset_index()
    return out

def extract_orderbook_features_snapshot(orderbook_snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    단일 시점 L2 스냅샷(dict)에서 즉시 계산:
      반환 dict keys:
        spread, mid_price, orderbook_imbalance, liquidity_gap, wall_strength, depth_balance
    """
    bids = orderbook_snapshot.get("bids") or []
    asks = orderbook_snapshot.get("asks") or []

    # 베스트 가격/사이즈 + mid/spread
    best = _best_prices_and_sizes(bids, asks)

    # 상위5 벽/밸런스
    wall = _top5_wall_and_balance(bids, asks)

    # 전체 불균형
    ob_imb = _orderbook_imbalance_total(bids, asks)

    liq_gap = best["top_bid_sz"] - best["top_ask_sz"]

    feats = {
        "spread": best["spread"],
        "mid_price": best["mid_price"],
        "orderbook_imbalance": ob_imb,
        "liquidity_gap": float(liq_gap),
        "wall_strength": float(wall["wall_strength"]),
        "depth_balance": float(wall["depth_balance"]),
    }

    # NaN/inf 방지
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0
    return feats
