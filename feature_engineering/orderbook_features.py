import pandas as pd
import numpy as np
from typing import List

EPS = 1e-12

def _to_dt_utc_naive(s) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

def _is_l2_schema(df: pd.DataFrame) -> bool:
    return {"bids", "asks"}.issubset(df.columns)

def _is_agg_schema(df: pd.DataFrame) -> bool:
    cols = df.columns.astype(str).tolist()
    return any(c.startswith("depth_pct_") for c in cols) or any(c.startswith("notional_pct_") for c in cols)

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
        # 배열/리스트 섞여도 robust
        try:
            bb = max([float(b[0]) for b in bids]) if bids else np.nan
            ba = min([float(a[0]) for a in asks]) if asks else np.nan
        except Exception:
            bb, ba = np.nan, np.nan
        spread = float(ba - bb) if np.isfinite(bb) and np.isfinite(ba) else np.nan
        mid = float((ba + bb) / 2.0) if np.isfinite(bb) and np.isfinite(ba) else np.nan

        bid_vol = sum([float(b[1]) for b in bids]) if bids else 0.0
        ask_vol = sum([float(a[1]) for a in asks]) if asks else 0.0
        ob_imb  = (bid_vol - ask_vol) / (bid_vol + ask_vol + EPS)

        tbq = float(bids[0][1]) if bids else 0.0
        taq = float(asks[0][1]) if asks else 0.0
        liq_gap = tbq - taq

        wall_b = max([float(q) for _, q in bids[:5]], default=0.0)
        wall_a = max([float(q) for _, q in asks[:5]], default=0.0)
        wall_strength = max(wall_b, wall_a)

        sum_b = sum([float(q) for _, q in bids[:5]])
        sum_a = sum([float(q) for _, q in asks[:5]])
        depth_balance = (sum_b - sum_a) / (sum_b + sum_a + EPS)

        return pd.Series([spread, mid, ob_imb, liq_gap, wall_strength, depth_balance])

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

    # 컬럼 이름 정규화
    cols = d.columns.astype(str).tolist()
    dep_pos = [c for c in cols if c.startswith("depth_pct_") and not c.startswith("depth_pct_-")]
    dep_neg = [c for c in cols if c.startswith("depth_pct_-")]
    not_pos = [c for c in cols if c.startswith("notional_pct_") and not c.startswith("notional_pct_-")]
    not_neg = [c for c in cols if c.startswith("notional_pct_-")]

    # 깊이 밴드 합계(±1~±K%)
    def _pick(seq: List[str], sign: str) -> List[str]:
        out2 = []
        for k in range(1, K+1):
            name = f"{'depth_pct_' if 'depth' in seq[0] else 'notional_pct_'}{sign}{k}"
            if name in d.columns:
                out2.append(name)
        return out2

    dep_posK = _pick(dep_pos or ["depth_pct_1"], "+")
    dep_negK = _pick(dep_neg or ["depth_pct_-1"], "-")
    not_posK = _pick(not_pos or ["notional_pct_1"], "+")
    not_negK = _pick(not_neg or ["notional_pct_-1"], "-")

    buy_sum  = d[dep_posK].sum(axis=1) if dep_posK else pd.Series(0.0, index=d.index)
    sell_sum = d[dep_negK].sum(axis=1) if dep_negK else pd.Series(0.0, index=d.index)
    out["depth_balance"] = (buy_sum - sell_sum) / (buy_sum.abs() + sell_sum.abs() + EPS)

    # notional의 최대값 기반 벽 세기 (pandas<1.5 호환: min_count 미사용)
    pos_max = d[not_posK].abs().max(axis=1) if not_posK else pd.Series(0.0, index=d.index)
    neg_max = d[not_negK].abs().max(axis=1) if not_negK else pd.Series(0.0, index=d.index)
    out["wall_strength"] = np.maximum(pos_max, neg_max)

    # ±1% 인접 유동성 차
    b1 = d.get("depth_pct_1"); s1 = d.get("depth_pct_-1")
    out["liquidity_gap"] = (b1.fillna(0.0) - s1.fillna(0.0)) if (b1 is not None and s1 is not None) else 0.0

    # 집계형에는 없는 값 → NaN 유지
    out["spread"] = np.nan
    out["mid_price"] = np.nan
    out["orderbook_imbalance"] = out["depth_balance"]

    out = out.sort_values("timestamp").set_index("timestamp").resample("1s").last().reset_index()
    return out

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
