# feature_engineering/orderbook_features.py
import pandas as pd
import numpy as np

EPS = 1e-12

def _is_l2_schema(df: pd.DataFrame) -> bool:
    return {"bids","asks"}.issubset(df.columns)

def _is_agg_schema(df: pd.DataFrame) -> bool:
    return any(c.startswith("depth_pct_") for c in df.columns) or any(c.startswith("notional_pct_") for c in df.columns)

# ---------- L2 스키마 ----------
def _extract_l2(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns = ['timestamp','bids','asks'] with bids/asks = list[[price, qty], ...]
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"])

    def _row_calc(row):
        bids = row.get("bids", [])
        asks = row.get("asks", [])
        # 리스트 형태 보장
        if isinstance(bids, (np.ndarray,)): bids = bids.tolist()
        if isinstance(asks, (np.ndarray,)): asks = asks.tolist()
        try:
            best_bid = max([b[0] for b in bids]) if bids else np.nan
            best_ask = min([a[0] for a in asks]) if asks else np.nan
            spread = float(best_ask - best_bid) if (np.isfinite(best_ask) and np.isfinite(best_bid)) else np.nan
            mid = float((best_ask + best_bid) / 2.0) if (np.isfinite(best_ask) and np.isfinite(best_bid)) else np.nan

            bid_vol = sum([b[1] for b in bids]) if bids else 0.0
            ask_vol = sum([a[1] for a in asks]) if asks else 0.0
            ob_imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + EPS)

            # 간단 유동성 갭: 최상단 호가량 차이
            top_bid_qty = bids[0][1] if bids else 0.0
            top_ask_qty = asks[0][1] if asks else 0.0
            liq_gap = float(top_bid_qty - top_ask_qty)

            # 벽 강도: 상단 5호가 기준 최대 큼직한 물량
            wall_b = max([q for _, q in bids[:5]], default=0.0)
            wall_a = max([q for _, q in asks[:5]], default=0.0)
            wall_strength = float(max(wall_b, wall_a))

            # depth balance: 상단 5호가 누계 비교
            sum_b = sum([q for _, q in bids[:5]])
            sum_a = sum([q for _, q in asks[:5]])
            depth_balance = (sum_b - sum_a) / (sum_b + sum_a + EPS)

            return pd.Series([spread, mid, ob_imb, liq_gap, wall_strength, depth_balance])
        except Exception:
            return pd.Series([np.nan]*6)

    out = df.copy()
    out[["spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]] = out.apply(_row_calc, axis=1)
    return out[["timestamp","spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]]

# ---------- 집계 스키마 ----------
def _extract_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns include depth_pct_{±1..±5}, notional_pct_{±1..±5}
    퍼센티지 밴드별 상대/절대 유동성 대용지표 구성.
    spread/mid_price는 계산 불가 → NaN.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"])

    out = df[["timestamp"]].copy()

    # depth balance
    buy_cols  = [c for c in df.columns if c.startswith("depth_pct_") and not c.startswith("depth_pct_-")]
    sell_cols = [c for c in df.columns if c.startswith("depth_pct_-")]
    buy_sum   = df[buy_cols].sum(axis=1, min_count=1) if buy_cols else 0.0
    sell_sum  = df[sell_cols].sum(axis=1, min_count=1) if sell_cols else 0.0
    out["depth_balance"] = (buy_sum - sell_sum) / (np.abs(buy_sum) + np.abs(sell_sum) + EPS)

    # liquidity_gap: ±1% 구간의 notional 차
    n_buy1  = df.get("notional_pct_1")
    n_sell1 = df.get("notional_pct_-1")
    out["liquidity_gap"] = (n_buy1.fillna(0) - n_sell1.fillna(0)) if (n_buy1 is not None and n_sell1 is not None) else 0.0

    # wall_strength: ±1..5% 구간의 notional 중 절댓값 최대
    band_pos = [c for c in df.columns if c.startswith("notional_pct_") and not c.startswith("notional_pct_-")]
    band_neg = [c for c in df.columns if c.startswith("notional_pct_-")]
    pos_max = df[band_pos].abs().max(axis=1, min_count=1) if band_pos else 0.0
    neg_max = df[band_neg].abs().max(axis=1, min_count=1) if band_neg else 0.0
    out["wall_strength"] = np.maximum(pos_max, neg_max)

    # 집계형에서는 주문가격이 없어 spread/mid_price/imbalance는 대체 불가 → NaN/0 처리
    out["spread"] = np.nan
    out["mid_price"] = np.nan

    # orderbook_imbalance: 심화 계산 불가 → depth_balance로 대리 (명시)
    out["orderbook_imbalance"] = out["depth_balance"]

    return out

def extract_orderbook_features(depth_df: pd.DataFrame) -> pd.DataFrame:
    if depth_df is None or depth_df.empty:
        return pd.DataFrame(columns=["timestamp","spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"])
    if _is_l2_schema(depth_df):
        out = _extract_l2(depth_df)
    elif _is_agg_schema(depth_df):
        out = _extract_agg(depth_df)
    else:
        # 알 수 없는 스키마: 타임스탬프만 유지
        out = depth_df[["timestamp"]].copy()
        for c in ["spread","mid_price","orderbook_imbalance","liquidity_gap","wall_strength","depth_balance"]:
            out[c] = np.nan
    # 1초 그리드에 맞추기 (최종 병합을 위해)
    out = out.sort_values("timestamp").set_index("timestamp").resample("1s").last().reset_index()
    return out
