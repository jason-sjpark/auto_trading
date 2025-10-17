import pandas as pd
import numpy as np
from typing import Dict, List

def _to_levels(xs) -> np.ndarray:
    """
    xs: [[price, size], ...] or np.array
    returns: np.array shape (n,2) float
    """
    if xs is None: return np.empty((0,2), dtype=float)
    arr = np.asarray(xs, dtype=object)
    # handle ragged
    rows = []
    for row in arr:
        try:
            p, q = float(row[0]), float(row[1])
            rows.append([p,q])
        except Exception:
            continue
    return np.array(rows, dtype=float) if rows else np.empty((0,2), dtype=float)

def calc_spread(bids: np.ndarray, asks: np.ndarray) -> float:
    if bids.shape[0] == 0 or asks.shape[0] == 0: return 0.0
    best_bid = bids[:,0].max()
    best_ask = asks[:,0].min()
    return max(best_ask - best_bid, 0.0)

def calc_mid_price(bids: np.ndarray, asks: np.ndarray) -> float:
    if bids.shape[0] == 0 or asks.shape[0] == 0: return 0.0
    best_bid = bids[:,0].max()
    best_ask = asks[:,0].min()
    return (best_bid + best_ask) / 2.0

def calc_orderbook_imbalance(bids: np.ndarray, asks: np.ndarray, depth: int = 5) -> float:
    b = bids[:depth,1].sum() if bids.shape[0] else 0.0
    a = asks[:depth,1].sum() if asks.shape[0] else 0.0
    return (b - a) / (b + a + 1e-9)

def calc_liquidity_gap(bids: np.ndarray, asks: np.ndarray, depth: int = 5) -> float:
    # price distance between top levels as proxy of liquidity gap
    if bids.shape[0] == 0 or asks.shape[0] == 0: return 0.0
    bb = bids[:depth,0]
    aa = asks[:depth,0]
    return float(np.mean(aa) - np.mean(bb))

def calc_wall_strength(levels: np.ndarray, depth: int = 10, threshold: float = 3.0) -> float:
    # 큰 호가벽: 상위 depth 구간에서 중앙값 대비 몇 배 큰 size가 있는지 count
    if levels.shape[0] == 0: return 0.0
    sizes = levels[:depth,1]
    if len(sizes) == 0: return 0.0
    med = np.median(sizes) + 1e-9
    return float(np.sum(sizes >= threshold * med))

def calc_depth_balance(bids: np.ndarray, asks: np.ndarray, depth: int = 10) -> float:
    b_notional = (bids[:depth,0] * bids[:depth,1]).sum() if bids.shape[0] else 0.0
    a_notional = (asks[:depth,0] * asks[:depth,1]).sum() if asks.shape[0] else 0.0
    return (b_notional - a_notional) / (b_notional + a_notional + 1e-9)

def extract_orderbook_features(snapshot: Dict) -> Dict[str, float]:
    """
    snapshot: {"timestamp": .., "bids": [[p,q],..], "asks": [[p,q],..]}
    """
    bids = _to_levels(snapshot.get("bids"))
    asks = _to_levels(snapshot.get("asks"))

    feats = {
        "timestamp": pd.to_datetime(snapshot.get("timestamp"), errors="coerce"),
        "spread": calc_spread(bids, asks),
        "mid_price": calc_mid_price(bids, asks),
        "orderbook_imbalance": calc_orderbook_imbalance(bids, asks, depth=5),
        "liquidity_gap": calc_liquidity_gap(bids, asks, depth=5),
        "wall_strength": (
            calc_wall_strength(asks, depth=10) + calc_wall_strength(bids, depth=10)
        ),
        "depth_balance": calc_depth_balance(bids, asks, depth=10),
    }
    # 숫자화/NaN 방지
    for k,v in feats.items():
        if k == "timestamp": continue
        try: feats[k] = float(v) if not pd.isna(v) else 0.0
        except Exception: feats[k] = 0.0
    return feats
