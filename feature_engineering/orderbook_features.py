import numpy as np
import pandas as pd

# ==============================================================
# 📊 호가창 기반 피처 계산 함수들
# ==============================================================

def calc_spread(bids, asks):
    """Best ask - best bid"""
    try:
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return best_ask - best_bid
    except Exception as e:
        print(f"[calc_spread] Warning: {e}")
        return np.nan


def calc_mid_price(bids, asks):
    """(best bid + best ask) / 2"""
    try:
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return (best_bid + best_ask) / 2.0
    except Exception as e:
        print(f"[calc_mid_price] Warning: {e}")
        return np.nan


def calc_orderbook_imbalance(bids, asks, depth: int = 5):
    """
    L2 호가 불균형 = (Σ bid_qty - Σ ask_qty) / (Σ bid_qty + Σ ask_qty)
    """
    try:
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan

        bids = np.array([[float(p), float(q)] for p, q in bids[:depth]], dtype=float)
        asks = np.array([[float(p), float(q)] for p, q in asks[:depth]], dtype=float)

        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
    except Exception as e:
        print(f"[calc_orderbook_imbalance] Warning: {e}")
        return np.nan


def calc_liquidity_gap(bids, asks, depth: int = 10):
    """
    호가 간격의 평균 — 유동성 공백
    큰 값일수록 유동성 공백(=체결 슬리피지 위험) 높음
    """
    try:
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan

        bid_prices = [float(p) for p, _ in bids[:depth]]
        ask_prices = [float(p) for p, _ in asks[:depth]]
        all_prices = sorted(bid_prices + ask_prices)
        diffs = np.diff(all_prices)
        if len(diffs) == 0:
            return 0.0
        return float(np.mean(np.abs(diffs)))
    except Exception as e:
        print(f"[calc_liquidity_gap] Warning: {e}")
        return np.nan


def calc_wall_strength(bids, asks, threshold_ratio: float = 3.0):
    """
    큰 매물벽(호가벽) 감지 비율
    상위 depth 내 최대 잔량 / 평균 잔량
    """
    try:
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan

        bid_qtys = np.array([float(q) for _, q in bids])
        ask_qtys = np.array([float(q) for _, q in asks])
        bid_ratio = np.max(bid_qtys) / (np.mean(bid_qtys) + 1e-6)
        ask_ratio = np.max(ask_qtys) / (np.mean(ask_qtys) + 1e-6)
        return (bid_ratio + ask_ratio) / 2.0
    except Exception as e:
        print(f"[calc_wall_strength] Warning: {e}")
        return np.nan


def calc_depth_balance(bids, asks, depth: int = 10):
    """
    상위 N호가 누적 체결 강도 비율
    """
    try:
        if isinstance(bids, np.ndarray):
            bids = bids.tolist()
        if isinstance(asks, np.ndarray):
            asks = asks.tolist()
        if bids is None or len(bids) == 0 or asks is None or len(asks) == 0:
            return np.nan

        bids = np.array([[float(p), float(q)] for p, q in bids[:depth]], dtype=float)
        asks = np.array([[float(p), float(q)] for p, q in asks[:depth]], dtype=float)

        total_bid_val = np.sum(bids[:, 0] * bids[:, 1])
        total_ask_val = np.sum(asks[:, 0] * asks[:, 1])
        denom = total_bid_val + total_ask_val
        if denom == 0:
            return 0.0
        return (total_bid_val - total_ask_val) / denom
    except Exception as e:
        print(f"[calc_depth_balance] Warning: {e}")
        return np.nan


# ==============================================================
# 🧩 통합 피처 추출 함수
# ==============================================================

def extract_orderbook_features(snapshot: dict) -> dict:
    """
    단일 시점 orderbook snapshot → 주요 피처 추출
    snapshot = {
        "timestamp": ...,
        "bids": [[price, qty], ...],
        "asks": [[price, qty], ...]
    }
    """
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    ts = snapshot.get("timestamp", None)

    feats = {
        "timestamp": pd.to_datetime(ts),
        "spread": calc_spread(bids, asks),
        "mid_price": calc_mid_price(bids, asks),
        "orderbook_imbalance": calc_orderbook_imbalance(bids, asks),
        "liquidity_gap": calc_liquidity_gap(bids, asks),
        "wall_strength": calc_wall_strength(bids, asks),
        "depth_balance": calc_depth_balance(bids, asks),
    }

    # NaN 안정화
    feats = {
        k: (v if k == "timestamp" else (0.0 if pd.isna(v) else float(v)))
        for k, v in feats.items()
    }

    return feats


# ==============================================================
# 🔬 테스트
# ==============================================================

if __name__ == "__main__":
    from datetime import datetime

    ob = {
        "timestamp": datetime.utcnow(),
        "bids": [[65000.0, 1.2], [64999.5, 0.8], [64999.0, 0.6], [64998.5, 0.4]],
        "asks": [[65000.5, 1.5], [65001.0, 1.1], [65001.5, 0.9], [65002.0, 0.7]],
    }

    feats = extract_orderbook_features(ob)
    print("✅ Orderbook feature sample:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
