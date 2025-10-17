import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ==============================================================
# 🔧 핵심 Feature 계산 함수들
# ==============================================================

def calc_spread(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    스프레드 계산 (호가창 최상단 ask - bid)
    """
    if not bids or not asks:
        return np.nan
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    return spread


def calc_mid_price(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    미드프라이스 = (best bid + best ask) / 2
    """
    if not bids or not asks:
        return np.nan
    return (float(bids[0][0]) + float(asks[0][0])) / 2


def calc_orderbook_imbalance(bids: List[List[float]], asks: List[List[float]], depth: int = 10) -> float:
    """
    호가 불균형: (Σbid_qty - Σask_qty) / (Σbid_qty + Σask_qty)
    """
    bids = np.array(bids[:depth], dtype=float)
    asks = np.array(asks[:depth], dtype=float)

    bid_vol = np.sum(bids[:, 1])
    ask_vol = np.sum(asks[:, 1])
    if bid_vol + ask_vol == 0:
        return 0.0
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    return imbalance


def calc_order_wall_ratio(bids: List[List[float]], asks: List[List[float]], threshold_ratio: float = 0.2) -> float:
    """
    호가벽 비율 계산 (상위 몇 개 가격대에 전체 물량의 몇 %가 몰려 있는가)
    예: 상위 2호가에 20% 이상 몰려 있으면 order wall 존재
    """
    bids = np.array(bids, dtype=float)
    asks = np.array(asks, dtype=float)
    total_bid_vol = np.sum(bids[:, 1])
    total_ask_vol = np.sum(asks[:, 1])
    if total_bid_vol == 0 or total_ask_vol == 0:
        return 0.0

    top2_bid_ratio = np.sum(bids[:2, 1]) / total_bid_vol
    top2_ask_ratio = np.sum(asks[:2, 1]) / total_ask_vol
    wall_ratio = (top2_bid_ratio + top2_ask_ratio) / 2
    return wall_ratio


def calc_liquidity_void(bids: List[List[float]], asks: List[List[float]], depth: int = 10) -> float:
    """
    유동성 공백(Liquidity void): 
    호가창 간격의 평균 크기 / 스프레드 대비 비율
    """
    bids = np.array(bids[:depth], dtype=float)
    asks = np.array(asks[:depth], dtype=float)

    if len(bids) < 2 or len(asks) < 2:
        return np.nan

    bid_gaps = np.diff(bids[:, 0])
    ask_gaps = np.diff(asks[:, 0])
    avg_gap = (np.mean(np.abs(bid_gaps)) + np.mean(np.abs(ask_gaps))) / 2
    spread = calc_spread(bids.tolist(), asks.tolist())

    if spread == 0:
        return 0.0

    return avg_gap / spread


# ==============================================================
# 🧠 메인 피처 계산 함수
# ==============================================================

def extract_orderbook_features(depth_snapshot: Dict) -> Dict:
    """
    단일 depth 스냅샷(JSON) → 피처 딕셔너리 변환

    | Feature               | 설명                            | 의미                |
    | --------------------- | -----------------------------   | -----------------  |
    | `spread`              | 최상단 매수·매도 간 가격차         | 슬리피지 위험 지표    |
    | `mid_price`           | (bid+ask)/2                    | 기준 가격            |
    | `orderbook_imbalance` | (Σbid - Σask) / (Σbid + Σask)  | 매수·매도 세력 비율   |
    | `order_wall_ratio`    | 상위 2호가 집중도                | 대기 유동성 집중 여부  |
    | `liquidity_void`      | 평균 호가 간격 / 스프레드         | 유동성 공백 정도      |

    """
    bids = depth_snapshot.get("bids", [])
    asks = depth_snapshot.get("asks", [])
    
    features = {
        "spread": calc_spread(bids, asks),
        "mid_price": calc_mid_price(bids, asks),
        "orderbook_imbalance": calc_orderbook_imbalance(bids, asks),
        "order_wall_ratio": calc_order_wall_ratio(bids, asks),
        "liquidity_void": calc_liquidity_void(bids, asks),
    }
    return features


# ==============================================================
# 🔬 테스트용 메인
# ==============================================================

if __name__ == "__main__":
    # 예시 입력 (depth.json 한 건)
    sample_depth = {
        "bids": [[54000.0, 2.1], [53999.5, 1.8], [53999.0, 1.2]],
        "asks": [[54000.5, 2.4], [54001.0, 3.0], [54001.5, 1.7]]
    }

    feats = extract_orderbook_features(sample_depth)
    print("🧩 Orderbook Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")
