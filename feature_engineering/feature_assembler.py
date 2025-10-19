import pandas as pd
import numpy as np
from typing import Dict, Optional

# 변경: 스냅샷 전용 함수로 교체
from feature_engineering.orderbook_features import extract_orderbook_features_snapshot
from feature_engineering.trade_features import extract_trade_features_snapshot

# ==============================================================
# ⚙️ Feature Assembler
# ==============================================================

class FeatureAssembler:
    """
    호가창 + 체결 데이터 기반 통합 Feature Set 생성기

    | 기능                 | 설명                                       |
    | ------------------ | ---------------------------------------- |
    | `assemble()`       | 단일 시점의 orderbook + trades → feature dict |
    | `assemble_batch()` | 여러 시점 데이터를 DataFrame으로 결합                |
    | `normalize` 옵션     | 실시간 학습용 피처 정규화                           |
    | `dropna` 옵션        | NaN 자동 처리 (모델 안정성 강화)                    |
    """

    def __init__(self, normalize: bool = True, dropna: bool = True):
        self.normalize = normalize
        self.dropna = dropna
        self.feature_minmax = {}

    # ----------------------------------------------------------
    def assemble(
        self,
        orderbook_snapshot: Dict,
        trades_df: pd.DataFrame,
        prev_avg_vol: float = 0.0,
    ) -> Dict:
        """
        단일 시점의 orderbook + trade 데이터를 통합 피처셋으로 변환
        """
        ob_feats = extract_orderbook_features_snapshot(orderbook_snapshot)
        tr_feats = extract_trade_features_snapshot(trades_df, prev_avg_vol)

        combined = {**ob_feats, **tr_feats}
        # timestamp 보정
        ts = orderbook_snapshot.get("timestamp")
        combined["timestamp"] = pd.to_datetime(ts, errors="coerce")

        # 정규화
        if self.normalize:
            combined = self._normalize_dict(combined)

        # NaN 제거
        if self.dropna:
            combined = {k: (0.0 if (isinstance(v, float) and (pd.isna(v) or np.isinf(v))) else v)
                        for k, v in combined.items()}

        return combined

    # ----------------------------------------------------------
    def _normalize_dict(self, feats: Dict) -> Dict:
        """
        간단한 Min-Max normalization (초기 구간에서 자동 업데이트)
        """
        normalized = {}
        for k, v in feats.items():
            if isinstance(v, (int, float)) and not pd.isna(v) and not np.isinf(v):
                if k not in self.feature_minmax:
                    self.feature_minmax[k] = {"min": float(v), "max": float(v)}
                else:
                    self.feature_minmax[k]["min"] = min(self.feature_minmax[k]["min"], float(v))
                    self.feature_minmax[k]["max"] = max(self.feature_minmax[k]["max"], float(v))

                vmin, vmax = self.feature_minmax[k]["min"], self.feature_minmax[k]["max"]
                if vmax != vmin:
                    normalized[k] = (float(v) - vmin) / (vmax - vmin)
                else:
                    normalized[k] = 0.5
            else:
                normalized[k] = v
        return normalized

    # ----------------------------------------------------------
    def assemble_batch(
        self,
        orderbook_snapshots: list,
        trade_windows: list,
        prev_avg_vol: float = 0.0,
    ) -> pd.DataFrame:
        """
        여러 시점의 데이터를 병합 → DataFrame 형태로 반환
        """
        rows = []
        for ob, tr in zip(orderbook_snapshots, trade_windows):
            feats = self.assemble(ob, tr, prev_avg_vol)
            rows.append(feats)
        df = pd.DataFrame(rows).sort_values("timestamp")
        df.reset_index(drop=True, inplace=True)
        return df


# ==============================================================
# 🔬 테스트용 메인
# ==============================================================

if __name__ == "__main__":
    from datetime import datetime

    # 샘플 orderbook 데이터
    sample_orderbook = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "bids": [[54000.0, 2.1], [53999.5, 1.8], [53999.0, 1.2]],
        "asks": [[54000.5, 2.4], [54001.0, 3.0], [54001.5, 1.7]],
    }

    # 샘플 trade 데이터
    trade_data = {
        "timestamp": pd.to_datetime([
            "2025-10-17 09:00:00.100",
            "2025-10-17 09:00:00.300",
            "2025-10-17 09:00:00.700",
            "2025-10-17 09:00:00.800"
        ]),
        "price": [54000.0, 54000.2, 54000.5, 54000.4],
        "qty": [0.3, 0.5, 0.7, 0.4],
        "side": ["buy", "buy", "sell", "sell"]
    }
    df_trades = pd.DataFrame(trade_data)

    assembler = FeatureAssembler(normalize=True)
    combined = assembler.assemble(sample_orderbook, df_trades)
    print("🧩 Combined Feature Set (Normalized):")
    for k, v in combined.items():
        print(f"  {k}: {v}")
