import pandas as pd
import numpy as np
from typing import List, Optional
from feature_engineering.orderbook_features import extract_orderbook_features
from feature_engineering.trade_features import extract_trade_features
from feature_engineering.technical_indicators import compute_technical_indicators

# ==============================================================
# 🧩 Batch Feature Pipeline
# ==============================================================

class BatchFeaturePipeline:
    """
    OHLCV + Trades + Depth 데이터를 받아 통합 피처 세트를 생성하는 모듈.
    """

    def __init__(self, timeframes: Optional[List[str]] = None):
        self.timeframes = timeframes or ["0.5s", "1s", "5s"]

    # ----------------------------------------------------------
    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        depth_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        3개 데이터셋을 받아 feature DataFrame 생성.
        필수 컬럼:
            ohlcv_df: [timestamp, open, high, low, close, volume]
            trades_df: [timestamp, price, qty, side]
            depth_df: [timestamp, bids, asks]
        """

        # ---------- 0) 입력 유효성 검증 ----------
        if ohlcv_df is None or trades_df is None or depth_df is None:
            print("❌ [build_features] 입력 중 None 존재")
            return pd.DataFrame(columns=["timestamp"])

        if len(ohlcv_df) == 0:
            print("⚠️ [build_features] ohlcv_df 비어있음")
            return pd.DataFrame(columns=["timestamp"])

        # ---------- 1) timestamp 표준화 ----------
        for df in [ohlcv_df, trades_df, depth_df]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True)
            df.dropna(subset=["timestamp"], inplace=True)
        ohlcv_df.drop_duplicates(subset=["timestamp"], inplace=True)

        # ---------- 2) Trade Feature ----------
        print("🟩 [build_features] 단계 2: Trade Feature 시작")
        tf_dfs = []
        for tf in self.timeframes:
            try:
                window = pd.to_timedelta(tf)
                grouped = trades_df.groupby(pd.Grouper(key="timestamp", freq=window))
                rows = []
                for ts, g in grouped:
                    if len(g) == 0:
                        continue
                    feats = extract_trade_features(g)
                    feats["timestamp"] = ts
                    feats["tf"] = tf
                    rows.append(feats)
                tf_df = pd.DataFrame(rows)
                print(f"  └─ {tf} 구간 결과:", tf_df.shape)
                if len(tf_df) > 0:
                    tf_df.rename(columns={c: f"{c}@{tf}" for c in tf_df.columns if c not in ["timestamp", "tf"]}, inplace=True)
                    tf_dfs.append(tf_df)
            except Exception as e:
                print(f"[build_features] Trade TF={tf} 실패:", e)

        trade_feat_df = None
        if tf_dfs:
            trade_feat_df = tf_dfs[0]
            for add in tf_dfs[1:]:
                trade_feat_df = pd.merge(trade_feat_df, add, on="timestamp", how="outer")
            print("✅ Trade Feature 병합 완료:", trade_feat_df.shape)
        else:
            print("⚠️ Trade Feature 비어 있음")

        # ---------- 3) Orderbook Feature ----------
        print("🟩 [build_features] 단계 3: Orderbook Feature 시작")
        orderbook_rows = []
        for _, ob in depth_df.iterrows():
            feats = extract_orderbook_features(ob.to_dict())
            orderbook_rows.append(feats)
        ob_df = pd.DataFrame(orderbook_rows)
        print("✅ Orderbook Feature 완료:", ob_df.shape)

        # ---------- 4) Technical Indicator ----------
        print("🟩 [build_features] 단계 4: Technical Indicator 시작")
        tech_df = compute_technical_indicators(ohlcv_df.copy())
        print("✅ Technical Feature 완료:", tech_df.shape)


        # ---------- 5) Merge all ----------
        merged = ohlcv_df.copy()
        for add_df in [trade_feat_df, ob_df, tech_df]:
            if add_df is not None and len(add_df) > 0:
                merged = pd.merge_asof(
                    merged.sort_values("timestamp"),
                    add_df.sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=3)
                )

        # ---------- 6) 중복 / NaN 처리 ----------
        merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
        merged.fillna(method="ffill", inplace=True)
        merged.fillna(method="bfill", inplace=True)

        # ---------- 7) 최소 보장 ----------
        if len(merged) == 0:
            print("⚠️ [build_features] 최종 병합 결과가 비어 있음 — 최소 1행 생성")
            merged = pd.DataFrame({
                "timestamp": [pd.Timestamp.utcnow()],
                "spread": [0.0],
                "mid_price": [0.0],
                "orderbook_imbalance": [0.0]
            })

        print(f"✅ [build_features] 결과 shape = {merged.shape}")
        return merged


# ==============================================================
# 🔬 테스트
# ==============================================================

if __name__ == "__main__":
    # 간단한 더미 데이터로 테스트
    import datetime
    now = pd.Timestamp.utcnow().floor("s")

    ohlcv = pd.DataFrame({
        "timestamp": pd.date_range(now, periods=5, freq="S"),
        "open": np.random.rand(5),
        "high": np.random.rand(5),
        "low": np.random.rand(5),
        "close": np.random.rand(5),
        "volume": np.random.rand(5)
    })

    trades = pd.DataFrame({
        "timestamp": pd.date_range(now, periods=30, freq="200ms"),
        "price": np.random.rand(30),
        "qty": np.random.rand(30),
        "side": np.random.choice(["buy", "sell"], size=30)
    })

    depth = pd.DataFrame({
        "timestamp": pd.date_range(now, periods=5, freq="S"),
        "bids": [[[100.0, 1.0], [99.5, 1.0]]] * 5,
        "asks": [[[100.5, 1.0], [101.0, 1.0]]] * 5
    })

    pipe = BatchFeaturePipeline(timeframes=["0.5s", "1s"])
    df = pipe.build_features(ohlcv_df=ohlcv, trades_df=trades, depth_df=depth)
    print(df.head())
