import pandas as pd

from feature_engineering.trade_features import extract_trade_features  # 배치용(DataFrame 반환)
from feature_engineering.orderbook_features import extract_orderbook_features
from feature_engineering.technical_indicators import compute_technical_indicators

class BatchFeaturePipeline:
    """
    배치/백테스트용 피처 파이프라인
      - base: OHLCV(1s)
      - trades: extract_trade_features(trades_df, windows=("0.5s","1s","5s")) 한 번에
      - depth : extract_orderbook_features(depth_df)  ← 전체 DataFrame 단위 호출
      - tech  : compute_technical_indicators(base)
      - external: (있으면) 1초 그리드 맞춰 join
    """
    def __init__(self, timeframes=("0.5s","1s","5s")):
        self.timeframes = tuple(timeframes)

    def build_features(self, ohlcv_df, trades_df, depth_df, external_df=None):
        # --- 1) base (OHLCV 1s)
        base = ohlcv_df.copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")\
                                .dt.tz_convert("UTC").dt.tz_localize(None)
        base = base.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # --- 2) trade features (한 번에 계산해서 1s 그리드에 맞춰진 DF 받음)
        try:
            tr = extract_trade_features(trades_df, windows=self.timeframes)
        except Exception as e:
            print(f"[trade features] error: {e}")
            tr = pd.DataFrame(columns=["timestamp"])

        # --- 3) orderbook features (전체 DF 단위)  ← row별 dict로 돌리던 기존 방식 제거
        try:
            ob = extract_orderbook_features(depth_df)
        except Exception as e:
            print(f"[orderbook features] error: {e}")
            ob = pd.DataFrame(columns=["timestamp"])

        # --- 4) technical indicators (base 위)
        try:
            ti = compute_technical_indicators(base)
        except Exception as e:
            print(f"[technicals] error: {e}")
            ti = pd.DataFrame(columns=["timestamp"])

        # --- 5) external (이미 1s 그리드로 만들어져 있음)
        ex = external_df if (external_df is not None and not external_df.empty) else pd.DataFrame(columns=["timestamp"])

        # --- 6) 순차 병합 (1초 타임라인 기준)
        merged = base.merge(tr, on="timestamp", how="left")
        merged = merged.merge(ob, on="timestamp", how="left")
        merged = merged.merge(ti, on="timestamp", how="left")
        if not ex.empty:
            merged = merged.merge(ex, on="timestamp", how="left")

        # --- 7) 최소한의 결측 보정
        merged = merged.sort_values("timestamp").ffill().bfill()

        # 숫자 컬럼만 남기려면 아래 주석 해제 (옵션)
        # for c in merged.columns:
        #     if c != "timestamp":
        #         merged[c] = pd.to_numeric(merged[c], errors="coerce")
        # merged = merged.fillna(0.0)

        return merged
