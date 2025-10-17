import pandas as pd
import numpy as np
from typing import List, Optional
from feature_engineering.orderbook_features import extract_orderbook_features
from feature_engineering.trade_features import extract_trade_features
from feature_engineering.technical_indicators import compute_technical_indicators
from feature_engineering.external_features import build_external_features

class BatchFeaturePipeline:
    def __init__(self, timeframes: Optional[List[str]] = None):
        self.timeframes = timeframes or ["0.5s","1s","5s"]

    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        depth_df: pd.DataFrame,
        external_df: Optional[pd.DataFrame] = None,  
    ) -> pd.DataFrame:
        if ohlcv_df is None or len(ohlcv_df)==0:
            print("⚠️ [build_features] ohlcv empty"); return pd.DataFrame(columns=["timestamp"])

        # ts 표준화
        for df in (ohlcv_df, trades_df, depth_df):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True); df.dropna(subset=["timestamp"], inplace=True)

        # 2) Trades (각 tf 그룹 → extract_trade_features)
        tf_dfs = []
        for tf in self.timeframes:
            try:
                grouped = trades_df.groupby(pd.Grouper(key="timestamp", freq=pd.to_timedelta(tf)))
                rows = []
                for ts,g in grouped:
                    if len(g)==0: continue
                    feats = extract_trade_features(g)
                    feats["timestamp"] = ts
                    rows.append(feats)
                tf_df = pd.DataFrame(rows)
                if not tf_df.empty:
                    tf_df.rename(columns={c:f"{c}@{tf}" for c in tf_df.columns if c!="timestamp"}, inplace=True)
                    tf_dfs.append(tf_df)
            except Exception as e:
                print(f"[trade tf {tf}] error:", e)
        trade_feat_df = None
        if tf_dfs:
            trade_feat_df = tf_dfs[0]
            for add in tf_dfs[1:]:
                trade_feat_df = pd.merge(trade_feat_df, add, on="timestamp", how="outer")
            trade_feat_df.sort_values("timestamp", inplace=True)

        # 3) Orderbook
        ob_rows = [ extract_orderbook_features(row._asdict() if hasattr(row,'_asdict') else row.to_dict())
                    for _,row in depth_df.iterrows() ]
        ob_df = pd.DataFrame(ob_rows) if ob_rows else pd.DataFrame(columns=["timestamp"])

        # 4) Technical
        tech_df = compute_technical_indicators(ohlcv_df.copy())

        # 5) Merge all (asof)
        merged = ohlcv_df.copy()
        for add in (trade_feat_df, ob_df, tech_df):
            if add is not None and len(add)>0:
                merged = pd.merge_asof(
                    merged.sort_values("timestamp"),
                    add.sort_values("timestamp"),
                    on="timestamp", direction="nearest",
                    tolerance=pd.Timedelta(seconds=3)
                )

        # ✅ 외부 피처 병합
        if external_df is not None and not external_df.empty:
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                external_df.sort_values("timestamp"),
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta(seconds=3)
            )

        merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
        merged.fillna(method="ffill", inplace=True); merged.fillna(method="bfill", inplace=True)
        if len(merged)==0:
            merged = pd.DataFrame({"timestamp":[pd.Timestamp.utcnow().floor("s")]})
        print(f"✅ [build_features] shape={merged.shape}")
        return merged
