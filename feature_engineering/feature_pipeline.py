from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import timedelta

# 우리 모듈들
from feature_engineering.technical_indicators import extract_technical_indicators
from feature_engineering.orderbook_features import extract_orderbook_features
from feature_engineering.trade_features import extract_trade_features
from feature_engineering.feature_assembler import FeatureAssembler

# ==============================================================
# 기본 설정
# ==============================================================

# 멀티 타임프레임 지원 (원하는 것만 켜도 됨)
DEFAULT_TFS = ["0.5s", "1s", "5s", "1min", "3min", "5min"]

# trades 집계 시 사용할 기본 윈도우(초)
TRADE_WINDOW_SECONDS = {
    "0.5s": 0.5,
    "1s": 1,
    "5s": 5,
    "1min": 60,
    "3min": 180,
    "5min": 300,
}

# depth 스냅샷과 trade/ohlcv를 맞출 때 허용 오차
ALIGN_TOLERANCE = pd.Timedelta(milliseconds=300)


# ==============================================================
# 유틸
# ==============================================================

def _ensure_ts(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df.sort_values(col)


def _resample_ohlcv(df_ohlcv: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    초단위 tf('0.5s','1s','5s') 또는 분 단위 tf('1min','3min','5min')로 리샘플링.
    df_ohlcv 컬럼: timestamp, open, high, low, close, volume
    """
    df = _ensure_ts(df_ohlcv)
    rule = tf.replace("min", "T")  # pandas 규칙(분) 표기: '1T','3T','5T'
    o = (
        df.set_index("timestamp")
          .resample(rule)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna()
          .reset_index()
    )
    return o


def _group_trades_by_window(df_trades: pd.DataFrame, window_sec: float) -> List[pd.DataFrame]:
    """
    trades를 고정 길이 창(window_sec)으로 연속 분할하여 리스트 반환.
    실시간/스트리밍에서는 윈도우 슬라이딩 로직으로 교체 가능.
    """
    df = _ensure_ts(df_trades)
    if df.empty:
        return []

    start = df["timestamp"].min()
    end = df["timestamp"].max()
    windows = []
    cur_start = start

    delta = pd.Timedelta(seconds=window_sec)

    while cur_start <= end:
        cur_end = cur_start + delta
        seg = df[(df["timestamp"] >= cur_start) & (df["timestamp"] < cur_end)]
        if len(seg) > 0:
            windows.append(seg)
        cur_start = cur_end
    return windows


def _nearest_join(left: pd.DataFrame, right: pd.DataFrame, on: str = "timestamp", tolerance=ALIGN_TOLERANCE, direction="nearest"):
    """
    pandas.merge_asof 래퍼: timestamp 기준 근접 매칭
    """
    left = _ensure_ts(left, on)
    right = _ensure_ts(right, on)
    return pd.merge_asof(
        left.sort_values(on),
        right.sort_values(on),
        on=on,
        direction=direction,
        tolerance=tolerance
    )


# ==============================================================
# 배치 파이프라인
# ==============================================================

class BatchFeaturePipeline:
    """
    배치(백테스트/학습 데이터 생성)용 통합 피처 파이프라인.
    입력:
      - ohlcv_df: 캔들(최소 1s 권장)
      - trades_df: aggTrades (timestamp, price, qty, side)
      - depth_df: depth 스냅샷 (timestamp, bids, asks)
    출력:
      - 멀티 타임프레임 결합 피처 DataFrame
    """
    def __init__(self, timeframes: List[str] = None, normalize: bool = True, dropna: bool = True):
        self.timeframes = timeframes or DEFAULT_TFS
        self.assembler = FeatureAssembler(normalize=normalize, dropna=dropna)

    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        depth_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # 안전정렬
        ohlcv_df = _ensure_ts(ohlcv_df)
        trades_df = _ensure_ts(trades_df)
        depth_df = _ensure_ts(depth_df)

        # --- 1) 기술적 지표 (기준 TF: 1s 또는 5s 이상 권장)
        tech_df = extract_technical_indicators(ohlcv_df)  # timestamp, 지표컬럼들

        # --- 2) TF별 trades 집계 → trade_features
        tf_trade_feats = []
        for tf in self.timeframes:
            window_sec = TRADE_WINDOW_SECONDS["1s"]  # 기본 1초 집계 후 리샘플 추천
            if tf in TRADE_WINDOW_SECONDS:
                window_sec = TRADE_WINDOW_SECONDS[tf]

            trade_windows = _group_trades_by_window(trades_df, window_sec)
            rows = []
            prev_avg_vol = 0.0

            # rolling 평균 갱신용
            vol_hist = []

            for tw in trade_windows:
                feats = extract_trade_features(tw, prev_avg_vol=prev_avg_vol)
                # 윈도우의 중앙 timestamp 또는 마지막 timestamp 사용
                ts = tw["timestamp"].max()
                feats["timestamp"] = ts
                feats["tf"] = tf
                rows.append(feats)

                # 평균 거래량 업데이트
                vol_hist.append(tw["qty"].sum())
                if len(vol_hist) > 30:
                    vol_hist.pop(0)
                prev_avg_vol = np.mean(vol_hist) if vol_hist else 0.0

            df_tf = pd.DataFrame(rows)
            if not df_tf.empty:
                df_tf.rename(columns={c: f"{c}@{tf}" for c in df_tf.columns if c not in ["timestamp", "tf"]}, inplace=True)
                tf_trade_feats.append(df_tf[["timestamp"] + [c for c in df_tf.columns if c not in ["tf"]]])

        trade_feat_df = None
        if tf_trade_feats:
            # timestamp 기준 outer merge
            trade_feat_df = tf_trade_feats[0]
            for add in tf_trade_feats[1:]:
                trade_feat_df = pd.merge(trade_feat_df, add, on="timestamp", how="outer")
            trade_feat_df = trade_feat_df.sort_values("timestamp")

        # --- 3) TF별 depth → orderbook_features
        ob_rows = []
        for _, row in depth_df.iterrows():
            ob = {"timestamp": row["timestamp"], "bids": row["bids"], "asks": row["asks"]}
            feat = extract_orderbook_features(ob)
            feat["timestamp"] = row["timestamp"]
            ob_rows.append(feat)
        ob_df = pd.DataFrame(ob_rows).sort_values("timestamp") if ob_rows else pd.DataFrame(columns=["timestamp"])

        # --- 4) 근접 조인으로 tech + trades + orderbook 결합
        # 기준은 기술지표(=캔들) timestamp
        feat_df = tech_df[["timestamp"]].copy()

        if trade_feat_df is not None and not trade_feat_df.empty:
            feat_df = _nearest_join(feat_df, trade_feat_df, "timestamp", tolerance=ALIGN_TOLERANCE)

        if ob_df is not None and not ob_df.empty:
            feat_df = _nearest_join(feat_df, ob_df, "timestamp", tolerance=ALIGN_TOLERANCE)

        # tech 컬럼도 붙이기
        feat_df = pd.merge_asof(
            feat_df.sort_values("timestamp"),
            tech_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=ALIGN_TOLERANCE
        )

        # 정리
        feat_df = feat_df.sort_values("timestamp").dropna().reset_index(drop=True)
        return feat_df


# ==============================================================
# 실시간 파이프라인
# ==============================================================

class RealtimeFeaturePipeline:
    """
    실시간(스트리밍)용 피처 파이프라인.
    - 최근 N초 OHLCV 슬라이스 + 현재 depth 스냅샷 + 최근 trades 윈도우
    - FeatureAssembler를 이용해 한 시점 feature dict 생성
    """
    def __init__(self, normalize: bool = True, dropna: bool = True):
        self.assembler = FeatureAssembler(normalize=normalize, dropna=dropna)

    def transform(
        self,
        latest_orderbook_snapshot: Dict,     # {"timestamp":..., "bids":[[p,q],...], "asks":[[p,q],...]}
        recent_trades_window: pd.DataFrame,  # 최근 1s/0.5s 등 윈도우
        recent_ohlcv_slice: pd.DataFrame     # 최근 수 초/분의 OHLCV (지표계산용)
    ) -> Dict:
        # 1) technical indicators (최근 슬라이스에서 최신 행만 사용)
        tech = extract_technical_indicators(recent_ohlcv_slice)
        tech_last = tech.iloc[-1:].copy() if not tech.empty else pd.DataFrame()

        # 2) orderbook + trade 를 assembler로 통합
        combined = self.assembler.assemble(
            orderbook_snapshot=latest_orderbook_snapshot,
            trades_df=recent_trades_window,
            prev_avg_vol=recent_trades_window["qty"].rolling(30).sum().mean() if not recent_trades_window.empty else 0.0
        )

        # 3) tech 컬럼 병합
        if not tech_last.empty:
            for col in tech_last.columns:
                if col == "timestamp":
                    continue
                combined[col] = tech_last.iloc[0][col]

        return combined
