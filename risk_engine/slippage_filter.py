# risk_engine/slippage_filter.py
# Slippage & Liquidity (S&L) v1 — robust, no-hard-crash design
# - 입력: features_df (1s 그리드 권장). 필수 최소: ['timestamp','close'].
# - 선택 입력: ['spread','mid_price','orderbook_imbalance','depth_balance',
#              'wall_strength','liquidity_gap','trade_count@1s','trade_imbalance@1s',
#              'buy_notional_ps@1s','sell_notional_ps@1s','feed_lag_ms', ...]
# - 출력: 입력 df에 다음 열을 추가하여 반환
#   ['relative_spread_bp','expected_slippage_bp','rv_3s_bp','risk_score',
#    'decision','size_scale','order_type_hint']

from __future__ import annotations
import os
import math
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

EPS = 1e-12

DEFAULT_CFG: Dict[str, Any] = {
    "risk_v": "1.0.0",
    "window_ms": 1000,
    "q_ref_usdt": 10000.0,
    "alpha": 25.0,
    "beta": 7.0,
    "thr": {
        "spread_bp": {"good": 0.20, "warn": 0.80, "bad": 1.50},
        "slip_bp":   {"good": 0.40, "warn": 0.80, "bad": 1.20},
        "vol_3s_bp_bad": 6.0
    },
    "feed_lag_hard_ms": 500,
    "policy": {
        "allow_max_risk": 30.0,
        "caution_max_risk": 60.0,
        "size_scale": {"allow": 1.0, "caution": 0.5, "block": 0.0},
        "order_type_hint": {
            "allow":  ["post-only", "marketable-limit"],
            "caution":["post-only", "tight-limit"]
        }
    },
    # spread/mid_price 가 비어 있을 때 사용할 보수적 대체값(bp)
    "fallback_spread_bp": 0.7
}

def _load_yaml_or_default(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if not path:
        return cfg
    if not os.path.exists(path):
        print(f"⚠️ config not found: {path} → using defaults")
        return cfg
    try:
        import yaml  # PyYAML
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # shallow merge (dict of dicts)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except Exception as e:
        print(f"⚠️ failed to load yaml: {e} → using defaults")
    return cfg

def _bp_from_spread_mid(spread: float, mid: float, fallback_bp: float) -> float:
    if spread is None or mid is None:
        return fallback_bp
    if not np.isfinite(spread) or not np.isfinite(mid) or mid <= 0:
        return fallback_bp
    return float(1e4 * spread / (mid + EPS))

def _zscore_s(s: pd.Series, win: int = 60) -> pd.Series:
    m = s.rolling(win, min_periods=max(5, win//5)).mean()
    v = s.rolling(win, min_periods=max(5, win//5)).std()
    return (s - m) / (v + EPS)

def _first_existing(df: pd.DataFrame, candidates: List[str], default=None):
    for c in candidates:
        if c in df.columns:
            return df[c]
    if default is None:
        # return a column of zeros with same index
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)

class SlippageFilter:
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        # load config (dict overrides file)
        self.cfg = _load_yaml_or_default(config_path)
        if config_dict:
            for k, v in config_dict.items():
                if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                    self.cfg[k].update(v)
                else:
                    self.cfg[k] = v

    # ------------------------------------------------------------------
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: 1초 그리드 권장. 최소 'timestamp','close' 포함.
        반환: 입력 df + S&L 결과 컬럼
        """
        if "timestamp" not in df.columns:
            raise ValueError("features df must include 'timestamp' column")
        if "close" not in df.columns:
            raise ValueError("features df must include 'close' column (for rv_3s)")

        out = df.copy()

        # ===== 1) Basic derived metrics =====
        # relative_spread_bp (fallback if spread/mid_price missing)
        spread = out["spread"] if "spread" in out.columns else pd.Series(np.nan, index=out.index)
        mid    = out["mid_price"] if "mid_price" in out.columns else pd.Series(np.nan, index=out.index)
        rel_spread_bp = 1e4 * spread.astype(float) / (mid.astype(float) + EPS)
        rel_spread_bp = rel_spread_bp.where(np.isfinite(rel_spread_bp) & (mid > 0), self.cfg["fallback_spread_bp"])

        # rv_3s_bp from close
        ret1 = out["close"].astype(float).pct_change().fillna(0.0)
        rv3s = np.sqrt((ret1.pow(2)).rolling(3, min_periods=1).sum()) * 1e4

        # trade_rate: prefer 1s count, else approximate from 0.5s, else zeros
        trade_rate = _first_existing(out, ["trade_count@1s", "trade_count"], default=0.0)
        if "trade_count@0.5s" in out.columns and "trade_count@1s" not in out.columns:
            trade_rate = out["trade_count@0.5s"].rolling(2, min_periods=1).sum()

        # trade_imbalance: prefer direct, else derive from buy/sell notional
        tri = _first_existing(out, ["trade_imbalance@1s", "trade_imbalance"], default=np.nan)
        if tri.isna().all():
            buy_not = _first_existing(out, ["buy_notional_ps@1s", "buy_notional@1s", "buy_notional_ps"], default=0.0)
            sell_not = _first_existing(out, ["sell_notional_ps@1s", "sell_notional@1s", "sell_notional_ps"], default=0.0)
            tri = (buy_not - sell_not) / (buy_not + sell_not + EPS)
        tri = tri.clip(-1, 1).fillna(0.0)

        # ob_imbalance proxy: prefer orderbook_imbalance else depth_balance
        obimb = _first_existing(out, ["orderbook_imbalance", "depth_balance"], default=0.0).clip(-1, 1).fillna(0.0)

        # liquidity proxy (quote notional or size). prefer wall_strength, else |liquidity_gap|
        liq_proxy = _first_existing(out, ["wall_strength"], default=np.nan)
        if liq_proxy.isna().all():
            liq_proxy = _first_existing(out, ["liquidity_gap"], default=0.0).abs()
        # avoid zeros
        liq_proxy = liq_proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0) + EPS

        # ===== 2) Expected slippage estimator =====
        # Mode B (best-of-book only / aggregated): slip ≈ max(spread/2, α * Q_ref / liquidity_proxy) + β * |trade_imb|
        alpha = float(self.cfg["alpha"])
        beta  = float(self.cfg["beta"])
        qref  = float(self.cfg["q_ref_usdt"])

        slip_from_liq = alpha * (qref / liq_proxy.astype(float))
        slip_from_spread = rel_spread_bp / 2.0
        expected_slip_bp = np.maximum(slip_from_spread, slip_from_liq) + beta * tri.abs()

        # clamp to a sane range (avoid explosions when liq very small)
        expected_slip_bp = expected_slip_bp.clip(lower=0.0, upper=20.0)

        # ===== 3) Subscores and risk =====
        thr = self.cfg["thr"]
        S_spread = (100.0 * rel_spread_bp / (thr["spread_bp"]["bad"] + EPS)).clip(0, 100)
        S_slip   = (100.0 * expected_slip_bp / (thr["slip_bp"]["bad"] + EPS)).clip(0, 100)
        S_vol    = (100.0 * rv3s / (thr["vol_3s_bp_bad"] + EPS)).clip(0, 100)
        z_burst  = _zscore_s(trade_rate.astype(float), 60)
        S_burst  = (100.0 * z_burst.clip(lower=0.0) / 5.0).clip(0, 100).fillna(0.0)
        S_imb    = (100.0 * obimb.abs()).clip(0, 100)
        S_spread = S_spread.fillna(0.0)
        S_slip   = S_slip.fillna(0.0)
        S_vol    = S_vol.fillna(0.0)
        S_imb    = S_imb.fillna(0.0)
        risk = (0.35*S_slip + 0.25*S_spread + 0.20*S_vol + 0.10*S_burst + 0.10*S_imb).clip(0, 100)

        # ===== 4) Decision policy =====
        allow_max = float(self.cfg["policy"]["allow_max_risk"])
        caution_max = float(self.cfg["policy"]["caution_max_risk"])
        feed_lag = _first_existing(out, ["feed_lag_ms"], default=0.0).fillna(0.0).astype(float)

        decision = np.where(risk <= allow_max, "ALLOW",
                     np.where(risk <= caution_max, "CAUTION", "BLOCK"))

        # hard guard: feed lag
        decision = np.where(feed_lag > float(self.cfg["feed_lag_hard_ms"]), "BLOCK", decision)

        # sizing & order-type hint
        size_scale = np.where(decision == "ALLOW",
                              float(self.cfg["policy"]["size_scale"]["allow"]),
                              np.where(decision == "CAUTION",
                                       float(self.cfg["policy"]["size_scale"]["caution"]),
                                       float(self.cfg["policy"]["size_scale"]["block"])))
        # choose first hint for simplicity
        hint_allow   = self.cfg["policy"]["order_type_hint"]["allow"][0]
        hint_caution = self.cfg["policy"]["order_type_hint"]["caution"][0]
        order_hint = np.where(decision == "ALLOW", hint_allow,
                      np.where(decision == "CAUTION", hint_caution, "post-only"))

        # ===== 5) Attach outputs =====
        out["relative_spread_bp"]   = rel_spread_bp.astype(float)
        out["expected_slippage_bp"] = expected_slip_bp.astype(float)
        out["rv_3s_bp"]             = rv3s.astype(float)
        out["risk_score"]           = risk.astype(float)
        out["decision"]             = decision.astype(str)
        out["size_scale"]           = size_scale.astype(float)
        out["order_type_hint"]      = order_hint.astype(str)

        return out

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",  required=True, help="features_1s parquet path")
    ap.add_argument("--out", dest="outp", required=False, default="", help="output parquet (optional)")
    ap.add_argument("--cfg", dest="cfg", required=False, default="config/risk_v1.yaml", help="yaml config path")
    args = ap.parse_args()

    feats = pd.read_parquet(args.inp)
    filt = SlippageFilter(config_path=args.cfg)
    res  = filt.apply(feats)

    if args.outp:
        os.makedirs(os.path.dirname(args.outp), exist_ok=True)
        res.to_parquet(args.outp, index=False)
        print(f"✅ saved: {args.outp} (shape={res.shape})")
    else:
        print(res[["timestamp","risk_score","decision","size_scale","order_type_hint"]].head(10))
