# feature_engineering/external_features.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

def _tz_norm(df: Optional[pd.DataFrame], tscol: str = "timestamp") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[tscol])
    df = df.copy()
    df[tscol] = pd.to_datetime(df[tscol], utc=True, errors="coerce")
    df[tscol] = df[tscol].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=[tscol]).drop_duplicates(subset=[tscol]).sort_values(tscol)
    return df

def _resample_to_seconds(df: pd.DataFrame, on: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[on])
    d = df.set_index(on).resample("1S").last().ffill().reset_index()
    return d

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(3, win//3)).mean()
    v = s.rolling(win, min_periods=max(3, win//3)).std()
    return (s - m) / (v.replace(0, np.nan))

def build_external_features(ext: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    """
    ext keys (있으면 사용, 없으면 건너뜀):
      - funding: ['timestamp','symbol','funding_rate']
      - open_interest: ['timestamp','symbol','open_interest']
      - long_short_ratio: ['timestamp','symbol','long_short_ratio','long_account','short_account']
      - index_mark: ['timestamp','symbol','mark_price','index_price','last_funding_rate']
      - liquidations: ['timestamp','symbol','side','price','qty']
      - arbitrage: ['timestamp','symbol','binance_mark','okx_mark','arb_spread']
    반환: 초 단위 표준화된 외부 피처 DF (timestamp 포함)
    """
    # ---------- 0) 개별 소스 정규화 ----------
    fund = _tz_norm(ext.get("funding"))
    oi   = _tz_norm(ext.get("open_interest"))
    lsr  = _tz_norm(ext.get("long_short_ratio"))
    im   = _tz_norm(ext.get("index_mark"))
    liq  = _tz_norm(ext.get("liquidations"))
    arb  = _tz_norm(ext.get("arbitrage"))

    # ---------- 1) 각 소스 → 파생 피처 ----------
    feats = []

    if not fund.empty:
        f = _resample_to_seconds(fund[["timestamp","funding_rate"]])
        f["funding_rate_diff"] = f["funding_rate"].diff()
        f["funding_rate_z"]    = _zscore(f["funding_rate"], 96)  # 8h @ 5m 기준 대략
        feats.append(f)

    if not oi.empty:
        o = _resample_to_seconds(oi[["timestamp","open_interest"]])
        o["oi_change"]     = o["open_interest"].diff()
        o["oi_change_pct"] = o["open_interest"].pct_change()
        o["oi_z"]          = _zscore(o["open_interest"], 600)  # ~10분 @ 1s
        feats.append(o)

    if not lsr.empty:
        l = _resample_to_seconds(lsr[["timestamp","long_short_ratio","long_account","short_account"]])
        l["lsr_z"] = _zscore(l["long_short_ratio"], 600)
        feats.append(l)

    if not im.empty:
        m = _resample_to_seconds(im[["timestamp","mark_price","index_price"]])
        m["basis_abs"]  = m["mark_price"] - m["index_price"]
        m["basis_pct"]  = m["basis_abs"] / m["index_price"].replace(0, np.nan)
        feats.append(m[["timestamp","basis_abs","basis_pct"]])

    if not arb.empty:
        a = _resample_to_seconds(arb[["timestamp","arb_spread"]])
        a["arb_spread_z"] = _zscore(a["arb_spread"], 600)
        feats.append(a)

    if not liq.empty:
        # 초 단위 합계/건수도 가능하지만 보통 청산은 듬성 → 1분 윈도우로 매끈하게
        lq = liq.copy()
        lq["count"] = 1
        lq["notional"] = pd.to_numeric(lq.get("price"), errors="coerce") * pd.to_numeric(lq.get("qty"), errors="coerce")
        lq = lq.set_index("timestamp").resample("1S").agg({
            "qty":"sum", "count":"sum", "notional":"sum"
        }).fillna(0.0).reset_index()
        lq.rename(columns={
            "qty":"liq_qty_1s", "count":"liq_count_1s", "notional":"liq_notional_1s"
        }, inplace=True)
        # 완충: 10초/60초 누적
        lq["liq_qty_10s"]      = lq["liq_qty_1s"].rolling(10, min_periods=1).sum()
        lq["liq_count_10s"]    = lq["liq_count_1s"].rolling(10, min_periods=1).sum()
        lq["liq_notional_10s"] = lq["liq_notional_1s"].rolling(10, min_periods=1).sum()
        lq["liq_qty_60s"]      = lq["liq_qty_1s"].rolling(60, min_periods=1).sum()
        lq["liq_count_60s"]    = lq["liq_count_1s"].rolling(60, min_periods=1).sum()
        lq["liq_notional_60s"] = lq["liq_notional_1s"].rolling(60, min_periods=1).sum()
        feats.append(lq)

    # ---------- 2) 병합 ----------
    if not feats:
        return pd.DataFrame(columns=["timestamp"])
    out = feats[0]
    for add in feats[1:]:
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            add.sort_values("timestamp"),
            on="timestamp", direction="nearest",
            tolerance=pd.Timedelta(seconds=3)
        )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    # 결측 보정
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out
