import pandas as pd
import numpy as np
from typing import Dict, Optional, List

EPS = 1e-12

def _to_dt_utc_naive(s) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)

def _to_1s_index(df: pd.DataFrame, ts_col="timestamp") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
    d = df.copy()
    d[ts_col] = _to_dt_utc_naive(d[ts_col])
    d = d.dropna(subset=[ts_col]).sort_values(ts_col)
    return d.set_index(ts_col).asfreq("1s")

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(5, win//5)).mean()
    v = s.rolling(win, min_periods=max(5, win//5)).std()
    return (s - m) / (v + EPS)

def build_external_features(inputs: Dict[str, pd.DataFrame],
                            liq_windows_s: List[int] = (30, 60),
                            big_liq_notional: float = 100_000.0) -> pd.DataFrame:
    """
    inputs keys (있으면 사용):
      - funding: [timestamp, symbol, funding_rate]
      - open_interest: [timestamp, symbol, open_interest]
      - long_short_ratio: [timestamp, symbol, long_short_ratio, long_account, short_account]
      - index_mark: [timestamp, symbol, mark_price, index_price, last_funding_rate]
      - arbitrage: [timestamp, symbol, binance_mark, okx_mark, arb_spread]
      - liquidations: [timestamp, side('buy'|'sell'), price, qty, notional]
    출력: 1초 그리드 DataFrame (timestamp index)
    """
    parts = []

    # ---------- Funding ----------
    fr = inputs.get("funding")
    if fr is not None and not fr.empty:
        f = _to_1s_index(fr)
        if "funding_rate" in f.columns:
            f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce").fillna(0.0)
            # 1시간 롤링 Z (1s 그리드에서 3600개)
            f["funding_rate_z"] = _zscore(f["funding_rate"], 3600)
            parts.append(f[["funding_rate", "funding_rate_z"]])

    # ---------- Open Interest ----------
    oi = inputs.get("open_interest")
    if oi is not None and not oi.empty:
        o = _to_1s_index(oi)
        if "open_interest" in o.columns:
            o["open_interest"] = pd.to_numeric(o["open_interest"], errors="coerce").fillna(method="ffill").fillna(0.0)
            o["oi_change"] = o["open_interest"].diff().fillna(0.0)
            o["oi_change_pct"] = (o["open_interest"].pct_change().replace([np.inf, -np.inf], np.nan)).fillna(0.0)
            parts.append(o[["open_interest","oi_change","oi_change_pct"]])

    # ---------- Long/Short Ratio ----------
    lsr = inputs.get("long_short_ratio")
    if lsr is not None and not lsr.empty:
        l = _to_1s_index(lsr)
        for c in ["long_short_ratio","long_account","short_account"]:
            if c in l.columns:
                l[c] = pd.to_numeric(l[c], errors="coerce")
        # 1h 기준 Z
        if "long_short_ratio" in l.columns:
            l["lsr_z"] = _zscore(l["long_short_ratio"].fillna(method="ffill").fillna(0.0), 3600)
        parts.append(l[[c for c in ["long_short_ratio","long_account","short_account","lsr_z"] if c in l.columns]])

    # ---------- Index/Mark ----------
    im = inputs.get("index_mark")
    if im is not None and not im.empty:
        m = _to_1s_index(im)
        for c in ["mark_price","index_price","last_funding_rate"]:
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce")
        parts.append(m[[c for c in ["mark_price","index_price","last_funding_rate"] if c in m.columns]])

    # ---------- Arbitrage Spread ----------
    arb = inputs.get("arbitrage")
    if arb is not None and not arb.empty:
        a = _to_1s_index(arb)
        if "arb_spread" in a.columns:
            a["arb_spread"] = pd.to_numeric(a["arb_spread"], errors="coerce")
        parts.append(a[[c for c in ["arb_spread"] if c in a.columns]])

    # ---------- Liquidations (실시간/로컬 집계) ----------
    liq = inputs.get("liquidations")
    if liq is not None and not liq.empty:
        q = liq.copy()
        # notional 없으면 price*qty로 생성
        if "notional" not in q.columns:
            q["notional"] = pd.to_numeric(q.get("price", 0), errors="coerce") * pd.to_numeric(q.get("qty", 0), errors="coerce")
        q = q[["timestamp","side","notional"]].copy()
        q["timestamp"] = _to_dt_utc_naive(q["timestamp"])
        q = q.dropna(subset=["timestamp"]).sort_values("timestamp")
        q["side"] = q["side"].astype(str).str.lower().where(lambda s: s.isin(["buy","sell"]), "buy")
        q["notional"] = pd.to_numeric(q["notional"], errors="coerce").fillna(0.0)

        # 1s 그리드
        qi = q.set_index("timestamp").assign(
            liq_count=1.0,
            liq_notional=lambda d: d["notional"],
            liq_buy=lambda d: np.where(d["side"]=="buy", d["notional"], 0.0),
            liq_sell=lambda d: np.where(d["side"]=="sell", d["notional"], 0.0),
            liq_big=lambda d: (d["notional"] >= float(big_liq_notional)).astype(float)
        )

        # 동일 1s 내 다건 합산
        qi = qi.resample("1s").sum().fillna(0.0)

        out = pd.DataFrame(index=qi.index)
        for W in liq_windows_s:
            win = f"{int(W)}s"
            out[f"liq_count@{W}s"] = qi["liq_count"].rolling(win, min_periods=1).sum()
            out[f"liq_notional_sum@{W}s"] = qi["liq_notional"].rolling(win, min_periods=1).sum()
            out[f"liq_max@{W}s"] = qi["liq_notional"].rolling(win, min_periods=1).max()
            out[f"liq_buy_sum@{W}s"] = qi["liq_buy"].rolling(win, min_periods=1).sum()
            out[f"liq_sell_sum@{W}s"] = qi["liq_sell"].rolling(win, min_periods=1).sum()
            out[f"liq_imbalance@{W}s"] = (
                out[f"liq_buy_sum@{W}s"] - out[f"liq_sell_sum@{W}s"]
            ) / (out[f"liq_buy_sum@{W}s"] + out[f"liq_sell_sum@{W}s"] + EPS)
            out[f"liq_big_count@{W}s"] = qi["liq_big"].rolling(win, min_periods=1).sum()
            out[f"liq_big_ratio@{W}s"] = (
                out[f"liq_big_count@{W}s"] / (out[f"liq_count@{W}s"] + EPS)
            )

        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["timestamp"])

    ext = pd.concat(parts, axis=1)
    ext = ext.sort_index().ffill().bfill()
    ext = ext.reset_index().rename(columns={"index": "timestamp"})
    return ext
