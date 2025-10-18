import pandas as pd
import numpy as np

EPS = 1e-12

def _to_utc_naive(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def _to_1s_index(df: pd.DataFrame, cols):
    """
    - timestamp → UTC-naive datetime
    - 1초 리샘플(last) + ffill/bfill
    - 반환: DatetimeIndex(초단위) 유지 (← 시간기반 rolling 가능)
    """
    d = df.copy()
    d["timestamp"] = _to_utc_naive(d["timestamp"])
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d = d.set_index("timestamp").resample("1s").last().ffill().bfill()
    return d  # DatetimeIndex 그대로 유지

def build_external_features(inputs: dict) -> pd.DataFrame:
    """
    inputs keys (optional):
      funding(timestamp,symbol,funding_rate)
      open_interest(timestamp,symbol,open_interest)
      long_short_ratio(timestamp,symbol,long_short_ratio,long_account,short_account)
      index_mark(timestamp,symbol,mark_price,index_price,last_funding_rate)
      arbitrage(timestamp,symbol,binance_mark,okx_mark,arb_spread)
      liquidations(...)  # 현재 미사용
    """
    # --- 베이스 타임라인(1초 그리드) 만들기
    frames = []
    for k, df in inputs.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["timestamp"] = _to_utc_naive(d["timestamp"])
        d = d.dropna(subset=["timestamp"])
        frames.append(d[["timestamp"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp"])

    base = pd.concat(frames, axis=0).drop_duplicates().sort_values("timestamp")
    base = base.set_index("timestamp").resample("1s").last()
    out = base.copy()  # DatetimeIndex 유지

    # --- Funding rate: 1초 그리드 + zscore(정수윈도우=3600초)
    fr = inputs.get("funding")
    if fr is not None and not fr.empty:
        fr1 = _to_1s_index(fr, ["funding_rate"])
        win = 3600  # 3600초 = 1시간
        mean = fr1["funding_rate"].rolling(win, min_periods=60).mean()
        std  = fr1["funding_rate"].rolling(win, min_periods=60).std()
        fr1["funding_rate_z"] = (fr1["funding_rate"] - mean) / (std + EPS)
        out = out.join(fr1[["funding_rate","funding_rate_z"]], how="left")

    # --- Open interest: 변화량/퍼센트 변화
    oi = inputs.get("open_interest")
    if oi is not None and not oi.empty:
        oi1 = _to_1s_index(oi, ["open_interest"])
        oi1["oi_change"]     = oi1["open_interest"].diff().fillna(0.0)
        oi1["oi_change_pct"] = oi1["open_interest"].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out = out.join(oi1[["open_interest","oi_change","oi_change_pct"]], how="left")

    # --- Long/Short ratio: zscore(1h=3600초)
    lsr = inputs.get("long_short_ratio")
    if lsr is not None and not lsr.empty:
        lsr1 = _to_1s_index(lsr, ["long_short_ratio","long_account","short_account"])
        win = 3600
        mean = lsr1["long_short_ratio"].rolling(win, min_periods=60).mean()
        std  = lsr1["long_short_ratio"].rolling(win, min_periods=60).std()
        lsr1["lsr_z"] = (lsr1["long_short_ratio"] - mean) / (std + EPS)
        out = out.join(lsr1[["long_short_ratio","long_account","short_account","lsr_z"]], how="left")

    # --- Index/Mark
    im = inputs.get("index_mark")
    if im is not None and not im.empty:
        im1 = _to_1s_index(im, ["mark_price","index_price","last_funding_rate"])
        out = out.join(im1[["mark_price","index_price","last_funding_rate"]], how="left")

    # --- Arbitrage
    arb = inputs.get("arbitrage")
    if arb is not None and not arb.empty:
        arb1 = _to_1s_index(arb, ["binance_mark","okx_mark","arb_spread"])
        out = out.join(arb1[["binance_mark","okx_mark","arb_spread"]], how="left")

    # 최종 정리: index→column
    out = out.sort_index().ffill().bfill().reset_index().rename(columns={"index":"timestamp"})
    return out
