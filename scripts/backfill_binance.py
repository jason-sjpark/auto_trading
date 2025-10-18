# scripts/backfill_binance.py
import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import zipfile

try:
    from data_feed.ohlcv_builder import resample_ohlcv
except Exception:
    resample_ohlcv = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

VISION_BASE = "https://data.binance.vision/data/futures/um/daily"
FAPI_BASE = "https://fapi.binance.com"

def log(msg: str, verbose: bool):
    if verbose:
        print(msg)

def daterange(start_date: str, end_date: str) -> List[pd.Timestamp]:
    s = pd.to_datetime(start_date).normalize()
    e = pd.to_datetime(end_date).normalize()
    return list(pd.date_range(s, e, freq="D"))

def _req(url: str, timeout=30) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None

def _to_dt_ms(x):
    """
    ms epoch → tz-aware UTC → tz-naive 로 변환.
    Series/Scalar 모두 안전 처리.
    """
    s = pd.to_datetime(x, unit="ms", utc=True, errors="coerce")
    if isinstance(s, pd.Series):
        return s.dt.tz_localize(None)
    else:
        return s.tz_localize(None)


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _parquet_safe(df: pd.DataFrame, path: Path):
    _ensure_dir(path)
    df.to_parquet(path, index=False)

# ------------------------
# ZIP → DataFrame (파일별 로깅/리트라이는 여기서)
# ------------------------
def read_zip_to_df(content: bytes, verbose: bool=False) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        parts = []
        names = sorted([n for n in zf.namelist() if not n.endswith("/")])
        if verbose:
            print(f"  • zip entries: {len(names)} files")

        for name in names:
            if verbose:
                print(f"    - reading {name}")
            with zf.open(name) as f:
                data = f.read()

                # 1) CSV(헤더 가정)
                try:
                    buf = io.BytesIO(data)
                    df = pd.read_csv(buf)
                    if verbose:
                        print(f"      csv header={list(df.columns)} rows={len(df)}")
                    parts.append(df)
                    continue
                except Exception as e:
                    if verbose:
                        print(f"      csv(header) fail: {e}")

                # 2) CSV(헤더 없음) → 7열일 가능성 높음 (trades 표준)
                try:
                    buf = io.BytesIO(data)
                    df = pd.read_csv(buf, header=None)
                    if verbose:
                        print(f"      csv(no-header) cols={df.shape[1]} rows={len(df)}")
                    parts.append(df)
                    continue
                except Exception as e:
                    if verbose:
                        print(f"      csv(no-header) fail: {e}")

                # 3) JSON Lines
                try:
                    lines = [json.loads(line) for line in data.splitlines() if line.strip()]
                    if lines:
                        df = pd.DataFrame(lines)
                        if verbose:
                            print(f"      jsonl rows={len(df)}")
                        parts.append(df)
                        continue
                except Exception as e:
                    if verbose:
                        print(f"      jsonl fail: {e}")

        if not parts:
            if verbose:
                print("  • zip had no readable parts")
            return pd.DataFrame()

        out = pd.concat(parts, ignore_index=True)
        if verbose:
            print(f"  • concat rows={len(out)} cols={list(out.columns)}")
        return out

# ------------------------
# trades / aggTrades 파서 (대소문자/무헤더 대응)
# ------------------------
def parse_trades_like_csv(df: pd.DataFrame, symbol: str, verbose: bool=False) -> pd.DataFrame:
    # 무헤더일 수 있음 → 열 이름 소문자 사본
    cols = [str(c) for c in df.columns]
    lower = [c.lower() for c in cols]
    colmap = {lower[i]: cols[i] for i in range(len(cols))}

    def pick(*cands) -> Optional[str]:
        for c in cands:
            if c.lower() in colmap:
                return colmap[c.lower()]
        return None

    # time(timestamp)
    tcol = pick("time", "timestamp", "T", "E")
    ts = None
    if tcol is not None:
        ts_num = pd.to_numeric(df[tcol], errors="coerce")
        ts = pd.to_datetime(ts_num, unit="ms", utc=True, errors="coerce")
    else:
        # 무헤더 CSV: 7열 가정 → [id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch]
        if df.shape[1] >= 5:
            tcol_idx = 4
            try:
                ts_num = pd.to_numeric(df.iloc[:, tcol_idx], errors="coerce")
                ts = pd.to_datetime(ts_num, unit="ms", utc=True, errors="coerce")
                if verbose:
                    print("      inferred time from 5th column (headerless)")
            except Exception:
                ts = pd.Series(pd.NaT, index=df.index)
        else:
            # 마지막 시도: 각 컬럼 스캔하여 ms epoch 같은 값 찾기
            ts = pd.Series(pd.NaT, index=df.index)
            for i in range(df.shape[1]):
                s = pd.to_numeric(df.iloc[:, i], errors="coerce")
                if s.notna().sum() > 0 and (s > 1e12).sum() > 10:
                    ts = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
                    if verbose:
                        print(f"      inferred time from column {i} by epoch scan")
                    break
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None) if hasattr(ts, "dt") else ts

    # price / qty
    pcol = pick("price", "p")
    qcol = pick("qty", "quantity", "q")
    price = pd.to_numeric(df[pcol], errors="coerce") if pcol else (pd.to_numeric(df.iloc[:,1], errors="coerce") if df.shape[1] > 1 else pd.Series(np.nan, index=df.index))
    qty   = pd.to_numeric(df[qcol], errors="coerce") if qcol else (pd.to_numeric(df.iloc[:,2], errors="coerce") if df.shape[1] > 2 else pd.Series(np.nan, index=df.index))

    # id
    idcol = pick("id", "aggTradeId", "a")
    tid = pd.to_numeric(df[idcol], errors="coerce") if idcol else (pd.to_numeric(df.iloc[:,0], errors="coerce") if df.shape[1] > 0 else pd.Series(np.nan, index=df.index))
    tid = tid.fillna(-1).astype(np.int64)

    # isBuyerMaker
    ibmcol = pick("isBuyerMaker", "m")
    ibm = df[ibmcol].astype(bool) if ibmcol else (df.iloc[:,5].astype(bool) if df.shape[1] > 5 else pd.Series(False, index=df.index, dtype=bool))

    out = pd.DataFrame({
        "timestamp": ts,
        "trade_id": tid,
        "price": price,
        "qty": qty,
        "is_buyer_maker": ibm,
    })
    out["side"] = np.where(out["is_buyer_maker"], "sell", "buy")
    before = len(out)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    if verbose:
        print(f"  • trades parsed: before={before}, after_ts_drop={len(out)}")
    return out

# ------------------------
# bookDepth 파서 (대소문자/JSONL 방어)
# ------------------------
def _row_to_levels(v) -> List[List[float]]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            arr = json.loads(v)
        except Exception:
            return []
    else:
        arr = v
    out = []
    for x in arr:
        try:
            p = float(x[0]); q = float(x[1])
            out.append([p, q])
        except Exception:
            continue
    return out

# scripts/backfill_binance.py 내 parse_bookdepth() 교체
def parse_bookdepth(df: pd.DataFrame, verbose: bool=False) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    lower = [c.lower() for c in cols]
    colmap = {lower[i]: cols[i] for i in range(len(cols))}
    def pick(*cands):
        for c in cands:
            if c.lower() in colmap:
                return colmap[c.lower()]
        return None

    # 1) 전통 스냅샷(bids/asks) 스키마 처리
    bcol = pick("bids"); acol = pick("asks")
    tcol = pick("time","timestamp","e","eventtime","t","transactiontime")
    if bcol and acol:
        ts = _to_dt_ms(pd.to_numeric(df[tcol], errors="coerce")) if tcol else pd.Series(pd.NaT, index=df.index)
        out = pd.DataFrame({
            "timestamp": ts,
            "bids": df[bcol].apply(_row_to_levels),
            "asks": df[acol].apply(_row_to_levels),
        })
        before = len(out)
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        if verbose: print(f"  • depth parsed(bids/asks): before={before}, after={len(out)}")
        return out

    # 2) 집계지표 스키마(percentage/depth/notional) 처리
    pcol = pick("percentage")
    dcol = pick("depth")
    ncol = pick("notional")
    tcol = pick("time","timestamp")
    if pcol and (dcol or ncol):
        if verbose: print("  • depth aggregated schema detected (percentage/depth/notional) → pivot wide")
        df2 = df[[tcol, pcol] + ([dcol] if dcol else []) + ([ncol] if ncol else [])].copy()
        raw_ts = df2[tcol]
        if pd.api.types.is_numeric_dtype(raw_ts):
            ts = _to_dt_ms(raw_ts)
        else:
            # 문자열/ISO 등 일반 포맷도 수용
            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
            if isinstance(ts, pd.Series):
                ts = ts.dt.tz_localize(None)
            else:
                ts = ts.tz_localize(None)
        df2[tcol] = ts
        # 퍼센트 레벨을 문자열 컬럼명으로 안전화 (예: '1', '2', '-1' 등)
        df2["_pct"] = df2[pcol].astype(str).str.replace("%","", regex=False)
        df2 = df2.dropna(subset=[tcol])
        # 와이드 피벗: timestamp 인덱스, 열은 pct별 depth/notional
        piv_parts = []
        if dcol:
            piv_d = df2.pivot_table(index=tcol, columns="_pct", values=dcol, aggfunc="last")
            piv_d = piv_d.add_prefix("depth_pct_")
            piv_parts.append(piv_d)
        if ncol:
            piv_n = df2.pivot_table(index=tcol, columns="_pct", values=ncol, aggfunc="last")
            piv_n = piv_n.add_prefix("notional_pct_")
            piv_parts.append(piv_n)
        if not piv_parts:
            return pd.DataFrame(columns=["timestamp","bids","asks"])
        wide = pd.concat(piv_parts, axis=1).sort_index().reset_index().rename(columns={tcol: "timestamp"})
        if verbose: print(f"  • depth aggregated wide rows={len(wide)} cols={len(wide.columns)}")
        return wide  # ⚠️ 이 경우 'bids/asks' 대신 퍼센트별 컬럼이 저장됨

    if verbose: print("  • depth: no recognized columns (return empty)")
    return pd.DataFrame(columns=["timestamp","bids","asks"])


# ------------------------
# Liquidations (REST allForceOrders) – 5분 청크 + 페이지네이션 + 로그
# ------------------------
def fetch_liquidations_day(symbol: str, day: pd.Timestamp, verbose: bool=False) -> pd.DataFrame:
    day = pd.to_datetime(day).tz_localize("UTC")
    start_day_ms = int(day.timestamp() * 1000)
    end_day_ms   = int((day + pd.Timedelta(days=1)).timestamp() * 1000)
    BASE = f"{FAPI_BASE}/fapi/v1/allForceOrders"
    CHUNK_MS = 5 * 60 * 1000

    all_parts = []
    chunk_start = start_day_ms
    while chunk_start < end_day_ms:
        chunk_end = min(chunk_start + CHUNK_MS, end_day_ms)
        cur_start = chunk_start
        if verbose:
            print(f"  • liq chunk {pd.to_datetime(cur_start, unit='ms')} → {pd.to_datetime(chunk_end, unit='ms')}")
        while cur_start < chunk_end:
            params = {"symbol": symbol, "startTime": cur_start, "endTime": chunk_end}
            last_exc = None
            for attempt in range(4):
                try:
                    r = requests.get(BASE, params=params, timeout=20)
                    if r.status_code in (418, 429):
                        if verbose: print(f"    - rate limit ({r.status_code}), retry {attempt+1}")
                        time.sleep(1.0 * (attempt+1))
                        continue
                    r.raise_for_status()
                    js = r.json()
                    df = pd.DataFrame(js)
                    if df.empty:
                        if verbose: print("    - empty page")
                        break
                    df["timestamp"] = _to_dt_ms(pd.to_numeric(df.get("time"), errors="coerce"))
                    df["price"]     = pd.to_numeric(df.get("price"), errors="coerce")
                    qty = pd.to_numeric(df.get("origQty"), errors="coerce")
                    if qty is None or (isinstance(qty, pd.Series) and qty.isna().all()):
                        qty = pd.to_numeric(df.get("executedQty"), errors="coerce")
                    df["qty"]  = qty
                    side = df.get("side") if "side" in df.columns else df.get("forceOrderType")
                    df["side"] = side.astype(str).str.lower() if side is not None else "unknown"
                    if "symbol" not in df.columns:
                        df["symbol"] = symbol
                    part = df[["timestamp","symbol","side","price","qty"]].dropna(subset=["timestamp"]).sort_values("timestamp")
                    if not part.empty:
                        all_parts.append(part)
                        last_ts_ms = int(part["timestamp"].iloc[-1].value // 10**6)
                        cur_start = last_ts_ms + 1
                        if verbose: print(f"    - got {len(part)} rows, continue from {pd.to_datetime(cur_start, unit='ms')}")
                        time.sleep(0.05)
                        continue
                    else:
                        break
                except Exception as e:
                    last_exc = e
                    if verbose: print(f"    - error {e}, retry {attempt+1}")
                    time.sleep(0.5 * (attempt+1))
                    continue
            break
        chunk_start = chunk_end

    if not all_parts:
        if verbose: print("  • liq: no rows for the day")
        return pd.DataFrame(columns=["timestamp","symbol","side","price","qty"])
    out = pd.concat(all_parts, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp","symbol","side","price","qty"]).sort_values("timestamp")
    if verbose: print(f"  • liq total rows={len(out)}")
    return out

# ------------------------
# 다운로드 + 변환 파이프라인
# ------------------------
def backfill_one_day(symbol: str, day: pd.Timestamp, tasks: List[str], make_ohlcv: bool, verbose: bool=False):
    ymd = day.strftime("%Y-%m-%d")
    ymd_nodash = day.strftime("%Y%m%d")

    # trades 우선, 실패 시 aggTrades
    if "trades" in tasks or "aggTrades" in tasks:
        url_trades = f"{VISION_BASE}/trades/{symbol}/{symbol}-trades-{ymd}.zip"
        url_agg    = f"{VISION_BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{ymd}.zip"
        for used, url in [("trades", url_trades), ("aggTrades", url_agg)]:
            log(f"[{used}] GET {url}", verbose)
            content = _req(url)
            if content is None:
                log(f"[{used}] missing {ymd}", verbose); 
                if used == "trades":
                    continue
                else:
                    print(f"[{used}] missing: {ymd}")
                    break
            raw = read_zip_to_df(content, verbose=verbose)
            std = parse_trades_like_csv(raw, symbol, verbose=verbose)
            outp = RAW_DIR / "aggTrades" / symbol / f"{ymd}.parquet"
            _parquet_safe(std, outp)
            print(f"[{used}] {ymd} rows={len(std)} → {outp}")
            if make_ohlcv and resample_ohlcv is not None and len(std) > 0:
                ticks = std[["timestamp", "price", "qty"]]
                ohlcv_1s = resample_ohlcv(ticks, freq="1S")
                ohlcv_out = PROC_DIR / f"ohlcv_1s_{ymd_nodash}.parquet"
                _parquet_safe(ohlcv_1s, ohlcv_out)
                print(f"[OHLCV 1s] {ymd} rows={len(ohlcv_1s)} → {ohlcv_out}")
            break  # trades or aggTrades 중 하나 성공했으면 다음 섹션으로

    # bookDepth
    if "bookDepth" in tasks:
        url = f"{VISION_BASE}/bookDepth/{symbol}/{symbol}-bookDepth-{ymd}.zip"
        log(f"[bookDepth] GET {url}", verbose)
        content = _req(url)
        if content:
            raw = read_zip_to_df(content, verbose=verbose)
            std = parse_bookdepth(raw, verbose=verbose)
            outp = RAW_DIR / "depth" / symbol / f"{ymd}.parquet"
            _parquet_safe(std, outp)
            print(f"[bookDepth] {ymd} rows={len(std)} → {outp}")
        else:
            print(f"[bookDepth] missing: {ymd}")

    # liquidations (REST)
    if "liquidationSnapshot" in tasks:
        print(f"[liquidations] {ymd} skipped for UM (public historical REST unavailable). Use WS collector for future data.")

# ------------------------
# 메인
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Binance UM backfill → Parquet (scalping-ready)")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--what", type=str, default="trades,bookDepth,liquidationSnapshot",
                    help="comma list: trades,aggTrades,bookDepth,liquidationSnapshot")
    ap.add_argument("--make-ohlcv", action="store_true", help="also build 1s OHLCV from trades per day")
    ap.add_argument("--verbose", action="store_true", help="print detailed logs")
    args = ap.parse_args()

    tasks = [w.strip() for w in args.what.split(",") if w.strip()]
    days = daterange(args.start, args.end)

    print(f"Backfill {args.symbol} {args.start}~{args.end} → {tasks}")
    for d in days:
        backfill_one_day(args.symbol, d, tasks, args.make_ohlcv, verbose=args.verbose)

    print("✔ done.")

if __name__ == "__main__":
    main()
