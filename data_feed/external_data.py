# data_feed/external_data.py
import time
from pathlib import Path
from typing import List, Optional, Union

import requests
import pandas as pd
import numpy as np

BASE_BINANCE = "https://fapi.binance.com"
BASE_OKX = "https://www.okx.com"
OUTDIR = Path("data/external")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 날짜 변환: unit 제어(기본 ms)
# -----------------------------
def _to_dt(
    x: Union[pd.Series, pd.DatetimeIndex, pd.Timestamp, int, float, str],
    unit: str = "ms",
):
    """
    - Binance/OKX는 보통 ms 단위 epoch → unit='ms'로 고정
    - Series → .dt.tz_convert, Index → .tz_convert, Timestamp → .tz_convert
    - 결과는 tz-naive(UTC 기준)로 반환
    """
    y = pd.to_datetime(x, utc=True, errors="coerce", unit=unit)
    if isinstance(y, pd.Series):
        return y.dt.tz_convert("UTC").dt.tz_localize(None)
    if isinstance(y, pd.DatetimeIndex):
        return y.tz_convert("UTC").tz_localize(None)
    if pd.isna(y):
        return y
    return y.tz_convert("UTC").tz_localize(None)

# -----------------------------
# 요청 유틸 (간단 재시도)
# -----------------------------
def _get(url: str, params: dict = None, timeout: int = 15, retries: int = 3, sleep: float = 0.6):
    last = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last

def _as_num(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def _ensure_schema(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """필요 컬럼이 비어도 스키마는 맞춰서 반환"""
    if df is None or df.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

def save_parquet(df: pd.DataFrame, name: str):
    """
    ✅ 빈 DF라도 스키마를 유지해 파일로 저장한다.
    audit_data_sources가 존재 여부를 확실하게 잡을 수 있도록.
    """
    p = OUTDIR / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        # 아예 None이면 저장하지 않음
        return None
    df.to_parquet(p, index=False)
    return str(p)

# -----------------------------
# 1) 펀딩비 (Funding Rate)
# -----------------------------
def fetch_funding_rates(symbol: str, limit: int = 1000) -> pd.DataFrame:
    url = f"{BASE_BINANCE}/fapi/v1/fundingRate"
    r = _get(url, params={"symbol": symbol, "limit": limit})
    js = r.json()
    df = pd.DataFrame(js)
    if df.empty:
        return _ensure_schema(None, ["timestamp","symbol","funding_rate"])
    df["timestamp"] = _to_dt(df["fundingTime"], unit="ms")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["symbol"] = df.get("symbol", symbol)
    df = df[["timestamp","symbol","funding_rate"]].dropna(subset=["timestamp"]).sort_values("timestamp")
    return _ensure_schema(df, ["timestamp","symbol","funding_rate"])

# -----------------------------
# 2) 미결제약정 (Open Interest)
# -----------------------------
def fetch_open_interest(symbol: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_BINANCE}/futures/data/openInterestHist"
    r = _get(url, params={"symbol": symbol, "period": interval, "limit": limit})
    js = r.json()
    df = pd.DataFrame(js)
    if df.empty:
        return _ensure_schema(None, ["timestamp","symbol","open_interest"])
    df["timestamp"] = _to_dt(df["timestamp"], unit="ms")
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["symbol"] = df.get("symbol", symbol)
    df = df[["timestamp","symbol","open_interest"]].dropna(subset=["timestamp"]).sort_values("timestamp")
    return _ensure_schema(df, ["timestamp","symbol","open_interest"])

# -----------------------------
# 3) 롱/숏 비율
# -----------------------------
def fetch_long_short_ratio(symbol: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_BINANCE}/futures/data/globalLongShortAccountRatio"
    r = _get(url, params={"symbol": symbol, "period": interval, "limit": limit})
    js = r.json()
    df = pd.DataFrame(js)
    if df.empty:
        return _ensure_schema(None, ["timestamp","symbol","long_short_ratio","long_account","short_account"])
    df["timestamp"] = _to_dt(df["timestamp"], unit="ms")
    # 바이낸스 스펙: longShortRatio/longAccount/shortAccount → 문자열 숫자
    df["long_short_ratio"] = pd.to_numeric(df.get("longShortRatio"), errors="coerce")
    df["long_account"]     = pd.to_numeric(df.get("longAccount"), errors="coerce")
    df["short_account"]    = pd.to_numeric(df.get("shortAccount"), errors="coerce")
    df["symbol"] = df.get("symbol", symbol)
    df = df[["timestamp","symbol","long_short_ratio","long_account","short_account"]].dropna(subset=["timestamp"]).sort_values("timestamp")
    return _ensure_schema(df, ["timestamp","symbol","long_short_ratio","long_account","short_account"])

# -----------------------------
# 4) 인덱스/마크 가격 (스냅샷)
# -----------------------------
def fetch_index_mark_price(symbol: str) -> pd.DataFrame:
    url = f"{BASE_BINANCE}/fapi/v1/premiumIndex"
    js = _get(url, params={"symbol": symbol}).json()
    ts = _to_dt(int(js["time"]), unit="ms")
    row = {
        "timestamp": ts,
        "symbol": js.get("symbol", symbol),
        "mark_price": _as_num(js.get("markPrice")),
        "index_price": _as_num(js.get("indexPrice")),
        "last_funding_rate": _as_num(js.get("lastFundingRate", 0.0)),
    }
    df = pd.DataFrame([row]).dropna(subset=["timestamp"]).sort_values("timestamp")
    return _ensure_schema(df, ["timestamp","symbol","mark_price","index_price","last_funding_rate"])

# -----------------------------
# 5) 강제청산 (Force Orders; MARKET_DATA)
# -----------------------------
def fetch_liquidations(
    symbol: Optional[str] = None,
    hours_sequence: List[int] = (6, 24, 72, 168),  # 6h → 1d → 3d → 7d 탐색
) -> pd.DataFrame:
    """
    - Binance 공용 청산: /fapi/v1/allForceOrders
    - 시간 범위가 없으면 400/빈응답이 자주 발생 → 점증적으로 시간창 확대
    - 최종 fallback: 시간 파라미터 제거 + limit=1000 (심볼/무심볼 둘 다 시도)
    """
    url = f"{BASE_BINANCE}/fapi/v1/allForceOrders"

    def _call_with_range(sym: Optional[str], hours: int):
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - hours * 60 * 60 * 1000
        params = {"startTime": start_ms, "endTime": end_ms}
        if sym:
            params["symbol"] = sym
        js = _get(url, params=params).json()
        df = pd.DataFrame(js)
        return df

    def _normalize(df: pd.DataFrame, sym: Optional[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return _ensure_schema(None, ["timestamp","symbol","side","price","qty"])
        df["timestamp"] = _to_dt(df.get("time"), unit="ms")
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        qty = pd.to_numeric(df.get("origQty"), errors="coerce")
        if qty is None or (isinstance(qty, pd.Series) and qty.isna().all()):
            qty = pd.to_numeric(df.get("executedQty"), errors="coerce")
        df["qty"] = qty
        side = df.get("side")
        if side is None:
            side = df.get("forceOrderType")
        df["side"] = side.astype(str).str.lower()
        if "symbol" not in df.columns:
            df["symbol"] = sym if sym else ""
        out = df[["timestamp","symbol","side","price","qty"]].dropna(subset=["timestamp"]).sort_values("timestamp")
        return _ensure_schema(out, ["timestamp","symbol","side","price","qty"])

    # 1) 시간창 확대 시도 (심볼 포함)
    for h in hours_sequence:
        try:
            df = _call_with_range(symbol, h)
            norm = _normalize(df, symbol)
            if not norm.empty:
                return norm
        except requests.HTTPError:
            continue

    # 2) 시간창 확대 시도 (심볼 제거)
    for h in hours_sequence:
        try:
            df = _call_with_range(None, h)
            norm = _normalize(df, None)
            if not norm.empty:
                return norm
        except requests.HTTPError:
            continue

    # 3) 최종 fallback: 시간 없이 limit=1000 (심볼 → 무심볼)
    for sym in (symbol, None):
        try:
            params = {"limit": 1000}
            if sym:
                params["symbol"] = sym
            js = _get(url, params=params).json()
            df = _normalize(pd.DataFrame(js), sym)
            if not df.empty:
                return df
        except Exception:
            continue

    # 완전 실패 → 스키마만 반환(빈 DF)
    return _ensure_schema(None, ["timestamp","symbol","side","price","qty"])

# -----------------------------
# 6) 거래소간 차익(Arbitrage Spread) 예시
# -----------------------------
def _okx_swap_symbol(binance_sym: str) -> str:
    if binance_sym.endswith("USDT"):
        base = binance_sym[:-4]
        return f"{base}-USDT-SWAP"
    if binance_sym.endswith("USD"):
        base = binance_sym[:-3]
        return f"{base}-USD-SWAP"
    return f"{binance_sym}-USDT-SWAP"

def fetch_multi_exchange_prices(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        b = _get(f"{BASE_BINANCE}/fapi/v1/premiumIndex", params={"symbol": sym}).json()
        b_price = _as_num(b.get("markPrice"))
        ts = _to_dt(int(b["time"]), unit="ms")
        okx_sym = _okx_swap_symbol(sym)
        o = _get(f"{BASE_OKX}/api/v5/public/mark-price", params={"instType":"SWAP","instId": okx_sym}).json()
        o_price = np.nan
        try:
            if isinstance(o, dict) and o.get("data"):
                o_price = _as_num(o["data"][0]["markPx"])
        except Exception:
            pass
        rows.append({"timestamp": ts, "symbol": sym, "binance_mark": b_price, "okx_mark": o_price})
    df = pd.DataFrame(rows).dropna(subset=["timestamp"]).sort_values("timestamp")
    df["arb_spread"] = df["binance_mark"] - df["okx_mark"]
    return _ensure_schema(df, ["timestamp","symbol","binance_mark","okx_mark","arb_spread"])

# -----------------------------
# CLI 실행: 예시 저장
# -----------------------------
if __name__ == "__main__":
    sym = "BTCUSDT"
    print("funding:", save_parquet(fetch_funding_rates(sym),          f"funding_rates_{sym}"))
    print("oi:",      save_parquet(fetch_open_interest(sym),          f"open_interest_{sym}"))
    print("lsr:",     save_parquet(fetch_long_short_ratio(sym),       f"long_short_ratio_{sym}"))
    print("idxmark:", save_parquet(fetch_index_mark_price(sym),       f"index_mark_{sym}"))
    print("liq:",     save_parquet(fetch_liquidations(sym),           f"liquidations_{sym}"))
    print("arb:",     save_parquet(fetch_multi_exchange_prices([sym]),f"arbitrage_spreads_{sym}"))
