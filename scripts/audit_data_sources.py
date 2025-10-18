import json, os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "ohlcv_1s": "data/processed/ohlcv_1s_*.parquet",
    "trades_raw": "data/raw/aggTrades/**/*.parquet",
    "depth_raw":  "data/raw/depth/**/*.parquet",
    "trades_rt":  "data/realtime/aggTrades/**/*.parquet",
    "depth_rt":   "data/realtime/depth/**/*.parquet",
    "liq_rt":     "data/realtime/liquidations/**/*.parquet",
    "funding":         "data/external/funding_rates_*.parquet",
    "open_interest":   "data/external/open_interest_*.parquet",
    "long_short_ratio":"data/external/long_short_ratio_*.parquet",
    "liquidations":    "data/external/liquidations_*.parquet",
    "index_mark":      "data/external/index_mark_*.parquet",
    "arbitrage":       "data/external/arbitrage_spreads_*.parquet",
}

def _glob(pattern: str):
    base = ROOT
    return sorted(base.rglob(pattern.split("/",1)[1])) if pattern.startswith("data/") else sorted(base.rglob(pattern))

def _candidate_files(pattern: str):
    files = _glob(pattern)
    return files

def _rows_fast(path: Path) -> int:
    try:
        md = pq.read_metadata(path)
        return sum(md.row_group(i).num_rows for i in range(md.num_row_groups))
    except Exception:
        try:
            return len(pd.read_parquet(path, columns=None))
        except Exception:
            return -1

def _pick_latest_nonempty(pattern: str):
    files = _candidate_files(pattern)
    if not files:
        return None, None
    # 최신부터 역순 검사
    for f in reversed(files):
        if os.path.getsize(f) > 0:
            rows = _rows_fast(f)
            if rows and rows > 0:
                return f, rows
    # 전부 비어있으면 최신 파일이라도 반환
    f = files[-1]
    return f, _rows_fast(f)

def _summary(path: Path, pre_rows: int = None):
    try:
        df = pd.read_parquet(path)
        rows = pre_rows if pre_rows is not None else len(df)
        cols = df.shape[1]
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            rng = [str(ts.min()), str(ts.max())]
        else:
            rng = ["NaT","NaT"]
        cols_list = list(map(str, df.columns))[:20]
    except Exception as e:
        return {"error": f"read fail: {e}", "path": str(path)}
    return {"rows": int(rows), "cols": int(cols), "range": rng, "columns": cols_list, "path": str(path)}

def main():
    report = {}  
    for key, pat in PATTERNS.items():
        f, rows = _pick_latest_nonempty(pat)
        if f is None:
            report[key] = {"exists": False, "detail": f"missing file matching {pat}"}
        else:
            report[key] = {"exists": True, "detail": _summary(f, pre_rows=rows)}
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
