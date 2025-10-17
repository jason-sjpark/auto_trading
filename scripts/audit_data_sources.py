# scripts/audit_data_sources.py
import os, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "ohlcv_1s": "data/processed/ohlcv_1s_*.parquet",
    "trades": "data/raw/aggTrades_*.parquet",
    "depth": "data/raw/depth_*.parquet",
    "funding": "data/external/funding_rates_*.parquet",
    "open_interest": "data/external/open_interest_*.parquet",
    "long_short_ratio": "data/external/long_short_ratio_*.parquet",
    "liquidations": "data/external/liquidations_*.parquet",
    "index_mark": "data/external/index_mark_*.parquet",
    "arbitrage": "data/external/arbitrage_spreads_*.parquet",
}

def glob_one(pattern: str):
    return sorted(ROOT.glob(pattern))[-1] if list(ROOT.glob(pattern)) else None

def summary(path: Path):
    try:
        df = pd.read_parquet(path)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return {"error": f"read fail: {e}"}
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        rng = (str(ts.min()), str(ts.max()))
    else:
        rng = ("NA", "NA")
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "range": rng,
        "columns": list(df.columns)[:20],
        "path": str(path),
    }

def main():
    report = {}
    for key, pat in DATA.items():
        f = glob_one(pat)
        if f is None:
            report[key] = {"exists": False, "detail": f"missing file matching {pat}"}
        else:
            report[key] = {"exists": True, "detail": summary(f)}
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
