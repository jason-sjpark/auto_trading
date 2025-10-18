import json, time, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from websocket import WebSocketApp  # <-- 명시적 임포트 (websocket-client)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "realtime" / "liquidations"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
BUFFER_SIZE = 200
FLUSH_INTERVAL_SEC = 10

def _utc_ms_to_naive(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)

def _chunk_dir(symbol: str) -> Path:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    d = OUT_ROOT / symbol / day
    d.mkdir(parents=True, exist_ok=True)
    return d

def _chunk_path(base_dir: Path) -> Path:
    stamp = datetime.utcnow().strftime("%H%M%S")
    seq = 0
    while True:
        p = base_dir / f"part-{stamp}-{seq:03d}.parquet"
        if not p.exists():
            return p
        seq += 1

def _flush(buffer, symbol: str):
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    buffer.clear()
    out = _chunk_path(_chunk_dir(symbol))
    df.to_parquet(out, index=False, compression="snappy")
    print(f"[WS] 💾 saved {len(df):,} rows → {out}")

def collect_liquidations(symbol: str = SYMBOL):
    url = f"wss://fstream.binance.com/ws/{symbol.lower()}@forceOrder"
    buffer = []
    last_flush = time.monotonic()

    def on_message(ws, message):
        nonlocal last_flush
        try:
            d = json.loads(message)
            o = d.get("o", {})
            ts = _utc_ms_to_naive(o.get("T"))
            side = str(o.get("S", "")).lower()
            price = pd.to_numeric(o.get("ap"), errors="coerce")
            qty   = pd.to_numeric(o.get("q"), errors="coerce")
            buffer.append({"timestamp": ts, "symbol": o.get("s", symbol), "side": side, "price": price, "qty": qty})

            if len(buffer) >= BUFFER_SIZE:
                _flush(buffer, symbol)
                last_flush = time.monotonic()

            now = time.monotonic()
            if now - last_flush >= FLUSH_INTERVAL_SEC:
                _flush(buffer, symbol)
                last_flush = now
        except Exception as e:
            print("parse err:", e)

    def on_open(ws): print("ws open", url)
    def on_error(ws, err): print("ws err:", err)
    def on_close(ws, a=None, b=None): print("ws closed")

    while True:
        try:
            ws = WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("ws top-level err:", e)
        finally:
            _flush(buffer, symbol)
        time.sleep(2)

if __name__ == "__main__":
    sym = SYMBOL if len(sys.argv) < 2 else sys.argv[1]
    collect_liquidations(sym)
