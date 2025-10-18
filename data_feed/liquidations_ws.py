# data_feed/liquidations_ws.py
import json, time, threading
from pathlib import Path
import pandas as pd
import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "liquidations_ws"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _ts_to_naive_utc(ms: int):
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)

def collect_liquidations(symbol="BTCUSDT"):
    stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@forceOrder"
    buffer = []

    def on_message(ws, message):
        try:
            d = json.loads(message)
            # 예시 페이로드: {'o': {'s': 'BTCUSDT','S': 'SELL','ap': '60234.12','q': '0.432','T': 1699999999999, ...}}
            o = d.get("o", {})
            ts = _ts_to_naive_utc(o.get("T"))
            side = str(o.get("S", "")).lower()
            price = pd.to_numeric(o.get("ap"), errors="coerce")
            qty   = pd.to_numeric(o.get("q"), errors="coerce")
            row = {"timestamp": ts, "symbol": o.get("s"), "side": side, "price": price, "qty": qty}
            buffer.append(row)
            # 배치 저장 (예: 100건마다)
            if len(buffer) >= 100:
                df = pd.DataFrame(buffer)
                buffer.clear()
                out = OUT_DIR / f"{symbol}.parquet"
                if out.exists():
                    old = pd.read_parquet(out)
                    df = pd.concat([old, df], ignore_index=True)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates()
                df.to_parquet(out, index=False)
                print(f"[WS] saved {len(df)} rows → {out}")
        except Exception as e:
            print("parse err:", e)

    def on_error(ws, err): print("ws err:", err)
    def on_close(ws, a,b): print("ws closed")
    def on_open(ws): print("ws open", stream)

    while True:
        try:
            ws = websocket.WebSocketApp(stream, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("ws top-level err:", e)
        time.sleep(2)

if __name__ == "__main__":
    collect_liquidations("BTCUSDT")
