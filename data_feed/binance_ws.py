import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==============================================================
# 설정
# ==============================================================
SYMBOL = "btcusdt"
WS_URL = f"wss://fstream.binance.com/stream?streams={SYMBOL}@aggTrade/{SYMBOL}@depth10@100ms"

# 데이터 저장 위치
DATA_PATH = Path("./data/raw")
DATA_PATH.mkdir(parents=True, exist_ok=True)

# 버퍼 크기 (예: 1000건마다 저장)
TRADE_BUFFER_SIZE = 1000
DEPTH_BUFFER_SIZE = 300

trade_buffer = []
depth_buffer = []

# ==============================================================
# 유틸 함수
# ==============================================================

def timestamp_ms_to_str(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def save_to_parquet(buffer, filename: str):
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    path = DATA_PATH / f"{filename}_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    if path.exists():
        # append 모드
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)
    print(f"💾 Saved {len(buffer)} rows → {path}")
    buffer.clear()

# ==============================================================
# WebSocket Listener
# ==============================================================

async def listen_binance():
    global trade_buffer, depth_buffer

    print("🚀 Connecting to Binance WebSocket...")
    async with websockets.connect(WS_URL, ping_interval=20) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            stream = data.get("stream")
            payload = data.get("data")

            # -------------------------------
            # aggTrade (체결 데이터)
            # -------------------------------
            if "aggTrade" in stream:
                record = {
                    "timestamp": timestamp_ms_to_str(payload["T"]),
                    "trade_id": payload["a"],
                    "price": float(payload["p"]),
                    "qty": float(payload["q"]),
                    "is_buyer_maker": payload["m"],
                    "side": "sell" if payload["m"] else "buy"
                }
                trade_buffer.append(record)

                if len(trade_buffer) >= TRADE_BUFFER_SIZE:
                    save_to_parquet(trade_buffer, "aggTrades")

            # -------------------------------
            # depth (호가창)
            # -------------------------------
            elif "depth" in stream:
                record = {
                    "timestamp": timestamp_ms_to_str(payload["E"]),
                    "bids": payload["b"][:10],
                    "asks": payload["a"][:10]
                }
                depth_buffer.append(record)

                if len(depth_buffer) >= DEPTH_BUFFER_SIZE:
                    save_to_parquet(depth_buffer, "depth")

            # progress 표시
            if len(trade_buffer) % 100 == 0 or len(depth_buffer) % 50 == 0:
                print(f"📡 Trades: {len(trade_buffer)} | Depth: {len(depth_buffer)}")

# ==============================================================
# 실행 진입점
# ==============================================================

if __name__ == "__main__":
    try:
        asyncio.run(listen_binance())
    except KeyboardInterrupt:
        print("🛑 Stopped by user")
        save_to_parquet(trade_buffer, "aggTrades")
        save_to_parquet(depth_buffer, "depth")
        print("✅ Graceful shutdown complete.")
