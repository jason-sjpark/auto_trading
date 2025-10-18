# data_feed/binance_ws.py  (교체본)
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import websockets

# =========================================
# 설정
# =========================================
SYMBOL = "BTCUSDT"  # 대문자 표준화
LEVEL = 10          # depth 레벨: 5/10/20
DEPTH_INTERVAL = "100ms"

# Combined stream (aggTrade + depth{LEVEL}@{INTERVAL})
STREAM = f"{SYMBOL.lower()}@aggTrade/{SYMBOL.lower()}@depth{LEVEL}@{DEPTH_INTERVAL}"
WS_URL = f"wss://fstream.binance.com/stream?streams={STREAM}"

# 저장 루트 (실시간은 realtime 아래, 추후 compact해서 raw로 이동)
ROOT = Path("./data/realtime")
TRADES_DIR = ROOT / "aggTrades" / SYMBOL
DEPTH_DIR  = ROOT / "depth"     / SYMBOL
TRADES_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

# 버퍼 크기: 버퍼 기준 조각 파일로 저장 (append 금지)
TRADE_BUFFER_SIZE = 500
DEPTH_BUFFER_SIZE = 500

# =========================================
# 유틸
# =========================================
def _utc_ms_to_ts(ms: int) -> pd.Timestamp:
    # UTC-aware → tz-naive 로 변환 (파이프라인 표준)
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)

def _chunk_path(base_dir: Path) -> Path:
    # 일자 파티션 디렉터리 (UTC 기준)
    day = datetime.utcnow().strftime("%Y-%m-%d")
    part_dir = base_dir / day
    part_dir.mkdir(parents=True, exist_ok=True)
    # 부분 파일명 (시간+증분)
    stamp = datetime.utcnow().strftime("%H%M%S")
    # 초 단위 안에서 중복 방지 위해 파일 존재 시 시퀀스 올림
    seq = 0
    while True:
        p = part_dir / f"part-{stamp}-{seq:03d}.parquet"
        if not p.exists():
            return p
        seq += 1

def _save_chunk(rows: List[dict], base_dir: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = _chunk_path(base_dir)
    # 압축(snappy), index 제외
    df.to_parquet(out, index=False, compression="snappy")
    print(f"💾 saved {len(rows):,} rows → {out}")
    rows.clear()

# =========================================
# 수집 루프
# =========================================
async def _collect_once():
    trade_buf: List[dict] = []
    depth_buf: List[dict] = []

    print(f"🚀 connect → {WS_URL}")
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            stream = data.get("stream", "")
            payload = data.get("data") or {}

            if "aggTrade" in stream:
                # https://binance-docs.github.io/apidocs/futures/en/#aggregate-trade-streams
                rec = {
                    "timestamp": _utc_ms_to_ts(payload["T"]),
                    "trade_id": int(payload["a"]),
                    "price": float(payload["p"]),
                    "qty": float(payload["q"]),
                    "is_buyer_maker": bool(payload["m"]),
                    "side": "sell" if payload["m"] else "buy",
                }
                trade_buf.append(rec)
                if len(trade_buf) >= TRADE_BUFFER_SIZE:
                    _save_chunk(trade_buf, TRADES_DIR)

            elif "depth" in stream:
                # https://binance-docs.github.io/apidocs/futures/en/#diff-depth-stream
                # combined depth{LEVEL}@{INTERVAL} stream payload uses 'b'/'a' and event time 'E'
                bids = payload.get("b") or payload.get("bids") or []
                asks = payload.get("a") or payload.get("asks") or []
                # 문자열->float 캐스팅 방어
                try:
                    bids = [[float(p), float(q)] for p, q in bids[:LEVEL]]
                except Exception:
                    bids = []
                try:
                    asks = [[float(p), float(q)] for p, q in asks[:LEVEL]]
                except Exception:
                    asks = []

                rec = {
                    "timestamp": _utc_ms_to_ts(payload["E"]),
                    "bids": bids,
                    "asks": asks,
                }
                depth_buf.append(rec)
                if len(depth_buf) >= DEPTH_BUFFER_SIZE:
                    _save_chunk(depth_buf, DEPTH_DIR)

            # 가벼운 진행 로그(버퍼 크기 클 때만 찍기)
            if (len(trade_buf) and len(trade_buf) % 2000 == 0) or (len(depth_buf) and len(depth_buf) % 500 == 0):
                print(f"📡 trade_buf={len(trade_buf):,} depth_buf={len(depth_buf):,}")

async def listen_binance_forever():
    retry = 0
    while True:
        try:
            await _collect_once()
            retry = 0  # 정상 종료 시(거의 없음) 초기화
        except KeyboardInterrupt:
            print("🛑 stopped by user")
            break
        except Exception as e:
            retry += 1
            wait = min(30, 2 ** min(retry, 5))  # 2,4,8,16,32 cap→30
            print(f"⚠️ ws error: {e} → reconnect in {wait}s (attempt {retry})")
            await asyncio.sleep(wait)

# =========================================
# 진입점
# =========================================
if __name__ == "__main__":
    try:
        asyncio.run(listen_binance_forever())
    finally:
        # 마지막 잔여 버퍼는 _collect_once 내부에서만 flush하므로
        # 여기서 따로 flush는 생략 (조각 파일로 이미 여러 번 저장함)
        pass
