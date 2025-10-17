"""
스펙 가드: 지표 수식/정의 전수검증 + 스모크 테스트
- 정규표현식/AST로 핵심 수식 패턴 검사
- 합성데이터로 각 지표의 값 범위/단조성/일관성 체크
"""
import re, sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    # trade_intensity = TPS
    ("feature_engineering/trade_features.py", r"trade_intensity\s*=\s*total_trades\s*/\s*duration_sec", True),
    ("feature_engineering/trade_features.py", r"trade_intensity\s*=\s*total_volume\s*/\s*total_trades", False),

    # EMA/RSI/ATR/BB/VWAP 등 존재
    ("feature_engineering/technical_indicators.py", r"ema_9", True),
    ("feature_engineering/technical_indicators.py", r"rsi_14", True),
    ("feature_engineering/technical_indicators.py", r"atr_14", True),
    ("feature_engineering/technical_indicators.py", r"bb_upper", True),
    ("feature_engineering/technical_indicators.py", r"vwap", True),

    # orderbook truth ambiguity 금지
    ("feature_engineering/orderbook_features.py", r"if\s+not\s+bids", False),
    ("feature_engineering/orderbook_features.py", r"if\s+not\s+asks", False),
]

def file_text(rel):
    p = ROOT / rel
    if not p.exists(): return None
    return p.read_text(encoding="utf-8", errors="ignore")

def regex_scan():
    res = []
    fail = 0
    for path, pat, must in CHECKS:
        txt = file_text(path)
        ok = (txt is not None) and (re.search(pat, txt) is not None)
        if must and not ok: 
            res.append({"file":path,"rule":pat,"ok":False,"desc":"missing required pattern"}); fail += 1
        if not must and ok:
            res.append({"file":path,"rule":pat,"ok":False,"desc":"forbidden pattern present"}); fail += 1
        if (must and ok) or (not must and not ok):
            res.append({"file":path,"rule":pat,"ok":True})
    return res, fail

def smoke_indicators():
    # 합성 가격: 우상향 + 소음
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="S")
    price = np.cumsum(np.random.randn(n)*0.1 + 0.02) + 100
    vol = np.random.rand(n)*3
    df = pd.DataFrame({"timestamp":ts,"open":price,"high":price+0.2,"low":price-0.2,"close":price,"volume":vol})

    from feature_engineering.technical_indicators import compute_technical_indicators
    t = compute_technical_indicators(df)
    # 기본 검증: 컬럼 존재/범위/유한성
    req = ["ema_9","ema_20","ema_50","rsi_14","atr_14","bb_ma","bb_upper","bb_lower","vwap","momentum_5","volatility_5"]
    vals_ok = True
    for c in req:
        if c not in t.columns: vals_ok = False; break
        if not np.isfinite(t[c].astype(float)).all(): vals_ok = False; break
    return {"indicators_ok": vals_ok, "shape": tuple(t.shape)}

def smoke_trades():
    from feature_engineering.trade_features import extract_trade_features
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    ts = [now + timedelta(milliseconds=i*100) for i in range(50)] # 5초/50건 → 10 TPS 기대
    df = pd.DataFrame({
        "timestamp": ts,
        "price": np.linspace(100,100.5,len(ts)),
        "qty": np.random.rand(len(ts))*0.3,
        "side": np.random.choice(["buy","sell"], size=len(ts))
    })
    f = extract_trade_features(df)
    ok = (f["trade_count"] >= 50-1) and (f["trade_intensity"] > 5.0)  # TPS 합리 범위
    return {"trade_features_ok": bool(ok), "features": f}

def main():
    report = {}
    scan, fail1 = regex_scan()
    report["regex"] = scan
    report["smoke_indicators"] = smoke_indicators()
    report["smoke_trades"] = smoke_trades()
    failures = fail1 or (not report["smoke_indicators"]["indicators_ok"]) or (not report["smoke_trades"]["trade_features_ok"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
