# 하루치 features_1s_*.parquet(+risk 컬럼 포함)에 대해

# 의사결정 비율,

# 기대 슬리피지 평균(전체 vs ALLOW),

# volatility 지표,

# 라벨 기반(있다면) 품질 신호
# 를 한 번에 요약해 줍니다.

# 사용법 예:
## risk 컬럼 이미 포함된 파일이면 그대로, 아니면 내부에서 자동 적용
# python -m scripts.eval_slippage_filter_offline --in data/features/features_1s_2025-10-17_risk.parquet --cfg config/risk_v1.yaml --out_csv data/reports/slippage_filter_sample_2025-10-17.csv


import os, argparse
import numpy as np
import pandas as pd

def _load_df(path):
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def _ensure_risk(df, cfg_path):
    """
    risk 컬럼이 없으면 on-the-fly로 S&L 필터 적용.
    """
    need = {"risk_score","decision","expected_slippage_bp","relative_spread_bp","rv_3s_bp"}
    if need.issubset(df.columns):
        return df
    from risk_engine.slippage_filter import SlippageFilter
    f = SlippageFilter(config_path=cfg_path)
    return f.apply(df)

def _ratio(series: pd.Series) -> pd.Series:
    return (series.value_counts(normalize=True)
                 .rename("ratio")
                 .pipe(lambda s: (s*100).round(2).astype(str) + "%"))

def main():
    ap = argparse.ArgumentParser()
    # 'in'은 예약어이므로 dest를 'input'으로 지정 (기존 --in도 유지)
    ap.add_argument("-i","--input","--in", dest="input", required=True,
                    help="features_1s parquet path (risk cols optional)")
    ap.add_argument("--cfg", default="config/risk_v1.yaml",
                    help="S&L yaml config path")
    ap.add_argument("--out-csv","--out_csv", dest="out_csv", default="",
                    help="optional: write sample columns to CSV")
    args = ap.parse_args()

    df = _load_df(args.input)
    df = _ensure_risk(df, args.cfg)

    n = len(df)
    dt0, dt1 = df["timestamp"].min(), df["timestamp"].max()
    print("==== S&L Offline Evaluation ====")
    print(f"Rows: {n} | Range: {dt0} → {dt1}")

    print("\n[Decision ratio]")
    print(_ratio(df["decision"]))

    slip_all   = float(df["expected_slippage_bp"].mean())
    slip_allow = float(df.loc[df["decision"]=="ALLOW","expected_slippage_bp"].mean())
    rv_all     = float(df["rv_3s_bp"].mean())
    rv_allow   = float(df.loc[df["decision"]=="ALLOW","rv_3s_bp"].mean())

    print("\n[Expected slippage bp] (lower is better)")
    print(f"overall: {slip_all:.3f} | ALLOW-only: {slip_allow:.3f} | Δ: {slip_all-slip_allow:+.3f}")
    print("[Volatility 3s bp]     (context)")
    print(f"overall: {rv_all:.3f} | ALLOW-only: {rv_allow:.3f}")

    if "label" in df.columns:
        print("\n[Label ratio overall]")
        print(_ratio(df["label"]))
        print("[Label ratio in ALLOW]")
        print(_ratio(df.loc[df['decision']=='ALLOW', 'label']))

    if "future_return" in df.columns:
        fr_all   = float(df["future_return"].mean())
        fr_allow = float(df.loc[df["decision"]=="ALLOW","future_return"].mean())
        print("\n[Future return mean]")
        print(f"overall: {fr_all:.6f} | ALLOW-only: {fr_allow:.6f}")

    if args.out_csv:
        cols = ["timestamp","decision","expected_slippage_bp","rv_3s_bp"]
        if "label" in df.columns: cols.append("label")
        if "future_return" in df.columns: cols.append("future_return")
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df[cols].to_csv(args.out_csv, index=False)
        print(f"\n📄 saved: {args.out_csv}")

if __name__ == "__main__":
    main()

