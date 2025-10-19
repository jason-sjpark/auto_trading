import pandas as pd
df = pd.read_parquet("data/features/features_1s_2025-10-17_risk.parquet")
print(df[["timestamp","relative_spread_bp","expected_slippage_bp","rv_3s_bp","risk_score","decision","size_scale","order_type_hint"]].head(5))
print("decision ratio:", df["decision"].value_counts(normalize=True).round(3))
