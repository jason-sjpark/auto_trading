# scripts/peek_parquet.py
import pandas as pd, os, sys
p = sys.argv[1] if len(sys.argv) > 1 else "data/features/features_1s.parquet"
df = pd.read_parquet(p)
print("path:", os.path.abspath(p))
print("rows, cols:", df.shape)
cols05 = [c for c in df.columns if "@0.5s" in c]
print("0.5s cols:", cols05)
print("non-null per col:")
print(df[cols05].notna().sum().sort_values())
print(df[["timestamp"] + cols05].head(10))
