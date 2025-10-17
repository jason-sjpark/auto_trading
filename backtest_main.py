import pandas as pd

df = pd.read_parquet("data/features/features_1s.parquet")
print("✅ 전체 shape:", df.shape)
print("컬럼 목록:", df.columns.tolist()[:10])
print(df.head())
