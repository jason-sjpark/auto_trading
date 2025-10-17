import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureSequenceDataset(Dataset):
    """
    features_df + labels_df를 timestamp 근접 조인(이미 합쳐진 DataFrame도 OK)해
    시계열 윈도우 텐서 (B, T, F)와 라벨 y를 반환.
    - required columns:
      * timestamp, label
      * feature columns: everything except [timestamp, label, future_return]
    """
    def __init__(self, merged_df: pd.DataFrame, seq_len: int = 30, feature_cols=None):
        df = merged_df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # feature columns 자동 탐지
        if feature_cols is None:
            exclude = {"timestamp", "label", "future_return"}
            feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        self.feature_cols = feature_cols

        # 넘파이 캐싱
        self.X = df[self.feature_cols].values.astype(np.float32)
        self.y = df["label"].values.astype(np.int64)
        self.seq_len = seq_len

        # 유효한 마지막 인덱스
        self.max_start = len(self.X) - self.seq_len
        if self.max_start <= 0:
            raise ValueError(f"Not enough rows ({len(self.X)}) for seq_len={self.seq_len}")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]          # (T, F)
        y = self.y[idx + self.seq_len - 1]           # 윈도우 마지막 시점의 라벨
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
