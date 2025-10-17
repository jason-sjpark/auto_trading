import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    간단·안정 LSTM 분류기
    - input: (B, T, F)
    - output: (B, num_classes)
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, num_classes: int = 3, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(d * hidden_size),
            nn.Linear(d * hidden_size, d * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        _, (hn, _) = self.lstm(x)          # hn: (num_layers*d, B, H)
        last = torch.cat([hn[-1], hn[-2]], dim=-1) if self.lstm.bidirectional else hn[-1]  # (B, H*dirs)
        return self.head(last)
