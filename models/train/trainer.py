import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

from models.train.dataset import FeatureSequenceDataset
from models.train.model_lstm import LSTMClassifier

# =============================
# 기본 설정 (필요시 CLI로 덮어쓰기)
# =============================
DEFAULT_CFG = dict(
    features_path="data/features/features_1s.parquet",  # BatchFeaturePipeline 결과 또는 병합본
    labels_source="data/processed/ohlcv_1s_*.parquet",  # (참고용) 안 써도 됨
    seq_len=30,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=30,
    patience=5,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    num_workers=0,
    valid_ratio=0.2,
    out_dir="models/checkpoints",
    onnx_path="models/outputs/scalping_lstm.onnx",
    scaler_path="models/outputs/feature_scaler.pkl",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def load_features_df(path: str) -> pd.DataFrame:
    # path는 단일 parquet/csv 가정. 필요시 glob 지원으로 확장 가능
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # timestamp 정리
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def split_and_scale(df: pd.DataFrame, valid_ratio: float, scaler_path: str):
    # feature 컬럼 자동 탐지
    exclude = {"timestamp", "label", "future_return"}
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    # 시간순 분할(누수 방지)
    n = len(df)
    n_valid = int(n * valid_ratio)
    train_df = df.iloc[:n - n_valid].copy()
    valid_df = df.iloc[n - n_valid:].copy()

    # 스케일러는 train에만 fit
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    valid_df[feat_cols] = scaler.transform(valid_df[feat_cols])

    # 저장
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(dict(scaler=scaler, feature_cols=feat_cols), scaler_path)

    return train_df, valid_df, feat_cols

def make_dataloaders(train_df, valid_df, seq_len, batch_size, num_workers, feature_cols):
    train_ds = FeatureSequenceDataset(train_df, seq_len=seq_len, feature_cols=feature_cols)
    valid_ds = FeatureSequenceDataset(valid_df, seq_len=seq_len, feature_cols=feature_cols)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, valid_loader

def compute_weights(train_df):
    y = train_df["label"].values
    classes = np.array([-1, 0, 1])
    mapped = pd.Series(y).map({-1:0, 0:1, 1:2}).values
    w = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=mapped)
    return torch.tensor(w, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        # 라벨을 { -1,0,1 } → {0,1,2}
        yb_map = torch.where(yb == -1, torch.zeros_like(yb), torch.where(yb == 0, torch.ones_like(yb), torch.full_like(yb, 2)))

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb_map)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_acc += (preds == yb_map).sum().item()
        total += xb.size(0)
    return total_loss / total, total_acc / total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yb_map = torch.where(yb == -1, torch.zeros_like(yb), torch.where(yb == 0, torch.ones_like(yb), torch.full_like(yb, 2)))
        logits = model(xb)
        loss = criterion(logits, yb_map)
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_acc += (preds == yb_map).sum().item()
        total += xb.size(0)
    return total_loss / total, total_acc / total

def export_onnx(model, input_size, seq_len, onnx_path, device):
    model.eval()
    dummy = torch.randn(1, seq_len, input_size, device=device)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17
    )

def main(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 1) 데이터 로드
    df = load_features_df(cfg["features_path"])
    if "label" not in df.columns:
        raise ValueError("features_path에는 'label' 컬럼이 포함되어 있어야 합니다. (scalping_labeler 병합 필요)")

    # 2) split & scale
    train_df, valid_df, feat_cols = split_and_scale(df, cfg["valid_ratio"], cfg["scaler_path"])

    # 3) dataloaders
    train_loader, valid_loader = make_dataloaders(
        train_df, valid_df, cfg["seq_len"], cfg["batch_size"], cfg["num_workers"], feat_cols
    )

    # 4) 모델/손실/옵티마이저
    device = cfg["device"]
    model = LSTMClassifier(input_size=len(feat_cols),
                           hidden_size=cfg["hidden_size"],
                           num_layers=cfg["num_layers"],
                           dropout=cfg["dropout"],
                           num_classes=3,
                           bidirectional=True).to(device)

    class_weights = compute_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # 5) 학습 루프 + 조기종료
    best_val = float("inf")
    best_path = os.path.join(cfg["out_dir"], "best_scalping_lstm.pt")
    patience, no_improve = cfg["patience"], 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, valid_loader, criterion, device)
        print(f"[{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | valid loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss + (1.0 - va_acc) < best_val:  # 손실↓, 정확도↑ 복합 기준(간단 가중)
            best_val = va_loss + (1.0 - va_acc)
            torch.save({"model": model.state_dict(), "cfg": cfg, "features": feat_cols}, best_path)
            no_improve = 0
            print(f"  ↳ saved: {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  ↳ early stopping triggered.")
                break

    # 6) ONNX export (best ckpt 로드 후)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    export_onnx(model, input_size=len(feat_cols), seq_len=cfg["seq_len"], onnx_path=cfg["onnx_path"], device=device)
    print(f"ONNX exported → {cfg['onnx_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in DEFAULT_CFG.items():
        argtype = type(v) if v is not None else str
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        else:
            parser.add_argument(f"--{k}", type=argtype, default=v)
    args = parser.parse_args()
    main(vars(args))
