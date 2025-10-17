from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib
import torch
import uvicorn
from typing import List, Dict

# ==============================================================
# ✅ 환경 설정
# ==============================================================

MODEL_PATH = "models/outputs/scalping_lstm.onnx"
SCALER_PATH = "models/outputs/feature_scaler.pkl"
SEQ_LEN = 30

app = FastAPI(title="Scalping LSTM Inference API", version="1.0")

# ==============================================================
# 🧠 모델 & 스케일러 로드
# ==============================================================

print("🚀 Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

print("🔧 Loading feature scaler...")
scaler_bundle = joblib.load(SCALER_PATH)
scaler = scaler_bundle["scaler"]
feature_cols = scaler_bundle["feature_cols"]

# ==============================================================
# 🧩 입력 스키마
# ==============================================================

class FeatureWindow(BaseModel):
    """
    seq_len(기본 30)개의 시점이 포함된 피처 윈도우 입력
    """
    sequence: List[Dict[str, float]]


# ==============================================================
# ⚙️ 헬퍼 함수
# ==============================================================

def prepare_input(sequence: List[Dict[str, float]]):
    """
    JSON 시퀀스를 (1, seq_len, num_features)로 변환 + 스케일링
    """
    # 피처 정렬 일관성 유지
    seq_array = np.array([[frame.get(f, 0.0) for f in feature_cols] for frame in sequence], dtype=np.float32)
    if seq_array.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - seq_array.shape[0]
        pad = np.zeros((pad_len, seq_array.shape[1]), dtype=np.float32)
        seq_array = np.vstack([pad, seq_array])
    elif seq_array.shape[0] > SEQ_LEN:
        seq_array = seq_array[-SEQ_LEN:]

    # 스케일링
    seq_scaled = scaler.transform(seq_array)
    return seq_scaled[np.newaxis, :, :]  # (1, seq_len, F)


# ==============================================================
# 🔮 추론 엔드포인트
# ==============================================================

@app.post("/predict")
def predict(features: FeatureWindow):
    """
    입력: JSON { sequence: [ {feature1: val, feature2: val, ...}, ... ] }
    출력: 상승/하락/횡보 확률
    """
    try:
        x = prepare_input(features.sequence)
        ort_inputs = {"input": x}
        logits = session.run(None, ort_inputs)[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().flatten()

        # 순서: [-1, 0, 1] (하락, 횡보, 상승)
        result = {
            "class_probs": {
                "down": float(probs[0]),
                "sideways": float(probs[1]),
                "up": float(probs[2])
            },
            "predicted_label": int(np.argmax(probs) - 1)  # 0→-1, 1→0, 2→1
        }
        return result

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Scalping Model Inference API is running 🚀"}


# ==============================================================
# 🏁 서버 실행
# ==============================================================

if __name__ == "__main__":
    uvicorn.run("models.serve.app:app", host="0.0.0.0", port=8000, reload=True)
