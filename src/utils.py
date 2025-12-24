import onnx
import onnxruntime as ort
import numpy as np

def check_model(onnx_model_path: str) -> bool:
    onnx.checker.check_model(onnx_model_path)


def load_model(onnx_model_path: str) -> None:
    model = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    return model


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
