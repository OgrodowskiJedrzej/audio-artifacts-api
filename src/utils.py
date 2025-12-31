import os
import logging

import onnx
import onnxruntime as ort
import numpy as np


logger = logging.getLogger(__name__)


def download_model_from_azure_ml(ml_client, model_name: str, model_version: str, download_path: str = "models/") -> str:
    os.makedirs(download_path, exist_ok=True)

    model = ml_client.models.get(name=model_name, version=model_version)

    local_path = ml_client.models.download(name=model.name, version=model.version, download_path=download_path)

    return os.path.join(local_path, "wavegram_logmel.onnx")


def load_model(onnx_model_path: str) -> ort.InferenceSession:
    if not os.path.isfile(onnx_model_path):
        raise RuntimeError(f"Model file not found: {onnx_model_path}")

    try:
        onnx.checker.check_model(onnx_model_path, full_check=True)
        return ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"]
        )
    except Exception:
        logger.exception("Failed to load ONNX model")
        raise RuntimeError("Model failed to load")


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)