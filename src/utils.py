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


def load_model(onnx_model_path: str) -> ort.InferenceSession | None:
    try:
        onnx.checker.check_model(onnx_model_path, full_check=True)
    except Exception as e:
        logger.error("Model check failed: %s", e)
        raise
    model = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    return model


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def normalize_input(waveform):
    input = waveform.astype(np.float32)
    if np.max(np.abs(input)) > 0:
        input = input / np.max(np.abs(input))
    return input
