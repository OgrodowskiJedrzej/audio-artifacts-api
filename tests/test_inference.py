import numpy as np
import pytest

from src.audio_utils import split_files_into_chunks
from src.inference import predict


class DummyModel:
    def __init__(self, outputs):
        """
        outputs: list of logits arrays to return per call
        """
        self.outputs = outputs
        self.call_idx = 0

    def run(self, _, inputs):
        output = self.outputs[self.call_idx]
        self.call_idx += 1
        return [output]


def test_predict_returns_artifact(monkeypatch):
    """Returns 'artifact' when any chunk predicts class 1."""

    # Mock chunks
    chunks = [
        np.zeros(160000),
        np.ones(160000),
    ]

    monkeypatch.setattr(
        "src.audio_utils.split_files_into_chunks",
        lambda *args, **kwargs: chunks,
    )

    # First chunk -> class 0, second chunk -> class 1
    model = DummyModel(
        outputs=[
            np.array([[0.9, 0.1]]),  # no_artifact
            np.array([[0.2, 0.8]]),  # artifact
        ]
    )

    waveform = np.random.randn(320000)
    result = predict(waveform, model)

    assert result == "artifact"


def test_predict_returns_no_artifact(monkeypatch):
    """Returns 'no_artifact' when all chunks predict class 0."""

    chunks = [
        np.zeros(160000),
        np.ones(160000),
    ]

    monkeypatch.setattr(
        "src.audio_utils.split_files_into_chunks",
        lambda *args, **kwargs: chunks,
    )

    model = DummyModel(
        outputs=[
            np.array([[0.8, 0.2]]),
            np.array([[0.7, 0.3]]),
        ]
    )

    waveform = np.random.randn(320000)
    result = predict(waveform, model)

    assert result == "no_artifact"


def test_predict_model_input_shape(monkeypatch):
    chunk_length = 5 * 32000

    chunks = [np.zeros(chunk_length)]

    monkeypatch.setattr(
        "src.audio_utils.split_files_into_chunks",
        lambda *args, **kwargs: chunks,
    )

    class ShapeCheckingModel:
        def run(self, _, inputs):
            waveform = inputs["waveform"]
            assert waveform.shape == (1, chunk_length)
            return [np.array([[1.0, 0.0]])]

    waveform = np.random.randn(chunk_length)
    result = predict(waveform, ShapeCheckingModel())

    assert result == "no_artifact"
