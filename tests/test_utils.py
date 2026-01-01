import os
import pytest
import onnx
import onnxruntime as ort

from src.utils import load_model


def test_load_model_success(monkeypatch):
    """Model passes check and InferenceSession is created."""

    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    # Mock onnx.checker.check_model to succeed
    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda *args, **kwargs: None,
    )

    # Mock ort.InferenceSession
    class DummySession:
        pass

    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda *args, **kwargs: DummySession(),
    )

    model = load_model("dummy.onnx")

    assert model is not None
    assert isinstance(model, DummySession)


def test_load_model_checker_failure(monkeypatch):
    """Model check fails and error is wrapped."""

    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    def mock_check_model(*args, **kwargs):
        raise RuntimeError("ONNX model validation failed")

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        mock_check_model,
    )

    with pytest.raises(RuntimeError, match="Model failed to load") as exc:
        load_model("invalid.onnx")

    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "ONNX model validation failed" in str(exc.value.__cause__)


def test_load_model_inference_session_failure(monkeypatch):
    """InferenceSession creation fails and error is wrapped."""

    monkeypatch.setattr(os.path, "isfile", lambda _: True)

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda *args, **kwargs: None,
    )

    def mock_inference_session(*args, **kwargs):
        raise RuntimeError("Failed to create session")

    monkeypatch.setattr(
        ort,
        "InferenceSession",
        mock_inference_session,
    )

    with pytest.raises(RuntimeError, match="Model failed to load") as exc:
        load_model("dummy.onnx")

    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "Failed to create session" in str(exc.value.__cause__)
