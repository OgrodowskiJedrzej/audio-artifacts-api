import pytest
import onnx
import onnxruntime as ort

from src.utils import load_model


def test_load_model_success(monkeypatch):
    """Model passes check and InferenceSession is created."""

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
    """Model check fails and an exception is raised."""

    def mock_check_model(*args, **kwargs):
        raise RuntimeError("ONNX model validation failed")

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        mock_check_model,
    )

    with pytest.raises(RuntimeError, match="ONNX model validation failed"):
        load_model("invalid.onnx")


def test_load_model_inference_session_failure(monkeypatch):
    """ONNX check passes but InferenceSession creation fails."""

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

    with pytest.raises(RuntimeError):
        load_model("dummy.onnx")
