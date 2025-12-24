from contextlib import asynccontextmanager
from fastapi import FastAPI
import numpy as np

from audio_utils import split_files_into_chunks, load_audio_file
from utils import load_model, softmax
from pydantic import BaseModel    

class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    triggered_at: float | None


model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model["model"] = load_model(onnx_model_path="models/wavegram_logmel.onnx")
    print("Model loaded.")
    # await some starting actions
    yield
    # cleanup
    
app = FastAPI(lifespan=lifespan)

@app.get("/predict", response_model=PredictResponse)
async def predict(audio_path: str, threshold: float = 0.5):

    session = model["model"]

    waveform = load_audio_file(audio_path)

    chunks = split_files_into_chunks(
        waveform,
        sample_rate=32000,
        length=5.,
        overlap=0.1
    )

    if len(chunks) == 0:
        return PredictResponse(
            predicted_class=0,
            confidence=0.0,
            triggered_at=None
        )
    for chunk in chunks:
        chunk = chunk[None, :]
        logits = session.run(None, {"waveform": chunk})[0]
        probs = softmax(logits)
        artifact_probs = probs[:, 1]
        max_idx = int(np.argmax(artifact_probs))
        max_conf = float(artifact_probs[max_idx])
        if max_conf >= threshold:
            return PredictResponse(
                predicted_class=1,
                confidence=max_conf,
            )

    return PredictResponse(
        predicted_class=0,
        confidence=float(np.max(probs[:, 0])),
    )
