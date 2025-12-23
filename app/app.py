from contextlib import asynccontextmanager
from fastapi import FastAPI
import onnxruntime as ort
import onnx
import librosa
import numpy as np

from audio_utils import split_files_into_chunks
from utils import load_model, softmax

model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model["model"] = load_model()
    # await some starting actions
    yield
    # cleanup

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(wavefile: np.ndarray):
    for chunk in split_files_into_chunks(wavefile, 32000, 5, 0.25):
        y_pred = model["model"].run(None, {"waveform": chunk})[0]
        predicted_class = np.argmax(softmax(y_pred), axis=1)[0]
        if predicted_class == 1: return predicted_class
    return predicted_class  