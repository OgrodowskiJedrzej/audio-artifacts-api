from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import numpy as np
import logging
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from audio_utils import split_files_into_chunks, load_audio_file
from utils import load_model

import logging

logger = logging.getLogger(__name__)

model = {}

class Input(BaseModel):
    input_url: UploadFile

class Output(BaseModel):
    filename: str
    predicted_class: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    model["model"] = load_model(onnx_model_path="models/wavegram_logmel.onnx")
    logger.debug("Model loaded.")
    yield
    
app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status" : "ok"}

def predict(input, session) -> int:
    chunks = split_files_into_chunks(
        input,
        sample_rate=32000,
        length=5.,
        overlap=0.1
    )
    for chunk in chunks:
            chunk = chunk[None, :]
            logits = session.run(None, {"waveform": chunk})[0]
            predicted_class = int(np.argmax(logits, axis=1)[0])
            if predicted_class == 1:
                return 1
    return 0


@app.post("/inference/file", response_model=Output)
async def inference_file(file: UploadFile = File(...)):
    session = model["model"]
    waveform = await load_audio_file(file)
    prediction = predict(waveform, session)
    return Output(
        filename=file.filename,
        predicted_class=prediction
    )
