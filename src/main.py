from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from src.utils import load_model
from src.api.v1.router import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(onnx_model_path="models/wavegram_logmel.onnx")
    logger.debug("Model loaded.")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok"}
