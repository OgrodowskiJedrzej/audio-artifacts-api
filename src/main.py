from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

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
    if not hasattr(app.state, "model"):
        return {"status": "unhealthy"}
    return {"status": "ok"}
