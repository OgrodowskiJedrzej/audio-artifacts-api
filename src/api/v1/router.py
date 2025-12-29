from fastapi import APIRouter
from src.api.v1.inference import router as inference_router

router = APIRouter()

router.include_router(inference_router, prefix="/inference", tags=["Inference"])
