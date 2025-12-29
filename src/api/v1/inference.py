from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from src.schemas.inference import InferenceOutputSchema
from src.inference import predict
from src.audio_utils import load_audio_file

router = APIRouter()

_file = File(...)


@router.post("/file", response_model=InferenceOutputSchema)
async def inference_file(request: Request, file: UploadFile = _file):
    session = getattr(request.app.state, "model", None)
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    waveform = await load_audio_file(file)
    prediction = predict(waveform, session)
    return InferenceOutputSchema(filename=file.filename, predicted_class=prediction)
