from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from schemas.inference import InferenceOutputSchema
from inference import predict
from audio_utils import load_audio_file

router = APIRouter()


@router.post("/file", response_model=InferenceOutputSchema)
async def inference_file(request: Request, file: UploadFile = File(...)):
    session = getattr(request.app.state, "model", None)
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    waveform = await load_audio_file(file)
    prediction = predict(waveform, session)
    return InferenceOutputSchema(filename=file.filename, predicted_class=prediction)
