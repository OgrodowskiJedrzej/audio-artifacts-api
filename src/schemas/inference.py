from pydantic import BaseModel
from fastapi import UploadFile


class InferenceInputSchema(BaseModel):
    input_url: UploadFile


class InferenceOutputSchema(BaseModel):
    filename: str
    predicted_class: int
