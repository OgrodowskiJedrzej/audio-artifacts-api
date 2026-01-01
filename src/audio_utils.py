import io

import numpy as np
import librosa
from fastapi import HTTPException

MAX_DURATION: float = 30.0


def normalize_input(waveform):
    input = waveform.astype(np.float32)
    if np.max(np.abs(input)) > 0:
        input = input / np.max(np.abs(input))
    return input


async def load_audio_file(file, sample_rate: int = 32000):
    """Load a waveform file, resample to `sample_rate`, convert to mono and normalize.

    Parameters:
        file: An async file-like object coroutine (e.g., FastAPI UploadFile). Uploaded file should be waveform.
        sample_rate: Target sample rate in Hz. Defaults to 32000.

    Returns:
        1-D float32 ndarray containing the normalized waveform sampled at `sample_rate`.
    """
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        wavefile, _ = librosa.load(io.BytesIO(file_bytes), sr=sample_rate, mono=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Audio decoding failed, check file format.") from e

    if (duration := len(wavefile) / sample_rate) > MAX_DURATION:
        raise HTTPException(status_code=413, detail=f"Audio too long ({duration:.2f}s > {MAX_DURATION}s).")

    return normalize_input(wavefile)


def split_files_into_chunks(waveform: np.ndarray, sample_rate: int, length: float, overlap: float) -> list[np.ndarray]:
    """Split a 1D audio waveform into fixed-length overlapping chunks.

    Model expects fixed-size inputs and long recordings must be processed incrementally.

    Parameters:
        waveform: One-dimensional NumPy array containing the audio waveform.
        sample_rate: Sampling rate of the audio signal.
        length: Desired chunk length in seconds.
        overlap: ractional overlap between consecutive chunks in the range [0.0, 1.0).
    Returns:
        List of 1D NumPy arrays, each of shape [length * sample_rate], representing audio chunks.
    """
    segment_desired_length = int(length * sample_rate)
    hop_len = int(segment_desired_length * (1 - overlap))
    if hop_len <= 0:
        hop_len = segment_desired_length

    chunks = []
    if waveform.shape[0] < segment_desired_length:
        return [
            np.pad(
                waveform,
                (0, segment_desired_length - waveform.shape[0]),
            )
        ]
    for start in range(0, max(0, waveform.shape[0] - segment_desired_length + 1), hop_len):
        chunk = waveform[start : start + segment_desired_length]
        if chunk.shape[0] < segment_desired_length:
            chunk = np.pad(chunk, (0, segment_desired_length - chunk.shape[0]))
        chunks.append(chunk)

    return chunks
