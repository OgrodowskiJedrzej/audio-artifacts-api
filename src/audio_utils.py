import numpy as np
import librosa
import io

async def load_audio_file(file, sample_rate: int = 32000):
    file_bytes = await file.read()
    wavefile, _ = librosa.load(io.BytesIO(
        file_bytes), sr=sample_rate, mono=True)
    return wavefile


def split_files_into_chunks(waveform: np.ndarray, sample_rate: int, length: float, overlap: float) -> list[np.ndarray]:
    segment_desired_length = int(length * sample_rate)
    hop_len = int(segment_desired_length * (1 - overlap))
    if hop_len <= 0:
        hop_len = segment_desired_length

    chunks = []
    for start in range(0, max(0, waveform.shape[0] - segment_desired_length + 1), hop_len):
        chunk = waveform[start: start + segment_desired_length]
        if chunk.shape[0] < segment_desired_length:
            chunk = np.pad(chunk, (0, segment_desired_length - chunk.shape[0]))
        chunks.append(chunk)

    return chunks
