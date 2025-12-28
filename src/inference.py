import numpy as np

from audio_utils import split_files_into_chunks


def predict(input: np.ndarray, model) -> int:
    """Run prediction on waveform file.

    Args:
        input: Loaded waveform.
        session: Loaded prediction model.
    Returns:
        Predicted class {0,1} no_artifact/ artifact.
    """
    chunks = split_files_into_chunks(
        input,
        sample_rate=32000,
        length=5.0,
        overlap=0.1,
    )
    for chunk in chunks:
        chunk = chunk[None, :]
        logits = model.run(None, {"waveform": chunk})[0]
        predicted_class = int(np.argmax(logits, axis=1)[0])
        if predicted_class == 1:
            return 1
    return 0
