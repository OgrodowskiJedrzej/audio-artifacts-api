import io
import numpy as np
import pytest

from src.audio_utils import load_audio_file, split_files_into_chunks


@pytest.mark.asyncio
async def test_load_audio_file(monkeypatch):
    """Audio file is read, passed to librosa, and waveform is returned."""

    fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    class DummyFile:
        async def read(self):
            return b"fake wav bytes"

    def mock_librosa_load(file_like, sr, mono):
        assert isinstance(file_like, io.BytesIO)
        assert sr == 32000
        assert mono is True
        return fake_audio, sr

    monkeypatch.setattr(
        "src.audio_utils.librosa.load",
        mock_librosa_load,
    )

    waveform = await load_audio_file(DummyFile())

    assert isinstance(waveform, np.ndarray)


def test_split_exact_length():
    waveform = np.arange(32000)
    chunks = split_files_into_chunks(
        waveform,
        sample_rate=32000,
        length=1.0,
        overlap=0.0,
    )

    assert len(chunks) == 1
    assert chunks[0].shape == (32000,)
    np.testing.assert_array_equal(chunks[0], waveform)


def test_split_short_waveform_padding():
    waveform = np.arange(1000)
    chunks = split_files_into_chunks(
        waveform,
        sample_rate=1000,
        length=2.0,
        overlap=0.0,
    )

    assert len(chunks) == 1
    assert chunks[0].shape == (2000,)
    np.testing.assert_array_equal(chunks[0][:1000], waveform)
    assert np.all(chunks[0][1000:] == 0)


def test_split_with_overlap():
    waveform = np.arange(10000)
    chunks = split_files_into_chunks(
        waveform,
        sample_rate=1000,
        length=2.0,
        overlap=0.5,
    )

    # chunk = 2000, hop = 1000
    assert len(chunks) > 5
    assert chunks[0].shape == (2000,)
    assert np.array_equal(chunks[0][1000:], chunks[1][:1000])


def test_split_overlap_edge_case():
    waveform = np.arange(4000)

    chunks = split_files_into_chunks(
        waveform,
        sample_rate=1000,
        length=2.0,
        overlap=1.0,  # forces hop_len <= 0
    )

    # hop_len should fall back to segment length
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.shape == (2000,)
