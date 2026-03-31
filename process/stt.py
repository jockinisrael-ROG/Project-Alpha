from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from utils.config_loader import load_config

_MODEL_CACHE: dict[Tuple[str, str, str], WhisperModel] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    cache_key = (model_size, device, compute_type)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL_CACHE[cache_key]


def _record_audio(
    sample_rate: int,
    max_seconds: float,
    silence_threshold: float,
    chunk_seconds: float,
    max_silence_seconds: float,
) -> np.ndarray:
    chunk_frames = max(1, int(sample_rate * chunk_seconds))
    chunks: List[np.ndarray] = []

    print("[STT] Listening... Speak now.")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        spoken = False
        silence_chunks = 0
        max_silence_chunks = max(1, int(max_silence_seconds / chunk_seconds))
        start_time = time.monotonic()

        while True:
            audio_chunk, _ = stream.read(chunk_frames)
            mono = audio_chunk[:, 0].copy()
            chunks.append(mono)

            energy = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
            if energy > silence_threshold:
                spoken = True
                silence_chunks = 0
            elif spoken:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    break

            elapsed = time.monotonic() - start_time
            if elapsed >= max_seconds:
                break

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32)


def record_and_transcribe() -> str:
    """Record microphone input and transcribe with faster-whisper."""
    config = load_config()
    stt_cfg = config["stt"]

    sample_rate = int(stt_cfg.get("sample_rate", 16000))
    max_seconds = float(stt_cfg.get("max_recording_seconds", stt_cfg.get("recording_seconds", 20)))
    recording_file = Path(stt_cfg.get("recording_file", "audio/recording.wav"))
    whisper_model_size = str(stt_cfg.get("whisper_model", "base"))
    language = stt_cfg.get("language", "en")
    device = str(stt_cfg.get("device", "cpu")).lower()
    compute_type = str(stt_cfg.get("compute_type", "int8"))
    silence_threshold = float(stt_cfg.get("silence_threshold", 0.01))
    chunk_seconds = float(stt_cfg.get("chunk_seconds", 0.1))
    max_silence_seconds = float(stt_cfg.get("max_silence_seconds", 0.45))

    recording_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio = _record_audio(
            sample_rate,
            max_seconds,
            silence_threshold,
            chunk_seconds,
            max_silence_seconds,
        )
        if audio.size == 0:
            return ""

        sf.write(recording_file, audio, sample_rate)

        model = _get_model(whisper_model_size, device, compute_type)
        segments, _info = model.transcribe(
            str(recording_file),
            language=language,
            vad_filter=True,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    except Exception as exc:
        print(f"[STT] Transcription error: {exc}")
        return ""
