from pathlib import Path
import time
from collections import deque
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
    speech_start_timeout_seconds: float,
    min_speech_seconds: float,
    threshold_multiplier: float,
) -> np.ndarray:
    chunk_frames = max(1, int(sample_rate * chunk_seconds))
    chunks: List[np.ndarray] = []
    pre_speech_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(0.35 / chunk_seconds)))

    print("[STT] Listening... Speak now.")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        spoken = False
        silence_chunks = 0
        speech_chunks = 0
        max_silence_chunks = max(1, int(max_silence_seconds / chunk_seconds))
        listen_start = time.monotonic()
        speech_start = 0.0
        # Start with a conservative estimate and adapt to ambient noise.
        noise_floor = max(silence_threshold / max(threshold_multiplier, 1.0), 1e-4)

        while True:
            audio_chunk, _ = stream.read(chunk_frames)
            mono = audio_chunk[:, 0].copy()

            energy = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
            dynamic_threshold = max(silence_threshold, noise_floor * max(threshold_multiplier, 1.0))

            if not spoken:
                # Keep an adaptive baseline before speech starts.
                noise_floor = (noise_floor * 0.95) + (energy * 0.05)
                pre_speech_buffer.append(mono)

                if energy > dynamic_threshold:
                    spoken = True
                    speech_start = time.monotonic()
                    silence_chunks = 0
                    speech_chunks = 1
                    # Keep a small pre-roll so first phoneme is not clipped.
                    chunks.extend(pre_speech_buffer)
                    chunks.append(mono)
                else:
                    waiting = time.monotonic() - listen_start
                    if waiting >= speech_start_timeout_seconds:
                        return np.array([], dtype=np.float32)
                continue

            chunks.append(mono)
            if energy > dynamic_threshold:
                speech_chunks += 1
                silence_chunks = 0
            else:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    break

            elapsed_speech = time.monotonic() - speech_start
            if elapsed_speech >= max_seconds:
                break

    if not chunks:
        return np.array([], dtype=np.float32)

    spoken_seconds = speech_chunks * chunk_seconds
    if spoken_seconds < min_speech_seconds:
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
    speech_start_timeout_seconds = float(stt_cfg.get("speech_start_timeout_seconds", 300.0))
    min_speech_seconds = float(stt_cfg.get("min_speech_seconds", 0.25))
    threshold_multiplier = float(stt_cfg.get("threshold_multiplier", 2.5))

    recording_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio = _record_audio(
            sample_rate,
            max_seconds,
            silence_threshold,
            chunk_seconds,
            max_silence_seconds,
            speech_start_timeout_seconds,
            min_speech_seconds,
            threshold_multiplier,
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
