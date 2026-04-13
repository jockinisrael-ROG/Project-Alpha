"""Speech-To-Text module using faster-whisper with voice activity detection."""

from __future__ import annotations

from pathlib import Path
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from utils.config_loader import load_config
from process.logger import setup_logger

logger = setup_logger(__name__)

_MODEL_CACHE: dict[Tuple[str, str, str], WhisperModel] = {}
_INPUT_DEVICE_CACHE: dict[object, Optional[int]] = {}
_DEVICE_LIST_LOGGED = False


def _log_input_devices() -> None:
    """Print all available input-capable audio devices for debugging."""
    global _DEVICE_LIST_LOGGED
    if _DEVICE_LIST_LOGGED:
        return

    try:
        devices = sd.query_devices()
        host_api = sd.query_hostapis()
        print("[STT] Available audio input devices:")
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            host_name = "unknown"
            host_api_index = device.get("hostapi")
            if host_api_index is not None and 0 <= int(host_api_index) < len(host_api):
                host_name = str(host_api[int(host_api_index)].get("name", "unknown"))
            print(
                f"[STT]   #{index}: {device.get('name', 'unknown')} | "
                f"inputs={device.get('max_input_channels', 0)} | host={host_name}"
            )
        _DEVICE_LIST_LOGGED = True
    except Exception as exc:
        print(f"[STT] Failed to enumerate devices: {exc}")


def _resolve_input_device(input_device_raw: object) -> Optional[int]:
    """Resolve a working input device or fall back to the default system input."""
    if input_device_raw in _INPUT_DEVICE_CACHE:
        return _INPUT_DEVICE_CACHE[input_device_raw]

    try:
        if input_device_raw not in (None, "", "default"):
            resolved_device = int(input_device_raw)
            _INPUT_DEVICE_CACHE[input_device_raw] = resolved_device
            return resolved_device

        default_device = sd.default.device
        if isinstance(default_device, (tuple, list)) and default_device:
            default_input = default_device[0]
            if default_input is not None and int(default_input) >= 0:
                resolved_device = int(default_input)
                _INPUT_DEVICE_CACHE[input_device_raw] = resolved_device
                return resolved_device

        default_input_info = sd.query_devices(kind="input")
        default_input_index = default_input_info.get("index")
        if default_input_index is not None:
            resolved_device = int(default_input_index)
            _INPUT_DEVICE_CACHE[input_device_raw] = resolved_device
            return resolved_device
    except Exception:
        pass

    try:
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) > 0:
                _INPUT_DEVICE_CACHE[input_device_raw] = index
                return index
    except Exception:
        pass

    _INPUT_DEVICE_CACHE[input_device_raw] = None
    return None


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
    input_device: int | None,
) -> np.ndarray:
    chunk_frames = max(1, int(sample_rate * chunk_seconds))
    chunks: List[np.ndarray] = []
    pre_speech_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(0.35 / chunk_seconds)))

    print("[STT] Recording started")
    print(f"[STT] sample_rate={sample_rate}, chunk_seconds={chunk_seconds}, max_seconds={max_seconds}")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=input_device,
        blocksize=chunk_frames,
    ) as stream:
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

            if len(chunks) == 0 and not spoken:
                rms_preview = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
                print(f"[STT] Raw audio preview RMS: {rms_preview:.6f}")

            energy = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
            dynamic_threshold = max(silence_threshold, noise_floor * max(threshold_multiplier, 1.0))
            trigger_threshold = min(dynamic_threshold, max(noise_floor * 1.6, silence_threshold * 0.55))

            if not spoken:
                # Keep an adaptive baseline before speech starts.
                noise_floor = (noise_floor * 0.95) + (energy * 0.05)
                pre_speech_buffer.append(mono)

                if energy > trigger_threshold:
                    print(f"[STT] Speech detected, energy={energy:.6f}, threshold={trigger_threshold:.6f}")
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
                        print("[STT] Speech start timeout reached")
                        return np.array([], dtype=np.float32)
                continue

            chunks.append(mono)
            if energy > trigger_threshold:
                speech_chunks += 1
                silence_chunks = 0
            else:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    break

            elapsed_speech = time.monotonic() - speech_start
            if elapsed_speech >= max_seconds:
                print("[STT] Max recording duration reached")
                break

    if not chunks:
        print("[STT] No chunks recorded")
        return np.array([], dtype=np.float32)

    spoken_seconds = speech_chunks * chunk_seconds
    if spoken_seconds < min_speech_seconds:
        print(f"[STT] Recorded speech too short: {spoken_seconds:.2f}s")
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks).astype(np.float32)
    print(f"[STT] Recording ended, frames={audio.shape[0]}, seconds={audio.shape[0] / sample_rate:.2f}")
    return audio


def _normalize_audio(audio: np.ndarray, target_rms: float = 0.04) -> np.ndarray:
    """Normalize input volume so quiet microphones are easier to transcribe."""
    if audio.size == 0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms <= 1e-6:
        return audio

    gain = float(np.clip(target_rms / rms, 1.0, 8.0))
    if gain <= 1.01:
        return audio

    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def _transcribe_audio_file(model: WhisperModel, recording_file: Path, language: str) -> str:
    """Transcribe a recorded WAV file and log the recognized text."""
    segments, _info = model.transcribe(
        str(recording_file),
        language=language,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        segments, _info = model.transcribe(
            str(recording_file),
            language=language,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()

    print(f"[STT] Recognized text: {text}")
    return text


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
    input_device_raw = stt_cfg.get("input_device", None)
    input_device = _resolve_input_device(input_device_raw)

    recording_file.parent.mkdir(parents=True, exist_ok=True)
    _log_input_devices()
    print(f"[STT] Selected input device: {input_device if input_device is not None else 'default'}")

    model = _get_model(whisper_model_size, device, compute_type)

    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            print(f"[STT] Recording attempt {attempt + 1}/2")
            audio = _record_audio(
                sample_rate,
                max_seconds,
                silence_threshold,
                chunk_seconds,
                max_silence_seconds,
                speech_start_timeout_seconds,
                min_speech_seconds,
                threshold_multiplier,
                input_device,
            )
            if audio.size == 0:
                print("[STT] Empty audio captured")
                continue

            audio = _normalize_audio(audio)
            print(f"[STT] Normalized audio RMS: {float(np.sqrt(np.mean(np.square(audio)) + 1e-12)):.6f}")

            sf.write(recording_file, audio, sample_rate)
            print(f"[STT] Audio written to: {recording_file}")

            text = _transcribe_audio_file(model, recording_file, language)
            if text:
                return text

            print("[STT] No text recognized on this attempt")
        except Exception as exc:
            last_error = exc
            print(f"[STT] Transcription error on attempt {attempt + 1}: {exc}")
            logger.error("STT attempt %s failed: %s", attempt + 1, exc, exc_info=True)

    if last_error is not None:
        logger.warning("STT failed after retry; using empty result")

    return ""
