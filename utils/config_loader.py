from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "assistant": {
        "name": "Alpha",
        "system_prompt": "You are Alpha, a cute anime tsundere AI assistant. Playful, teasing, slightly rude but caring.",
    },
    "stt": {
        "vad_mode": True,
        "sample_rate": 16000,
        "recording_seconds": 4,
        "max_recording_seconds": 20,
        "recording_file": "audio/recording.wav",
        "whisper_model": "base",
        "language": "en",
        "device": "cpu",
        "compute_type": "int8",
        "silence_threshold": 0.008,
        "chunk_seconds": 0.1,
        "max_silence_seconds": 0.45,
    },
    "tts": {
        "voice": "en-US-AnaNeural",
        "output_file": "audio/output.wav",
        "in_memory_mode": True,
        "async_queue": True,
        "warmup": True,
    },
    "llm": {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "streaming": True,
        "stream_tts_chunk_chars": 24,
        "stream_tts_min_tail_chars": 6,
        "stream_tts_max_delay_sec": 0.25,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_model": "openai/gpt-4o-mini",
        "openrouter_api_key": "",
        "site_url": "http://localhost",
        "app_name": "Alpha Local Assistant",
        "temperature": 0.7,
        "max_history_messages": 12,
    },
    "memory": {
        "db_path": "memory/memory.db",
    },
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    return _deep_merge(DEFAULT_CONFIG, loaded)
