"""Configuration loader."""

from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "assistant": {
        "name": "Alpha",
        "system_prompt": "You are Alpha, a cute anime tsundere AI assistant. Playful, teasing, respectful, and caring.",
    },
    "stt": {
        "vad_mode": True,
        "sample_rate": 16000,
        "recording_seconds": 4,
        "max_recording_seconds": 20,
        "speech_start_timeout_seconds": 300,
        "min_speech_seconds": 0.2,
        "recording_file": "audio/recording.wav",
        "whisper_model": "base",
        "language": "en",
        "device": "cpu",
        "compute_type": "int8",
        "silence_threshold": 0.004,
        "threshold_multiplier": 1.9,
        "chunk_seconds": 0.1,
        "max_silence_seconds": 0.8,
        "input_device": None,
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
    "vision": {
        "enabled": True,
        "camera_index": 0,
        "sample_seconds": 0.15,
        "turn_interval": 1,
        "auto_calibration": True,
        "calibration_refresh_minutes": 120,
        "calibration_sample_seconds": 1.8,
        "enhanced_models": False,
    },
}


def _merge_dicts(base, updates):
    """Recursively merge updates into base."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Load config from config/config.yaml or return defaults."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        
        if not isinstance(loaded, dict):
            logger.error(f"Invalid YAML in {config_path}")
            return DEFAULT_CONFIG
            
        logger.info(f"Loaded config from {config_path}")
        return _merge_dicts(DEFAULT_CONFIG, loaded)
        
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return DEFAULT_CONFIG
