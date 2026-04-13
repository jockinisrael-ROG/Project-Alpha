"""FastAPI backend for a low-latency realtime AI voice assistant."""

from __future__ import annotations

import asyncio
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from time import time
from typing import Dict, Literal, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import soundfile as sf

from process.llm import get_backend_status, get_response
from process.logger import setup_logger
from process.stt import record_and_transcribe
from process.tts import _synthesize, warmup_tts
from utils.config_loader import load_config

EmotionType = Literal["happy", "sad", "angry", "surprised", "neutral"]


class VoiceRequest(BaseModel):
    text: Optional[str] = None

logger = setup_logger(__name__)
CONFIG = load_config()
EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix="assistant-bg")
ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Riko Realtime Assistant API",
    description="Low-latency backend for realtime voice assistant",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


def detect_emotion(text: str) -> EmotionType:
    """Keyword-based emotion classifier with deterministic labels."""
    if not text:
        return "neutral"

    t = text.lower()
    surprised_words = {
        "wow",
        "whoa",
        "omg",
        "unexpected",
        "no way",
        "seriously",
        "really",
        "surprised",
    }
    angry_words = {
        "angry",
        "mad",
        "furious",
        "hate",
        "annoyed",
        "stupid",
        "idiot",
        "damn",
        "wtf",
    }
    sad_words = {
        "sad",
        "depressed",
        "upset",
        "cry",
        "lonely",
        "hurt",
        "tired",
        "hopeless",
        "sorry",
    }
    happy_words = {
        "happy",
        "great",
        "awesome",
        "love",
        "excited",
        "nice",
        "cool",
        "fun",
        "yay",
        "thanks",
    }

    if any(word in t for word in surprised_words):
        return "surprised"
    if any(word in t for word in angry_words):
        return "angry"
    if any(word in t for word in sad_words):
        return "sad"
    if any(word in t for word in happy_words):
        return "happy"
    return "neutral"


def _run_background(fn, *args) -> None:
    EXECUTOR.submit(fn, *args)


def _cleanup_old_audio(max_files: int = 30) -> None:
    """Keep only newest generated voice files to avoid unbounded disk growth."""
    try:
        files = sorted(AUDIO_DIR.glob("voice_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[max_files:]:
            with suppress(Exception):
                stale.unlink()
    except Exception:
        # Cleanup is best-effort and should never break request handling.
        pass


def _synthesize_wav(reply: str, output_name: str) -> Optional[Path]:
    """Generate a WAV file from reply text and return the generated file path."""
    output_path = AUDIO_DIR / output_name
    try:
        temp_path = AUDIO_DIR / f"{output_name}.tmp.wav"
        logger.info("TTS start")
        tts_cfg = CONFIG.get("tts", {})
        voice = tts_cfg.get("voice", "en-US-AnaNeural")
        asyncio.run(_synthesize(reply, temp_path, voice, "+0%", "+0Hz"))
        # Atomic swap avoids partial writes while serving static files.
        os.replace(temp_path, output_path)
        logger.info("TTS done")
        _cleanup_old_audio()
        return output_path
    except Exception as exc:
        logger.error("TTS generation error: %s", exc, exc_info=True)
        return None


def voice_pipeline(text_input: Optional[str] = None, base_url: str = "http://127.0.0.1:8000/") -> Dict[str, object]:
    """STT -> LLM -> emotion detection -> TTS -> JSON."""
    user_text = ""

    if text_input and text_input.strip():
        user_text = text_input.strip()
    else:
        logger.info("mic trigger")
        logger.info("STT start")
        try:
            user_text = record_and_transcribe().strip()
        except Exception as exc:
            logger.warning("STT failed, returning empty result: %s", exc, exc_info=True)

    if not user_text:
        user_text = ""

    if not user_text:
        reply = "I didn't catch that. Please try again."
        emotion = "neutral"
        logger.info("LLM start")
        audio_name = f"voice_{uuid4().hex}.wav"
        audio_file = _synthesize_wav(reply, audio_name)
        if not audio_file:
            raise RuntimeError("Failed to generate audio file")

        return {
            "text": reply,
            "emotion": emotion,
            "audio_url": f"{base_url}audio/{audio_name}?v={int(time() * 1000)}",
        }

    logger.info("LLM start")
    reply = get_response(user_text)

    emotion = detect_emotion(reply)

    audio_name = f"voice_{uuid4().hex}.wav"
    audio_file = _synthesize_wav(reply, audio_name)
    if not audio_file:
        raise RuntimeError("Failed to generate audio file")

    return {
        "text": reply,
        "emotion": emotion,
        "audio_url": f"{base_url}audio/{audio_name}?v={int(time() * 1000)}",
    }


@app.api_route("/voice", methods=["GET", "POST"], tags=["Assistant"])
async def voice(request: Request, payload: Optional[VoiceRequest] = None) -> Dict[str, object]:
    """Realtime voice endpoint with optional text fallback."""
    try:
        text_input = payload.text if payload else None
        base_url = str(request.base_url)
        return await asyncio.to_thread(voice_pipeline, text_input, base_url)
    except Exception as exc:
        logger.error("Voice endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Voice pipeline failed")


@app.get("/status", tags=["Health"])
async def status() -> Dict[str, str]:
    return {"backend": get_backend_status(), "status": "ok"}


def initialize_assistant() -> None:
    try:
        tts_cfg = CONFIG.get("tts", {})
        if bool(tts_cfg.get("warmup", True)):
            warmup_tts()
        logger.info("Assistant initialized - LLM backend: %s", get_backend_status())
    except Exception as exc:
        logger.error("Initialization error: %s", exc, exc_info=True)


@app.on_event("startup")
async def startup_event() -> None:
    initialize_assistant()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    with suppress(Exception):
        EXECUTOR.shutdown(wait=False, cancel_futures=True)


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred_port: int, max_tries: int = 30) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(f"No free port found in range {preferred_port}-{preferred_port + max_tries - 1}")


if __name__ == "__main__":
    host = (os.getenv("SERVER_HOST", "0.0.0.0") or "0.0.0.0").strip()
    try:
        preferred_port = int((os.getenv("SERVER_PORT", "8000") or "8000").strip())
    except ValueError:
        preferred_port = 8000

    port = _pick_port(host, preferred_port)
    if port != preferred_port:
        logger.warning("Port %s is busy, using %s instead", preferred_port, port)

    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    logger.info("Server starting at http://%s:%s", connect_host, port)
    uvicorn.run(app, host=host, port=port)
