"""Text-To-Speech module using Edge TTS with async streaming and caching."""

import atexit
import asyncio
import io
import importlib
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import sounddevice as sd
import soundfile as sf

from utils.config_loader import load_config
from process.events import dispatch_event
from process.logger import setup_logger

logger = setup_logger(__name__)


_TTS_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=4)
_TTS_STOP = threading.Event()
_TTS_WORKER: Optional[threading.Thread] = None
_AUDIO_CACHE: Dict[str, bytes] = {}
_CACHE_LIMIT = 32
_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_THREAD: Optional[threading.Thread] = None
_ASYNC_LOCK = threading.Lock()


def _load_edge_tts():
    try:
        return importlib.import_module("edge_tts")
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed. Run: pip install edge-tts") from exc


def _detect_tone(text: str) -> str:
    """Disabled for speed. Always return neutral."""
    return "neutral"


def _voice_params(tone: str) -> Tuple[str, str]:
    if tone == "excited":
        return "+12%", "+8Hz"
    if tone == "curious":
        return "+4%", "+2Hz"
    if tone == "soft":
        return "-8%", "-4Hz"
    return "+0%", "+0Hz"


async def _synthesize(text: str, output_wav: Path, voice: str, rate: str, pitch: str) -> None:
    edge_tts = _load_edge_tts()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_mp3 = output_wav.with_suffix(".mp3")

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(temp_mp3))

    # Convert to WAV to keep a stable playback/output format.
    data, sample_rate = sf.read(str(temp_mp3), dtype="float32")
    sf.write(str(output_wav), data, sample_rate)

    if temp_mp3.exists():
        temp_mp3.unlink()


async def _synthesize_to_bytes(text: str, voice: str) -> bytes:
    """Stream audio directly without MP3 conversion overhead."""
    edge_tts = _load_edge_tts()
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%", pitch="+0Hz")

    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])

    return b"".join(chunks)


def _cached_audio_key(text: str, voice: str) -> str:
    return f"{voice}|{text.strip()}"


def _set_cache(key: str, value: bytes) -> None:
    if key in _AUDIO_CACHE:
        _AUDIO_CACHE.pop(key)
    _AUDIO_CACHE[key] = value
    while len(_AUDIO_CACHE) > _CACHE_LIMIT:
        # Remove oldest inserted item.
        oldest_key = next(iter(_AUDIO_CACHE))
        _AUDIO_CACHE.pop(oldest_key, None)


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP, _ASYNC_THREAD
    with _ASYNC_LOCK:
        if _ASYNC_LOOP is not None and _ASYNC_THREAD is not None and _ASYNC_THREAD.is_alive():
            return _ASYNC_LOOP

        loop = asyncio.new_event_loop()

        def _loop_runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_loop_runner, name="tts-async-loop", daemon=True)
        thread.start()

        _ASYNC_LOOP = loop
        _ASYNC_THREAD = thread
        return loop


def _run_async(coro):
    loop = _ensure_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _shutdown_async_loop() -> None:
    global _ASYNC_LOOP, _ASYNC_THREAD
    with _ASYNC_LOCK:
        if _ASYNC_LOOP is None:
            return
        _ASYNC_LOOP.call_soon_threadsafe(_ASYNC_LOOP.stop)
        if _ASYNC_THREAD is not None and _ASYNC_THREAD.is_alive():
            _ASYNC_THREAD.join(timeout=1.0)
        _ASYNC_LOOP = None
        _ASYNC_THREAD = None


def speak(text: str, use_memory_mode: Optional[bool] = None) -> Path:
    if not text.strip():
        return Path()

    config = load_config()
    tts_cfg = config["tts"]
    voice = tts_cfg["voice"]
    output_wav = Path(tts_cfg["output_file"])
    in_memory_mode = bool(tts_cfg.get("in_memory_mode", False))
    if use_memory_mode is not None:
        in_memory_mode = use_memory_mode

    # Skip tone detection for speed - always use neutral
    cache_key = _cached_audio_key(text, voice)

    if in_memory_mode:
        try:
            audio_bytes = _AUDIO_CACHE.get(cache_key)
            if audio_bytes is None:
                # Synthesize directly without MP3 conversion
                audio_bytes = _run_async(_synthesize_to_bytes(text, voice))
                _set_cache(cache_key, audio_bytes)
            # Play in background without blocking (no sd.wait())
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            sd.play(audio_data, sample_rate)
            return Path()
        except Exception as e:
            logger.debug(f"TTS in-memory mode failed, falling back: {e}")

    # File mode fallback - still async
    try:
        _run_async(_synthesize(text, output_wav, voice, "+0%", "+0Hz"))
        audio_data, sample_rate = sf.read(str(output_wav), dtype="float32")
        sd.play(audio_data, sample_rate)
    except Exception as e:
        logger.error(f"TTS failed: {e}")

    return output_wav


def _tts_worker_loop() -> None:
    while not _TTS_STOP.is_set() or not _TTS_QUEUE.empty():
        try:
            text = _TTS_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            dispatch_event("on_speak_start", text)
            speak(text)
            dispatch_event("on_speak_end", text)
        finally:
            _TTS_QUEUE.task_done()


def start_tts_worker() -> None:
    global _TTS_WORKER
    if _TTS_WORKER is not None and _TTS_WORKER.is_alive():
        return

    _TTS_STOP.clear()
    _TTS_WORKER = threading.Thread(target=_tts_worker_loop, name="tts-worker", daemon=True)
    _TTS_WORKER.start()


def enqueue_speak(text: str) -> None:
    if text.strip():
        start_tts_worker()
        if _TTS_QUEUE.full():
            try:
                _TTS_QUEUE.get_nowait()
                _TTS_QUEUE.task_done()
            except queue.Empty:
                pass
        _TTS_QUEUE.put(text)


def clear_tts_queue() -> None:
    while True:
        try:
            _TTS_QUEUE.get_nowait()
            _TTS_QUEUE.task_done()
        except queue.Empty:
            break


def wait_for_tts_queue(timeout: Optional[float] = None) -> None:
    if timeout is None:
        _TTS_QUEUE.join()
        return

    start = time.perf_counter()
    while _TTS_QUEUE.unfinished_tasks > 0:
        if time.perf_counter() - start >= timeout:
            return
        time.sleep(0.01)


def stop_tts_worker() -> None:
    _TTS_STOP.set()
    if _TTS_WORKER is not None and _TTS_WORKER.is_alive():
        _TTS_WORKER.join(timeout=1.0)


def warmup_tts() -> None:
    _load_edge_tts()
    try:
        _run_async(_synthesize_to_bytes("Ready.", "en-US-AnaNeural"))
    except Exception:
        # Warmup is best-effort and should not block startup.
        pass


atexit.register(_shutdown_async_loop)
