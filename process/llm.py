"""Language Model inference module using OpenRouter API with streaming and memory support."""

import json
import logging
import os
import random
import re
from typing import Any, Dict, Generator, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
from pathlib import Path

from utils.config_loader import load_config
from utils.memory_db import init_memory_db, load_recent_messages, save_message
from process.logger import setup_logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = setup_logger(__name__)


CONFIG = load_config()
MAX_HISTORY = int(CONFIG["llm"].get("max_history_messages", 12))
MODEL = str(CONFIG["llm"].get("model", "openai/gpt-4o-mini"))
TEMPERATURE = float(CONFIG["llm"].get("temperature", 0.7))
MAX_TOKENS = int(CONFIG["llm"].get("max_tokens", 300))
PROVIDER = str(CONFIG["llm"].get("provider", "openrouter")).lower()
OPENROUTER_BASE_URL = str(
    CONFIG["llm"].get("openrouter_base_url", "https://openrouter.ai/api/v1")
)
OPENROUTER_MODEL = str(CONFIG["llm"].get("openrouter_model", MODEL))
OPENROUTER_SITE_URL = str(CONFIG["llm"].get("site_url", "http://localhost"))
OPENROUTER_APP_NAME = str(CONFIG["llm"].get("app_name", "Alpha Local Assistant"))
MEMORY_DB_PATH = str(CONFIG.get("memory", {}).get("db_path", "memory/memory.db"))

_DISALLOWED_PATTERNS = [
    (re.compile(r"\bidiot\b", re.IGNORECASE), "friend"),
    (re.compile(r"\bbaka\b", re.IGNORECASE), "friend"),
    (re.compile(r"\bhell\b", re.IGNORECASE), "heck"),
]


def sanitize_response(text: str) -> str:
    sanitized = text or ""
    for pattern, replacement in _DISALLOWED_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _load_system_prompt() -> str:
    personality_path = Path(__file__).resolve().parent.parent / "config" / "personality.txt"
    if personality_path.exists():
        text = personality_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return str(CONFIG["assistant"].get("system_prompt", "You are Alpha."))


SYSTEM_PROMPT = _load_system_prompt()
init_memory_db(MEMORY_DB_PATH)
_persisted_messages = load_recent_messages(MEMORY_DB_PATH, MAX_HISTORY)
for _msg in _persisted_messages:
    if _msg.get("role") == "assistant":
        _msg["content"] = sanitize_response(str(_msg.get("content", "")))
chat_history: List[Dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *_persisted_messages,
]
_backend: Optional[str] = None
_openai_client: Optional[Any] = None


def _build_user_message(
    user_input: str, 
    emotion_label: Optional[str] = None,
    vision_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build enhanced user message with optional vision and emotion context.
    
    Args:
        user_input: The user's message
        emotion_label: Optional emotion/scene tags from basic vision
        vision_context: Optional rich vision context from enhanced_vision module
        
    Returns:
        Formatted message with vision context for the LLM
    """
    message_parts = [user_input]
    
    # Add rich vision understanding if available
    if vision_context and vision_context.get("available"):
        from process.enhanced_vision import format_vision_for_llm
        vision_prompt = format_vision_for_llm(vision_context)
        if vision_prompt and "[Vision unavailable]" not in vision_prompt:
            message_parts.append(f"\n{vision_prompt}")
            logger.debug(f"Added enhanced vision context to message")
    
    # Legacy emotion-based context (for backward compatibility)
    elif emotion_label:
        # Parse scene data from pipe-separated tags
        tags = emotion_label.split("|")
        emotion = tags[0] if tags else "neutral"
        
        # Extract detailed scene features
        clothing_color = next((t.replace("wearing_", "") for t in tags if t.startswith("wearing_")), None)
        lighting = next((t for t in tags if t in ["very_dim", "dim", "bright", "very_bright"]), None)
        bg_complexity = next((t for t in tags if t in ["cluttered", "very_cluttered", "moderate", "clean", "organized_space"]), None)
        bg_objects = next((int(t.replace("objects_", "")) for t in tags if t.startswith("objects_")), 0)
        has_multiple_people = "multiple_people" in tags
        
        # Legacy support for old tag format
        has_dim_lighting = "dim_lighting" in tags or lighting in ["dim", "very_dim"]
        has_bright_lighting = "bright_lighting" in tags or lighting in ["bright", "very_bright"]
        is_cluttered = "cluttered_space" in tags or bg_complexity in ["cluttered", "very_cluttered"]
        color = next((t.replace("color_", "") for t in tags if t.startswith("color_")), None)
        
        # Build emotion context
        emotion_contexts = {
            "happy": "they appear happy and smiling. Respond warmly and match their positive energy!",
            "sad": "they appear sad or upset. Be empathetic, caring, and offer support naturally.",
            "angry": "they appear angry or frustrated. Be calm, understanding, and help them cool down.",
            "neutral": "they have a neutral expression. Be friendly and welcoming.",
        }
        emotion_context = emotion_contexts.get(emotion, emotion)
        
        # Build scene context and suggestions
        scene_suggestions = []
        
        # Clothing observations
        if clothing_color:
            scene_suggestions.append(f"they're wearing {clothing_color} - nice style choice!")
        
        # Lighting observations
        if lighting == "very_dim":
            scene_suggestions.append("their room is very dark - suggest turning on lights for better focus and mood")
        elif lighting == "dim":
            scene_suggestions.append("the lighting is a bit dim - they might benefit from brighter light")
        elif lighting == "very_bright":
            scene_suggestions.append("it's quite bright where they are - maybe suggest adjusting for comfort")
        elif has_dim_lighting:
            scene_suggestions.append("their environment looks dim - they might benefit from turning on a light")
        elif has_bright_lighting:
            scene_suggestions.append("the lighting is quite bright - maybe suggest adjusting it for comfort")
        
        # Background/environment observations
        if bg_complexity == "very_cluttered":
            scene_suggestions.append("their space looks quite cluttered - mention that organizing could help them focus")
        elif bg_complexity == "cluttered" or is_cluttered:
            scene_suggestions.append("their space looks a bit busy - organizing could help with focus")
        elif bg_complexity == "moderate":
            scene_suggestions.append("their workspace has some items around")
        
        if bg_objects > 1:
            scene_suggestions.append(f"you can see about {bg_objects} distinct items")
        
        # Color mood
        if color:
            scene_suggestions.append(f"their surroundings have a {color} tone")
        
        # People detection
        if has_multiple_people:
            scene_suggestions.append("there are multiple people around them")
        
        scene_part = ""
        if scene_suggestions:
            scene_part = "\n[Scene: " + "; ".join(scene_suggestions) + "]"
        
        message_parts.append(
            f"\n[Camera emotion: {emotion_context}{scene_part}]"
        )
        logger.debug(f"Added emotion context to message")
    
    return "".join(message_parts)




def add_reactions(text: str) -> str:
    prefixes = ["*hmph*", "*sigh*", "*looks away*", "*blushes slightly*"]
    if text and random.random() < 0.45:
        return f"{random.choice(prefixes)} {text}"
    return text


def style_text(text: str) -> str:
    styled = text.strip()
    if not styled:
        return styled

    if "!" in styled and random.random() < 0.25:
        styled = styled.replace("!", "!!")
        while "!!!" in styled:
            styled = styled.replace("!!!", "!!")

    if random.random() < 0.3 and not styled.endswith(("...", "!", "?")):
        styled = f"{styled}..."

    return styled


def _trim_history() -> None:
    if len(chat_history) > MAX_HISTORY + 1:
        recent = chat_history[-MAX_HISTORY:]
        chat_history[:] = [{"role": "system", "content": SYSTEM_PROMPT}, *recent]


def _post_json(url: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> Dict:
    request = Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _openrouter_key() -> str:
    raw_key = (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or str(CONFIG["llm"].get("openrouter_api_key", "")).strip()
    )
    # Guard against accidental quoting from shell/env parsing.
    return raw_key.strip().strip('"').strip("'")


def _make_openrouter_client() -> Any:
    key = _openrouter_key()
    if not key:
        raise RuntimeError("OpenRouter API key is missing")

    return OpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )


def _can_use_openrouter() -> bool:
    if OpenAI is None or not _openrouter_key():
        return False

    try:
        request = Request(
            f"{OPENROUTER_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {_openrouter_key()}"},
            method="GET",
        )
        with urlopen(request, timeout=6):
            pass
        return True
    except Exception:
        return False


def initialize_backend() -> str:
    global _backend, _openai_client

    if _backend is not None:
        return _backend

    if PROVIDER == "openrouter" and _can_use_openrouter():
        _openai_client = _make_openrouter_client()
        _backend = "openrouter"
        return _backend

    raise RuntimeError(
        "OpenRouter is not available. Set OPENROUTER_API_KEY (or llm.openrouter_api_key in config/config.yaml) and check internet connection."
    )


def get_backend_status() -> str:
    try:
        backend = initialize_backend()
        return f"{backend} ({OPENROUTER_MODEL})"
    except Exception as exc:
        return f"unavailable ({exc})"


def _openrouter_response(messages: List[Dict[str, str]]) -> str:
    global _openai_client
    if _openai_client is None:
        _openai_client = _make_openrouter_client()

    response = _openai_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def _openrouter_stream(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    global _openai_client
    if _openai_client is None:
        _openai_client = _make_openrouter_client()

    stream = _openai_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield str(delta)


def get_response(
    user_input: str, 
    emotion_label: Optional[str] = None,
    vision_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Get a non-streaming LLM response.
    
    Args:
        user_input: User's message
        emotion_label: Optional emotion labels from basic vision
        vision_context: Optional rich vision context dict from enhanced_vision module
        
    Returns:
        LLM response text
    """
    llm_user_message = _build_user_message(user_input, emotion_label, vision_context)
    chat_history.append({"role": "user", "content": llm_user_message})
    save_message(MEMORY_DB_PATH, "user", user_input)

    try:
        initialize_backend()
        reply = _openrouter_response(chat_history)
    except (RuntimeError, URLError, TimeoutError, OSError) as exc:
        reply = f"I could not reach the LLM backend right now: {exc}"
    except Exception as exc:
        reply = f"I hit an LLM error: {exc}"

    reply = sanitize_response(reply)
    chat_history.append({"role": "assistant", "content": reply})
    save_message(MEMORY_DB_PATH, "assistant", reply)
    _trim_history()
    return reply


def get_response_stream(
    user_input: str, 
    emotion_label: Optional[str] = None,
    vision_context: Optional[Dict[str, Any]] = None,
) -> Generator[str, None, str]:
    """Get a streaming LLM response.
    
    Args:
        user_input: User's message
        emotion_label: Optional emotion labels from basic vision
        vision_context: Optional rich vision context dict from enhanced_vision module
        
    Yields:
        Response text chunks
        
    Returns:
        Final complete response text
    """
    llm_user_message = _build_user_message(user_input, emotion_label, vision_context)
    chat_history.append({"role": "user", "content": llm_user_message})
    save_message(MEMORY_DB_PATH, "user", user_input)

    parts: List[str] = []
    try:
        initialize_backend()
        for piece in _openrouter_stream(chat_history):
            parts.append(piece)
            yield piece
    except (RuntimeError, URLError, TimeoutError, OSError) as exc:
        error_reply = f"I could not reach the LLM backend right now: {exc}"
        parts = [error_reply]
        yield error_reply
    except Exception as exc:
        error_reply = f"I hit an LLM error: {exc}"
        parts = [error_reply]
        yield error_reply

    final_reply = sanitize_response("".join(parts).strip())
    chat_history.append({"role": "assistant", "content": final_reply})
    save_message(MEMORY_DB_PATH, "assistant", final_reply)
    _trim_history()
    return final_reply


def get_automation_intent(user_input: str) -> Optional[str]:
    """Use the LLM to normalize natural command phrasing into a canonical automation intent.

    Returns one of:
    - open youtube
    - open google
    - open gmail
    - open github
    - open chatgpt
    - open notepad
    - open calculator
    - open cmd
    - open task manager
    - open file explorer
    - open settings
    - open downloads
    - open documents
    - open desktop
    - open chrome
    - open edge
    - open vscode
    - open netflix
    - open stackoverflow
    - open weather
    - open whatsapp
    - time
    - date
    - search google for <query>
    - search youtube for <query>
    or None if no automation intent is detected.
    """
    text = user_input.strip()
    if not text:
        return None

    system = (
        "You are an intent normalizer for local desktop automation. "
        "Return ONLY one exact token from this set: "
        "open youtube | open google | open gmail | open github | open chatgpt | "
        "open whatsapp | open notepad | open calculator | open cmd | open task manager | "
        "open file explorer | open settings | open downloads | open documents | open desktop | "
        "open chrome | open edge | open vscode | open netflix | open stackoverflow | open weather | "
        "time | date | search google for <query> | search youtube for <query> | none. "
        "For search requests, keep the query words unchanged. "
        "If the user is not clearly asking for one of these, return none."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]

    try:
        initialize_backend()
        normalized = _openrouter_response(messages).strip().lower()
    except Exception:
        return None

    allowed = {
        "open youtube",
        "open google",
        "open gmail",
        "open github",
        "open chatgpt",
        "open notepad",
        "open calculator",
        "open cmd",
        "open task manager",
        "open file explorer",
        "open settings",
        "open downloads",
        "open documents",
        "open desktop",
        "open chrome",
        "open edge",
        "open vscode",
        "open netflix",
        "open stackoverflow",
        "open weather",
        "open whatsapp",
        "time",
        "date",
    }
    if normalized in allowed:
        return normalized

    if normalized.startswith("search google for ") and len(normalized) > len("search google for "):
        return normalized

    if normalized.startswith("search youtube for ") and len(normalized) > len("search youtube for "):
        return normalized

    if normalized in {"none", "no", "null", "n/a", ""}:
        return None

    return None
