import json
import os
import random
from typing import Any, Dict, Generator, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
from pathlib import Path

from utils.config_loader import load_config
from utils.memory_db import init_memory_db, load_recent_messages, save_message

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CONFIG = load_config()
MAX_HISTORY = int(CONFIG["llm"].get("max_history_messages", 12))
MODEL = str(CONFIG["llm"].get("model", "openai/gpt-4o-mini"))
TEMPERATURE = float(CONFIG["llm"].get("temperature", 0.7))
PROVIDER = str(CONFIG["llm"].get("provider", "openrouter")).lower()
OPENROUTER_BASE_URL = str(
    CONFIG["llm"].get("openrouter_base_url", "https://openrouter.ai/api/v1")
)
OPENROUTER_MODEL = str(CONFIG["llm"].get("openrouter_model", MODEL))
OPENROUTER_SITE_URL = str(CONFIG["llm"].get("site_url", "http://localhost"))
OPENROUTER_APP_NAME = str(CONFIG["llm"].get("app_name", "Alpha Local Assistant"))
MEMORY_DB_PATH = str(CONFIG.get("memory", {}).get("db_path", "memory/memory.db"))


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
chat_history: List[Dict[str, str]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *_persisted_messages,
]
_backend: Optional[str] = None
_openai_client: Optional[Any] = None


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
    return (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or str(CONFIG["llm"].get("openrouter_api_key", "")).strip()
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
        _openai_client = OpenAI(
            api_key=_openrouter_key(),
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )
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
        _openai_client = OpenAI(
            api_key=_openrouter_key(),
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )

    response = _openai_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()


def _openrouter_stream(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=_openrouter_key(),
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )

    stream = _openai_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield str(delta)


def get_response(user_input: str) -> str:
    chat_history.append({"role": "user", "content": user_input})
    save_message(MEMORY_DB_PATH, "user", user_input)

    try:
        initialize_backend()
        reply = _openrouter_response(chat_history)
    except (RuntimeError, URLError, TimeoutError, OSError) as exc:
        reply = f"I could not reach the LLM backend right now: {exc}"
    except Exception as exc:
        reply = f"I hit an LLM error: {exc}"

    chat_history.append({"role": "assistant", "content": reply})
    save_message(MEMORY_DB_PATH, "assistant", reply)
    _trim_history()
    return reply


def get_response_stream(user_input: str) -> Generator[str, None, str]:
    chat_history.append({"role": "user", "content": user_input})
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

    final_reply = "".join(parts).strip()
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
