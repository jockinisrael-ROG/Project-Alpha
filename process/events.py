from typing import Any, Callable, Dict, List


EventCallback = Callable[..., None]

_ALLOWED_EVENTS = {
    "on_user_input",
    "on_response",
    "on_speak_start",
    "on_speak_end",
}

_callbacks: Dict[str, List[EventCallback]] = {event: [] for event in _ALLOWED_EVENTS}


def register_event(event_name: str, callback: EventCallback) -> None:
    if event_name not in _callbacks:
        raise ValueError(f"Unsupported event: {event_name}")
    _callbacks[event_name].append(callback)


def dispatch_event(event_name: str, *args: Any, **kwargs: Any) -> None:
    for callback in _callbacks.get(event_name, []):
        try:
            callback(*args, **kwargs)
        except Exception:
            # Event callbacks should never break the chat loop.
            continue
