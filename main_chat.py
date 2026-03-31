from process.llm import (
    add_reactions,
    get_backend_status,
    get_automation_intent,
    get_response,
    get_response_stream,
    style_text,
)
from process.automation import handle_automation
from process.stt import record_and_transcribe
from process.tts import enqueue_speak, speak, start_tts_worker, warmup_tts
from process.vision import (
    calibrate_clothing_color,
    detect_emotion_snapshot,
    ensure_live_calibration,
)
from utils.config_loader import load_config


def main() -> None:
    config = load_config()
    assistant_name = config["assistant"]["name"]
    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})
    vad_mode_enabled = bool(stt_cfg.get("vad_mode", False))
    warmup_enabled = bool(tts_cfg.get("warmup", True))
    async_queue_enabled = bool(tts_cfg.get("async_queue", True))
    stream_enabled = bool(config.get("llm", {}).get("streaming", False))
    vision_cfg = config.get("vision", {})
    vision_enabled = bool(vision_cfg.get("enabled", False))
    vision_camera_index = int(vision_cfg.get("camera_index", 0))
    vision_sample_seconds = float(vision_cfg.get("sample_seconds", 1.0))
    vision_auto_calibration = bool(vision_cfg.get("auto_calibration", True))
    vision_calibration_refresh_minutes = float(vision_cfg.get("calibration_refresh_minutes", 120.0))
    vision_calibration_sample_seconds = float(vision_cfg.get("calibration_sample_seconds", 1.8))

    print(f"\n=== {assistant_name} Voice Assistant Prototype ===")
    print(f"[LLM] Backend: {get_backend_status()}\n")
    if vad_mode_enabled:
        print("[STT] VAD mode enabled. Speak naturally; no Enter needed.\n")
    if vision_enabled:
        print("[Vision] Say 'calibrate vision' once to run one-click color calibration.\n")
        if vision_auto_calibration:
            calibration_status = ensure_live_calibration(
                camera_index=vision_camera_index,
                sample_seconds=vision_calibration_sample_seconds,
                refresh_minutes=vision_calibration_refresh_minutes,
            )
            if calibration_status:
                print(f"[Vision] {calibration_status}")
            print()

    if warmup_enabled:
        warmup_tts()
    if async_queue_enabled:
        start_tts_worker()

    while True:
        if not vad_mode_enabled:
            command = input("Press Enter to talk (or type 'q' to quit): ").strip().lower()
            if command == "q":
                print(f"Bye. {assistant_name} is going on break.")
                break

        user_text = record_and_transcribe()
        if not user_text:
            if not vad_mode_enabled:
                print("[STT] I did not catch that. Try again.")
            continue

        lower_user_text = user_text.lower()
        if vision_enabled and any(
            phrase in lower_user_text
            for phrase in [
                "calibrate vision",
                "vision calibration",
                "calibrate camera",
                "calibrate color",
            ]
        ):
            result = calibrate_clothing_color(
                camera_index=vision_camera_index,
                sample_seconds=max(1.6, vision_sample_seconds * 8.0),
            )
            print(f"[Vision] {result}")
            try:
                if async_queue_enabled:
                    enqueue_speak(result)
                else:
                    speak(result)
            except Exception as exc:
                print(f"[TTS] Playback error: {exc}")
            continue

        automation_result = handle_automation(user_text)
        if not automation_result:
            normalized_intent = get_automation_intent(user_text)
            if normalized_intent:
                automation_result = handle_automation(normalized_intent)
        if automation_result:
            print(f"[User] {user_text}")
            print(f"[{assistant_name}] {automation_result}")
            try:
                if async_queue_enabled:
                    enqueue_speak(automation_result)
                else:
                    speak(automation_result)
            except Exception as exc:
                print(f"[TTS] Playback error: {exc}")
            continue

        emotion_label = None
        if vision_enabled:
            emotion_label = detect_emotion_snapshot(
                camera_index=vision_camera_index,
                sample_seconds=vision_sample_seconds,
            )
            if emotion_label:
                print(f"[Vision] Emotion detected: {emotion_label}")
            else:
                print("[Vision] No clear face detected this turn.")

        if stream_enabled:
            parts: list[str] = []
            for piece in get_response_stream(user_text, emotion_label=emotion_label):
                parts.append(piece)
            ai_text = "".join(parts).strip()
            ai_text = add_reactions(ai_text)
            ai_text = style_text(ai_text)
        else:
            ai_text = get_response(user_text, emotion_label=emotion_label)
            ai_text = add_reactions(ai_text)
            ai_text = style_text(ai_text)

        print(f"[User] {user_text}")
        print(f"[{assistant_name}] {ai_text}")

        try:
            if async_queue_enabled:
                enqueue_speak(ai_text)
            else:
                speak(ai_text)
        except Exception as exc:
            print(f"[TTS] Playback error: {exc}")


if __name__ == "__main__":
    main()