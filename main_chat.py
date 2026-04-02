"""Main Voice Assistant application with multi-modal AI capabilities."""

import logging
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
from process.enhanced_vision import capture_and_analyze
from utils.config_loader import load_config
from process.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Main application entry point for voice assistant with multi-modal support."""
    try:
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
        vision_turn_interval = max(1, int(vision_cfg.get("turn_interval", 2)))
        vision_auto_calibration = bool(vision_cfg.get("auto_calibration", True))
        vision_calibration_refresh_minutes = float(vision_cfg.get("calibration_refresh_minutes", 120.0))
        vision_calibration_sample_seconds = float(vision_cfg.get("calibration_sample_seconds", 1.8))
        enhanced_vision_enabled = bool(vision_cfg.get("enhanced_models", False))

        logger.info(f"Initializing {assistant_name} Voice Assistant")
        
        backend_status = get_backend_status()
        logger.info(f"LLM Backend: {backend_status}")
        
        if vad_mode_enabled:
            logger.info("VAD mode enabled")
        if vision_enabled:
            logger.info("Vision mode enabled")
            if enhanced_vision_enabled:
                logger.info("Enhanced vision models enabled (CLIP, BLIP, YOLO, LLaVA)")
            if vision_auto_calibration:
                calibration_status = ensure_live_calibration(
                    camera_index=vision_camera_index,
                    sample_seconds=vision_calibration_sample_seconds,
                    refresh_minutes=vision_calibration_refresh_minutes,
                )
                if calibration_status:
                    logger.info(f"Vision calibration: {calibration_status}")

        if warmup_enabled:
            logger.info("Warming up TTS")
            warmup_tts()
        if async_queue_enabled:
            logger.info("Starting TTS worker thread")
            start_tts_worker()

        logger.info("Assistant ready")
        main_loop(
            assistant_name=assistant_name,
            vad_mode_enabled=vad_mode_enabled,
            async_queue_enabled=async_queue_enabled,
            stream_enabled=stream_enabled,
            vision_enabled=vision_enabled,
            vision_camera_index=vision_camera_index,
            vision_sample_seconds=vision_sample_seconds,
            vision_turn_interval=vision_turn_interval,
            enhanced_vision_enabled=enhanced_vision_enabled,
        )
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise


def main_loop(
    assistant_name: str,
    vad_mode_enabled: bool,
    async_queue_enabled: bool,
    stream_enabled: bool,
    vision_enabled: bool,
    vision_camera_index: int,
    vision_sample_seconds: float,
    vision_turn_interval: int,
    enhanced_vision_enabled: bool = False,
) -> None:
    """Main conversation loop with error handling."""
    try:
        turn_count = 0
        while True:
            try:
                if not vad_mode_enabled:
                    command = input("Press Enter to talk (or type 'q' to quit): ").strip().lower()
                    if command == "q":
                        logger.info("User requested exit")
                        print(f"Bye. {assistant_name} is going on break.")
                        break

                user_text = record_and_transcribe()
                if not user_text:
                    logger.debug("No speech detected")
                    continue

                logger.info(f"User input: {user_text}")
                turn_count += 1
                lower_user_text = user_text.lower()
                
                # Vision calibration command
                if vision_enabled and any(
                    phrase in lower_user_text
                    for phrase in [
                        "calibrate vision",
                        "vision calibration",
                        "calibrate camera",
                        "calibrate color",
                    ]
                ):
                    logger.info("Vision calibration requested")
                    result = calibrate_clothing_color(
                        camera_index=vision_camera_index,
                        sample_seconds=max(1.6, vision_sample_seconds * 8.0),
                    )
                    try:
                        if async_queue_enabled:
                            enqueue_speak(result)
                        else:
                            speak(result)
                    except Exception as exc:
                        logger.error(f"TTS playback error: {exc}")
                        print(f"[TTS] Playback error: {exc}")
                    continue

                # Automation handling
                automation_result = handle_automation(user_text)
                if not automation_result:
                    normalized_intent = get_automation_intent(user_text)
                    if normalized_intent:
                        logger.info(f"Automation intent: {normalized_intent}")
                        automation_result = handle_automation(normalized_intent)
                        
                if automation_result:
                    logger.info(f"Automation executed: {automation_result}")
                    try:
                        if async_queue_enabled:
                            enqueue_speak(automation_result)
                        else:
                            speak(automation_result)
                    except Exception as exc:
                        logger.error(f"TTS playback error: {exc}")
                        print(f"[TTS] Playback error: {exc}")
                    continue

                # Vision analysis - basic or enhanced
                emotion_label = None
                vision_context = None
                
                should_run_vision = vision_enabled and (
                    (turn_count % vision_turn_interval == 0)
                    or ("how do i look" in lower_user_text)
                    or ("what am i wearing" in lower_user_text)
                    or ("what color" in lower_user_text)
                    or ("dress" in lower_user_text)
                    or ("shirt" in lower_user_text)
                )

                if should_run_vision:
                    if enhanced_vision_enabled:
                        # Use advanced AI models for scene understanding
                        try:
                            logger.debug("Running enhanced vision analysis...")
                            vision_context = capture_and_analyze(
                                camera_index=vision_camera_index,
                                sample_seconds=vision_sample_seconds,
                            )
                            if vision_context.get("available"):
                                logger.info(f"Enhanced vision models used: {vision_context.get('models_used')}")
                            else:
                                logger.debug("Enhanced vision not available")
                        except Exception as e:
                            logger.error(f"Enhanced vision error: {e}")
                    else:
                        # Use basic emotion detection
                        emotion_label = detect_emotion_snapshot(
                            camera_index=vision_camera_index,
                            sample_seconds=vision_sample_seconds,
                        )
                        if emotion_label:
                            logger.info(f"Vision emotion: {emotion_label}")
                        else:
                            logger.debug("No clear face detected")

                # LLM response generation
                if stream_enabled:
                    logger.debug("Using streaming response")
                    parts: list[str] = []
                    print(f"[{assistant_name}] ", end="", flush=True)
                    for piece in get_response_stream(
                        user_text, 
                        emotion_label=emotion_label,
                        vision_context=vision_context,
                    ):
                        parts.append(piece)
                        print(piece, end="", flush=True)
                    print()
                    ai_text = "".join(parts).strip()
                    ai_text = add_reactions(ai_text)
                    ai_text = style_text(ai_text)
                else:
                    logger.debug("Using non-streaming response")
                    ai_text = get_response(
                        user_text, 
                        emotion_label=emotion_label,
                        vision_context=vision_context,
                    )
                    ai_text = add_reactions(ai_text)
                    ai_text = style_text(ai_text)

                logger.info(f"Assistant response ready")

                # TTS playback
                try:
                    if async_queue_enabled:
                        enqueue_speak(ai_text)
                    else:
                        speak(ai_text)
                except Exception as exc:
                    logger.error(f"TTS playback error: {exc}")
                    print(f"[TTS] Playback error: {exc}")
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}", exc_info=True)
                
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        print(f"[FATAL ERROR] {e}")
        raise



if __name__ == "__main__":
    main()