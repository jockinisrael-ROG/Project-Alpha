"""Computer vision module for emotion detection and color calibration using OpenCV."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from process.logger import setup_logger

logger = setup_logger(__name__)


_FACE_CASCADE = None
_SMILE_CASCADE = None
_EYES_CASCADE = None
_WARNED_MISSING_CV = False
_WARNED_CAMERA_UNAVAILABLE = False
_WARNED_CASCADE_UNAVAILABLE = False
_CALIBRATION_CACHE = None
_CALIBRATION_LOADED = False
_LAST_STABLE_COLOR = "unknown"
_LAST_STABLE_COUNT = 0
_PENDING_COLOR = "unknown"
_PENDING_COLOR_HITS = 0
_LAST_STABLE_EMOTION = "neutral"
_PENDING_EMOTION = "neutral"
_PENDING_EMOTION_HITS = 0


def _get_calibration_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "vision_calibration.json"


def _hue_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, 180 - d)


def _hue_to_label(hue: int) -> str:
    h = int(hue)
    if h <= 10 or h >= 170:
        return "red"
    if 11 <= h <= 24:
        return "orange"
    if 25 <= h <= 35:
        return "yellow"
    if 36 <= h <= 85:
        return "green"
    if 86 <= h <= 130:
        return "blue"
    if 131 <= h <= 155:
        return "purple"
    return "pink"


def _load_calibration() -> Optional[dict]:
    global _CALIBRATION_CACHE, _CALIBRATION_LOADED
    if _CALIBRATION_LOADED:
        return _CALIBRATION_CACHE

    _CALIBRATION_LOADED = True
    path = _get_calibration_path()
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "hue" not in data or "label" not in data:
            return None
        _CALIBRATION_CACHE = data
        return _CALIBRATION_CACHE
    except Exception:
        return None


def _save_calibration(data: dict) -> bool:
    global _CALIBRATION_CACHE, _CALIBRATION_LOADED
    path = _get_calibration_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        _CALIBRATION_CACHE = data
        _CALIBRATION_LOADED = True
        return True
    except Exception:
        return False


def _ensure_cascades() -> bool:
    global _FACE_CASCADE, _SMILE_CASCADE, _EYES_CASCADE

    if cv2 is None:
        return False

    if _FACE_CASCADE is not None and _SMILE_CASCADE is not None and _EYES_CASCADE is not None:
        return True

    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
    eyes_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    _FACE_CASCADE = cv2.CascadeClassifier(face_path)
    _SMILE_CASCADE = cv2.CascadeClassifier(smile_path)
    _EYES_CASCADE = cv2.CascadeClassifier(eyes_path)

    return (
        _FACE_CASCADE is not None
        and _SMILE_CASCADE is not None
        and _EYES_CASCADE is not None
        and not _FACE_CASCADE.empty()
        and not _SMILE_CASCADE.empty()
        and not _EYES_CASCADE.empty()
    )


def _analyze_clothing_color(roi_bgr: np.ndarray) -> str:
    if cv2 is None or np is None or roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    # Use the center region to reduce background spill into clothing estimation.
    h, w = roi_bgr.shape[:2]
    cx1 = int(w * 0.2)
    cx2 = int(w * 0.8)
    cy1 = int(h * 0.05)
    cy2 = int(h * 0.95)
    cropped = roi_bgr[cy1:cy2, cx1:cx2] if cy2 > cy1 and cx2 > cx1 else roi_bgr

    small = cv2.resize(cropped, (64, 64))

    # Lightweight gray-world white balance to reduce webcam color cast.
    b_ch, g_ch, r_ch = cv2.split(small.astype(np.float32))
    b_mean = float(np.mean(b_ch)) + 1e-6
    g_mean = float(np.mean(g_ch)) + 1e-6
    r_mean = float(np.mean(r_ch)) + 1e-6
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    b_ch *= gray_mean / b_mean
    g_ch *= gray_mean / g_mean
    r_ch *= gray_mean / r_mean
    balanced = cv2.merge([
        np.clip(b_ch, 0, 255).astype(np.uint8),
        np.clip(g_ch, 0, 255).astype(np.uint8),
        np.clip(r_ch, 0, 255).astype(np.uint8),
    ])

    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Lower saturation threshold to catch pastel colors like light pink
    # Pastel colors have naturally low saturation
    color_mask = (s_ch >= 8) & (v_ch >= 35)
    color_count = int(np.count_nonzero(color_mask))
    total_count = int(hsv.shape[0] * hsv.shape[1])

    if color_count >= max(110, int(total_count * 0.08)):
        h_vals = h_ch[color_mask]
        # Weight votes by saturation/value so vivid clothing colors dominate dim background.
        w_vals = (s_ch[color_mask].astype(np.float32) / 255.0) * (v_ch[color_mask].astype(np.float32) / 255.0)

        def _w(mask: np.ndarray) -> float:
            if mask.size == 0:
                return 0.0
            return float(np.sum(w_vals[mask]))

        red_mask = (h_vals <= 10) | (h_vals >= 170)
        bins = {
            "red": _w(red_mask),
            "orange": _w((h_vals >= 11) & (h_vals <= 24)),
            "yellow": _w((h_vals >= 25) & (h_vals <= 35)),
            "green": _w((h_vals >= 36) & (h_vals <= 85)),
            "blue": _w((h_vals >= 86) & (h_vals <= 130)),
            "purple": _w((h_vals >= 131) & (h_vals <= 150)),
            "pink": _w((h_vals >= 151) & (h_vals <= 169)),
        }

        # Extra red rescue for low-light maroon/red shirts.
        red_ratio = float(np.count_nonzero(red_mask)) / float(max(1, h_vals.size))
        if red_ratio >= 0.14:
            return "red"

        best_color = max(bins, key=bins.get)
        best_score = bins[best_color]
        total_score = float(sum(bins.values())) + 1e-6
        live_confidence = best_score / total_score

        calibration = _load_calibration()
        if calibration and color_count > 0:
            cal_hue = int(calibration.get("hue", 0))
            cal_sat = int(calibration.get("sat", 70))
            cal_val = int(calibration.get("val", 70))
            hue_vals = h_ch[color_mask]
            sat_vals = s_ch[color_mask]
            val_vals = v_ch[color_mask]

            near_cal_mask = (
                np.array([_hue_distance(h, cal_hue) <= 12 for h in hue_vals])
                & (sat_vals >= max(30, int(cal_sat * 0.70)))
                & (val_vals >= max(30, int(cal_val * 0.60)))
            )
            near_ratio = float(np.count_nonzero(near_cal_mask)) / float(max(1, hue_vals.size))
            cal_label = str(calibration.get("label", "unknown"))

            # Only trust calibration when live signal is weak OR it agrees with live result.
            # If live color is strong and disagrees, live color wins.
            if live_confidence < 0.18 and near_ratio >= 0.20 and cal_label != "unknown":
                return cal_label
            if cal_label == best_color and near_ratio >= 0.10:
                return cal_label

        if live_confidence >= 0.20:
            return best_color

    mean_v = float(np.mean(v_ch))
    mean_s = float(np.mean(s_ch))
    median_h = int(np.median(h_ch))
    mode_h_bins = np.bincount(h_ch.flatten())
    mode_h = int(np.argmax(mode_h_bins))

    # Fallback pastel pink detection if main detection fails
    # Light pink: median/mode hue in pink range with low-moderate saturation
    if ((145 <= median_h <= 175) or (145 <= mode_h <= 175)) and mean_s >= 8:
        return "pink"
    
    # Pastel red/magenta
    if ((mode_h <= 10) or (mode_h >= 165)) and mean_s >= 12 and 150 <= mean_v <= 230:
        return "red"

    if mean_v <= 50:
        return "black"
    if mean_s <= 20 and mean_v >= 200:
        return "white"
    if mean_s <= 24:
        return "gray"
    return "unknown"


def calibrate_clothing_color(camera_index: int = 0, sample_seconds: float = 1.8) -> str:
    """Capture a short live sample and persist a clothing color profile for better accuracy."""
    if cv2 is None or np is None:
        return "OpenCV is not available, calibration skipped."

    if not _ensure_cascades():
        return "Vision cascades are unavailable, calibration skipped."

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return f"Camera index {camera_index} is not available for calibration."

    for _ in range(4):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.03)

    hue_samples = []
    sat_samples = []
    val_samples = []
    weights = []
    deadline = time.time() + max(1.2, float(sample_seconds))
    frames = 0

    try:
        while time.time() < deadline and frames < 30:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            frames += 1
            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            faces = _FACE_CASCADE.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(48, 48),
            )
            if len(faces) == 0:
                time.sleep(0.02)
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            clothing_y_start = min(h_frame - 1, y + h)
            clothing_y_end = min(h_frame, clothing_y_start + int(h * 1.6))
            clothing_x_start = max(0, x + int(0.12 * w))
            clothing_x_end = min(w_frame, x + int(0.88 * w))
            if clothing_y_end <= clothing_y_start or clothing_x_end <= clothing_x_start:
                time.sleep(0.02)
                continue

            roi = frame[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end]
            hsv = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2HSV)
            h_ch = hsv[:, :, 0].reshape(-1)
            s_ch = hsv[:, :, 1].reshape(-1)
            v_ch = hsv[:, :, 2].reshape(-1)

            mask = (s_ch >= 35) & (v_ch >= 35)
            if np.count_nonzero(mask) < 180:
                time.sleep(0.02)
                continue

            h_vals = h_ch[mask]
            s_vals = s_ch[mask]
            v_vals = v_ch[mask]
            w_vals = (s_vals.astype(np.float32) / 255.0) * (v_vals.astype(np.float32) / 255.0)

            hue_samples.extend(h_vals.tolist())
            sat_samples.extend(s_vals.tolist())
            val_samples.extend(v_vals.tolist())
            weights.extend(w_vals.tolist())
            time.sleep(0.02)
    finally:
        cap.release()

    if len(hue_samples) < 300 or len(weights) != len(hue_samples):
        return "Calibration failed: not enough clothing signal. Face camera with upper body visible and try again."

    hist = np.zeros((180,), dtype=np.float32)
    for i, h in enumerate(hue_samples):
        hist[int(h)] += float(weights[i])

    peak_hue = int(np.argmax(hist))
    label = _hue_to_label(peak_hue)
    sat_med = int(np.median(np.array(sat_samples, dtype=np.float32)))
    val_med = int(np.median(np.array(val_samples, dtype=np.float32)))

    payload = {
        "hue": peak_hue,
        "label": label,
        "sat": sat_med,
        "val": val_med,
        "captured_at": int(time.time()),
    }
    if not _save_calibration(payload):
        return "Calibration computed, but failed to save profile."

    return f"Calibration saved. I will treat your clothing color as {label} with higher priority."


def ensure_live_calibration(
    camera_index: int = 0,
    sample_seconds: float = 1.8,
    refresh_minutes: float = 120.0,
) -> Optional[str]:
    """Ensure a clothing color calibration profile exists and refresh it periodically."""
    calibration = _load_calibration()
    now_ts = int(time.time())
    refresh_seconds = int(max(60.0, float(refresh_minutes) * 60.0))

    needs_refresh = calibration is None
    if calibration is not None:
        captured_at = int(calibration.get("captured_at", 0) or 0)
        if captured_at <= 0 or (now_ts - captured_at) >= refresh_seconds:
            needs_refresh = True

    if not needs_refresh:
        label = str(calibration.get("label", "unknown"))
        return f"Calibration active: {label}."

    result = calibrate_clothing_color(camera_index=camera_index, sample_seconds=sample_seconds)
    return result


def _build_scene_tags(emotion: str, scene_data: dict) -> str:
    tags = [emotion]

    clothing = scene_data.get("dominant_clothing_color", "unknown")
    if clothing != "unknown":
        tags.append(f"wearing_{clothing}")

    lighting = scene_data.get("lighting_condition", "normal")
    if lighting != "normal":
        tags.append(lighting)

    complexity = scene_data.get("background_complexity", "clean")
    if complexity != "clean":
        tags.append(complexity)

    obj_count = int(scene_data.get("background_objects", 0))
    if obj_count > 0:
        tags.append(f"objects_{obj_count}")

    if int(scene_data.get("face_count", 0)) > 1:
        tags.append("multiple_people")

    return "|".join(tags)


def _resolve_stable_clothing_color(clothing_votes: dict) -> str:
    """Resolve clothing color with anti-flicker logic across turns."""
    global _LAST_STABLE_COLOR, _LAST_STABLE_COUNT, _PENDING_COLOR, _PENDING_COLOR_HITS

    if not clothing_votes:
        return _LAST_STABLE_COLOR if _LAST_STABLE_COLOR != "unknown" else "unknown"

    sorted_votes = sorted(clothing_votes.items(), key=lambda item: item[1], reverse=True)
    top_color, top_count = sorted_votes[0]
    second_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    total_votes = sum(clothing_votes.values())
    confidence = float(top_count) / float(max(1, total_votes))

    # First confident detection sets the stable color immediately.
    if _LAST_STABLE_COLOR == "unknown":
        _LAST_STABLE_COLOR = top_color
        _LAST_STABLE_COUNT = top_count
        _PENDING_COLOR = "unknown"
        _PENDING_COLOR_HITS = 0
        return _LAST_STABLE_COLOR

    # If current top agrees with stable color, keep it and clear pending transition.
    if top_color == _LAST_STABLE_COLOR:
        _LAST_STABLE_COUNT = max(_LAST_STABLE_COUNT, top_count)
        _PENDING_COLOR = "unknown"
        _PENDING_COLOR_HITS = 0
        return _LAST_STABLE_COLOR

    # Require stronger evidence before switching to a new color.
    strong_turn = top_count >= 2 and confidence >= 0.66 and (top_count - second_count) >= 1
    very_strong_turn = top_count >= 3 and confidence >= 0.75

    if very_strong_turn:
        _LAST_STABLE_COLOR = top_color
        _LAST_STABLE_COUNT = top_count
        _PENDING_COLOR = "unknown"
        _PENDING_COLOR_HITS = 0
        return _LAST_STABLE_COLOR

    if strong_turn:
        if _PENDING_COLOR == top_color:
            _PENDING_COLOR_HITS += 1
        else:
            _PENDING_COLOR = top_color
            _PENDING_COLOR_HITS = 1

        # Switch only after repeated strong confirmation across turns.
        if _PENDING_COLOR_HITS >= 2:
            _LAST_STABLE_COLOR = top_color
            _LAST_STABLE_COUNT = top_count
            _PENDING_COLOR = "unknown"
            _PENDING_COLOR_HITS = 0
            return _LAST_STABLE_COLOR

    # Weak/ambiguous turns keep previous stable color.
    return _LAST_STABLE_COLOR


def _resolve_stable_emotion(emotion_votes: dict) -> str:
    """Resolve emotion with anti-flicker logic across turns."""
    global _LAST_STABLE_EMOTION, _PENDING_EMOTION, _PENDING_EMOTION_HITS

    if not emotion_votes:
        return _LAST_STABLE_EMOTION

    sorted_votes = sorted(emotion_votes.items(), key=lambda item: item[1], reverse=True)
    top_emotion, top_count = sorted_votes[0]
    second_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    total_votes = sum(emotion_votes.values())
    confidence = float(top_count) / float(max(1, total_votes))

    if top_emotion == _LAST_STABLE_EMOTION:
        _PENDING_EMOTION = "neutral"
        _PENDING_EMOTION_HITS = 0
        return _LAST_STABLE_EMOTION

    # Allow confident happy quickly to keep responses lively.
    if top_emotion == "happy" and top_count >= 1 and confidence >= 0.45:
        _LAST_STABLE_EMOTION = "happy"
        _PENDING_EMOTION = "neutral"
        _PENDING_EMOTION_HITS = 0
        return _LAST_STABLE_EMOTION

    # If currently neutral, allow a strong one-turn switch to avoid getting stuck.
    if (
        _LAST_STABLE_EMOTION == "neutral"
        and top_emotion in {"sad", "angry"}
        and top_count >= 3
        and confidence >= 0.70
    ):
        _LAST_STABLE_EMOTION = top_emotion
        _PENDING_EMOTION = "neutral"
        _PENDING_EMOTION_HITS = 0
        return _LAST_STABLE_EMOTION

    # For sad/angry, require repeated confirmation to avoid false positives.
    strong_turn = top_count >= 2 and confidence >= 0.52 and (top_count - second_count) >= 1
    if strong_turn:
        if _PENDING_EMOTION == top_emotion:
            _PENDING_EMOTION_HITS += 1
        else:
            _PENDING_EMOTION = top_emotion
            _PENDING_EMOTION_HITS = 1

        if _PENDING_EMOTION_HITS >= 2:
            _LAST_STABLE_EMOTION = top_emotion
            _PENDING_EMOTION = "neutral"
            _PENDING_EMOTION_HITS = 0
            return _LAST_STABLE_EMOTION

    # Ambiguous turns default to neutral instead of over-triggering sad/angry.
    if top_emotion == "neutral" and confidence >= 0.35:
        _LAST_STABLE_EMOTION = "neutral"
        _PENDING_EMOTION = "neutral"
        _PENDING_EMOTION_HITS = 0
        return _LAST_STABLE_EMOTION

    return _LAST_STABLE_EMOTION


def detect_emotion_snapshot(
    camera_index: int = 0,
    sample_seconds: float = 0.20,
    frame_interval: float = 0.035,
) -> Optional[str]:
    """Return compact scene/emotion tags without showing or storing camera frames."""
    global _WARNED_MISSING_CV, _WARNED_CAMERA_UNAVAILABLE, _WARNED_CASCADE_UNAVAILABLE

    if cv2 is None:
        if not _WARNED_MISSING_CV:
            logger.debug("OpenCV not installed")
            _WARNED_MISSING_CV = True
        return None

    if not _ensure_cascades():
        if not _WARNED_CASCADE_UNAVAILABLE:
            logger.debug("Could not load OpenCV cascades")
            _WARNED_CASCADE_UNAVAILABLE = True
        return None

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        if not _WARNED_CAMERA_UNAVAILABLE:
            logger.debug(f"Camera index {camera_index} not available")
            _WARNED_CAMERA_UNAVAILABLE = True
        return None

    for _ in range(2):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.02)

    emotion_votes = {"happy": 0, "sad": 0, "angry": 0, "neutral": 0}
    clothing_votes = {}
    scene_data = {
        "brightness": 128.0,
        "background_objects": 0,
        "face_count": 0,
        "dominant_clothing_color": "unknown",
        "background_complexity": "clean",
        "lighting_condition": "normal",
    }

    frames_processed = 0
    deadline = time.time() + max(0.08, float(sample_seconds))

    try:
        while time.time() < deadline and frames_processed < 4:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(frame_interval)
                continue

            frames_processed += 1
            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            brightness = float(gray.mean())
            scene_data["brightness"] = brightness
            if brightness < 70:
                scene_data["lighting_condition"] = "very_dim"
            elif brightness < 100:
                scene_data["lighting_condition"] = "dim"
            elif brightness > 200:
                scene_data["lighting_condition"] = "very_bright"
            elif brightness > 170:
                scene_data["lighting_condition"] = "bright"
            else:
                scene_data["lighting_condition"] = "normal"

            faces = _FACE_CASCADE.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(48, 48),
            )

            scene_data["face_count"] = max(scene_data["face_count"], int(len(faces)))

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = gray_eq[y : y + h, x : x + w]

                smiles = _SMILE_CASCADE.detectMultiScale(
                    face_roi,
                    scaleFactor=1.25,
                    minNeighbors=5,
                    minSize=(20, 20),
                )
                eyes = _EYES_CASCADE.detectMultiScale(
                    face_roi,
                    scaleFactor=1.12,
                    minNeighbors=4,
                    minSize=(10, 10),
                )

                upper_roi = face_roi[: max(1, h // 3), :]
                lower_roi = face_roi[max(0, (2 * h) // 3) :, :]
                upper_mean = float(upper_roi.mean()) if upper_roi.size > 0 else 128.0
                lower_mean = float(lower_roi.mean()) if lower_roi.size > 0 else 128.0
                eye_count = int(len(eyes))
                smile_count = int(len(smiles))

                # Per-frame emotion classification: prefer clear signals over strict thresholds.
                frame_emotion = "neutral"
                if smile_count > 0:
                    frame_emotion = "happy"
                elif smile_count == 0 and eye_count >= 2:
                    # Eyes detected but no smile: likely neutral or concentrating
                    frame_emotion = "neutral"
                elif eye_count == 0:
                    # No eyes detected: either sad (eyes down) or poor detection
                    # Rely on lower face brightness as secondary signal
                    if lower_mean < 90 and brightness < 140:
                        frame_emotion = "sad"
                    else:
                        frame_emotion = "neutral"
                else:
                    # Some eyes detected, no smile: check brightness patterns
                    if upper_mean < 85 and brightness >= 75:
                        frame_emotion = "angry"
                    else:
                        frame_emotion = "neutral"

                emotion_votes[frame_emotion] = emotion_votes.get(frame_emotion, 0) + 1

                # Torso-focused ROI below face to reduce background contamination.
                clothing_y_start = min(h_frame - 1, y + h)
                clothing_y_end = min(h_frame, clothing_y_start + int(h * 1.5))
                clothing_x_start = max(0, x + int(0.1 * w))
                clothing_x_end = min(w_frame, x + int(0.9 * w))
                if clothing_y_end > clothing_y_start and clothing_x_end > clothing_x_start:
                    clothing_roi = frame[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end]
                    color = _analyze_clothing_color(clothing_roi)
                    if color != "unknown":
                        clothing_votes[color] = clothing_votes.get(color, 0) + 1

            bg_roi = gray[int(h_frame * 0.45) :, :]
            if bg_roi.size > 0:
                bg_edges = cv2.Canny(bg_roi, 40, 120)
                contours, _ = cv2.findContours(bg_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant = 0
                for contour in contours:
                    if cv2.contourArea(contour) > 600:
                        significant += 1
                scene_data["background_objects"] = min(8, significant)

                if significant > 10:
                    scene_data["background_complexity"] = "very_cluttered"
                elif significant > 5:
                    scene_data["background_complexity"] = "cluttered"
                elif significant > 2:
                    scene_data["background_complexity"] = "moderate"
                else:
                    scene_data["background_complexity"] = "clean"

            time.sleep(frame_interval)
    finally:
        cap.release()

    if frames_processed == 0 or scene_data["face_count"] == 0:
        return None

    if clothing_votes:
        scene_data["dominant_clothing_color"] = _resolve_stable_clothing_color(clothing_votes)

    best = _resolve_stable_emotion(emotion_votes)

    return _build_scene_tags(best, scene_data)
