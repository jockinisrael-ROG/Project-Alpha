"""Enhanced vision module using multiple AI models for comprehensive scene understanding."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

from process.logger import setup_logger

logger = setup_logger(__name__)

# Optional imports - graceful degradation if not installed
_CLIP_AVAILABLE = False
_LLAVA_AVAILABLE = False
_BLIP_AVAILABLE = False
_YOLO_AVAILABLE = False

try:
    import torch
    from PIL import Image
    import clip
    _CLIP_AVAILABLE = True
    logger.debug("CLIP available")
except ImportError:
    logger.warning("CLIP not installed. Install with: pip install clip")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_layer_weights, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from transformers import TextStreamer
    _LLAVA_AVAILABLE = True
    logger.debug("LLaVA available")
except ImportError:
    logger.warning("LLaVA not installed. Install with: pip install llava-rlhf")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _BLIP_AVAILABLE = True
    logger.debug("BLIP available")
except ImportError:
    logger.warning("BLIP not installed. Install with: pip install transformers pillow")

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
    logger.debug("YOLO available")
except ImportError:
    logger.warning("YOLO not installed. Install with: pip install ultralytics")


# Model caches to avoid reloading
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_LLAVA_MODEL = None
_LLAVA_PROCESSOR = None
_BLIP_PROCESSOR = None
_BLIP_MODEL_INST = None
_YOLO_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu" if _CLIP_AVAILABLE else None


def _load_clip() -> tuple[Any, Any] | tuple[None, None]:
    """Load CLIP model and preprocessing."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    
    if not _CLIP_AVAILABLE:
        return None, None
    
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    
    try:
        logger.debug("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=_DEVICE)
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
        return model, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        return None, None


def _load_blip() -> tuple[Any, Any] | tuple[None, None]:
    """Load BLIP model for image captioning."""
    global _BLIP_PROCESSOR, _BLIP_MODEL_INST
    
    if not _BLIP_AVAILABLE:
        return None, None
    
    if _BLIP_PROCESSOR is not None:
        return _BLIP_PROCESSOR, _BLIP_MODEL_INST
    
    try:
        logger.debug("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_PROCESSOR = processor
        _BLIP_MODEL_INST = model
        return processor, model
    except Exception as e:
        logger.error(f"Failed to load BLIP: {e}")
        return None, None


def _load_yolo() -> Any | None:
    """Load YOLO model for object detection."""
    global _YOLO_MODEL
    
    if not _YOLO_AVAILABLE:
        return None
    
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    
    try:
        logger.info("Loading YOLO model...")
        model = YOLO("yolov8m.pt")  # Medium size model
        _YOLO_MODEL = model
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None


def _load_llava() -> tuple[Any, Any] | tuple[None, None]:
    """Load LLaVA model for visual question answering."""
    global _LLAVA_MODEL, _LLAVA_PROCESSOR
    
    if not _LLAVA_AVAILABLE:
        return None, None
    
    if _LLAVA_MODEL is not None:
        return _LLAVA_MODEL, _LLAVA_PROCESSOR
    
    try:
        logger.info("Loading LLaVA model...")
        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_layer_weights(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )
        _LLAVA_MODEL = (tokenizer, model, image_processor, context_len)
        _LLAVA_PROCESSOR = image_processor
        return _LLAVA_MODEL, _LLAVA_PROCESSOR
    except Exception as e:
        logger.error(f"Failed to load LLaVA: {e}")
        return None, None


def extract_vision_context(
    frame: np.ndarray,
    camera_index: int = 0,
    sample_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    Extract comprehensive visual context from a frame using multiple AI models.
    
    Args:
        frame: Input image frame (BGR from OpenCV)
        camera_index: Camera device index
        sample_seconds: Sample duration in seconds
        
    Returns:
        Dictionary with vision context including captions, objects, scene understanding
    """
    if frame is None or frame.size == 0:
        logger.warning("Empty frame provided")
        return {"available": False, "error": "Empty frame"}
    
    context = {
        "available": True,
        "timestamp": time.time(),
        "frame_shape": frame.shape,
        "models_used": [],
    }
    
    # Convert BGR to RGB for model processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # BLIP Image Captioning
    try:
        processor, model = _load_blip()
        if processor is not None and model is not None:
            logger.debug("Running BLIP image captioning...")
            inputs = processor(pil_image, return_tensors="pt")
            out = model.generate(**inputs, max_length=128)
            caption = processor.decode(out[0], skip_special_tokens=True)
            context["caption"] = caption
            context["models_used"].append("BLIP")
            logger.info(f"BLIP caption: {caption}")
    except Exception as e:
        logger.error(f"BLIP captioning error: {e}")
    
    # YOLO Object Detection
    try:
        yolo_model = _load_yolo()
        if yolo_model is not None:
            logger.debug("Running YOLO object detection...")
            results = yolo_model(frame, verbose=False)
            detected_objects = []
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    confidence = float(box.conf[0])
                    if confidence > 0.5:
                        detected_objects.append({
                            "name": class_name,
                            "confidence": round(confidence, 3)
                        })
            
            if detected_objects:
                context["detected_objects"] = detected_objects
                context["models_used"].append("YOLO")
                logger.info(f"Detected objects: {[o['name'] for o in detected_objects]}")
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
    
    # CLIP Image-Text Matching
    try:
        clip_model, preprocess = _load_clip()
        if clip_model is not None:
            logger.debug("Running CLIP analysis...")
            
            # List of descriptive prompts for scene understanding
            prompts = [
                "professional workspace",
                "casual home environment",
                "outdoor setting",
                "bright lighting",
                "dim lighting",
                "cluttered space",
                "organized space",
                "person visible",
                "person not visible",
                "multiple people",
                "single person",
                "modern decor",
                "minimal decor",
            ]
            
            image_tensor = preprocess(pil_image).unsqueeze(0).to(_DEVICE)
            text_inputs = clip.tokenize(prompts).to(_DEVICE)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                text_features = clip_model.encode_text(text_inputs)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0)
            
            # Get top matching concepts
            top_indices = similarity.argsort(descending=True)[:5]
            clip_concepts = [
                f"{prompts[i]}: {similarity[i]:.3f}"
                for i in top_indices
            ]
            context["clip_concepts"] = clip_concepts
            context["models_used"].append("CLIP")
            logger.info(f"CLIP concepts: {clip_concepts}")
            
    except Exception as e:
        logger.error(f"CLIP analysis error: {e}")
    
    # LLaVA Visual Question Answering
    try:
        llava_model, image_processor = _load_llava()
        if llava_model is not None:
            logger.debug("Running LLaVA visual QA...")
            tokenizer, model, processor, context_len = llava_model
            
            # Ask specific questions about the image
            questions = [
                "What is the main activity or action happening in this image?",
                "Describe the lighting conditions and overall atmosphere.",
                "What is the primary mood or feeling conveyed by this scene?",
            ]
            
            qa_results = []
            for question in questions:
                try:
                    image_tensor = image_processor.preprocess(pil_image, return_tensors='pt')['pixel_values'].to(model.device)
                    
                    prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).to(model.device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=True,
                            temperature=0.2,
                            max_new_tokens=256,
                            use_cache=True,
                        )
                    
                    answer = tokenizer.decode(output_ids[0]).strip()
                    qa_results.append({"question": question, "answer": answer})
                    
                except Exception as e:
                    logger.warning(f"LLaVA QA error: {e}")
            
            if qa_results:
                context["llava_qa"] = qa_results
                context["models_used"].append("LLaVA")
                logger.info(f"LLaVA QA completed: {len(qa_results)} answers")
                
    except Exception as e:
        logger.error(f"LLaVA QA error: {e}")
    
    logger.info(f"Vision context extraction complete. Models used: {context['models_used']}")
    return context


def capture_and_analyze(
    camera_index: int = 0,
    sample_seconds: float = 1.0,
) -> Dict[str, Any]:
    """
    Capture frame from camera and extract comprehensive vision context.
    
    Args:
        camera_index: Camera device index
        sample_seconds: Sample duration (used for consistency with existing code)
        
    Returns:
        Vision context dictionary with analysis results
    """
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Camera {camera_index} not available")
            return {"available": False, "error": f"Camera {camera_index} not available"}
        
        # Read a few frames to let camera stabilize
        for _ in range(3):
            cap.read()
        
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to capture frame")
            return {"available": False, "error": "Failed to capture frame"}
        
        return extract_vision_context(frame, camera_index, sample_seconds)
        
    except Exception as e:
        logger.error(f"Camera capture error: {e}")
        return {"available": False, "error": str(e)}
    finally:
        if cap is not None:
            cap.release()


def format_vision_for_llm(vision_context: Dict[str, Any]) -> str:
    """
    Format vision context into a prompt-ready string for the LLM.
    
    Args:
        vision_context: Vision context dictionary from extract_vision_context()
        
    Returns:
        Formatted string ready to include in LLM prompts
    """
    if not vision_context.get("available"):
        return "[Vision unavailable]"
    
    parts = ["[Visual Analysis]"]
    
    if "caption" in vision_context:
        parts.append(f"Scene: {vision_context['caption']}")
    
    if "detected_objects" in vision_context:
        objects = ", ".join([f"{obj['name']} ({obj['confidence']:.0%})" for obj in vision_context['detected_objects']])
        parts.append(f"Detected: {objects}")
    
    if "clip_concepts" in vision_context:
        concepts = "; ".join(vision_context["clip_concepts"])
        parts.append(f"Scene understanding: {concepts}")
    
    if "llava_qa" in vision_context:
        qa_text = "; ".join([f"Q: {qa['question'][:40]}... A: {qa['answer'][:60]}..." for qa in vision_context['llava_qa']])
        parts.append(f"Scene insights: {qa_text}")
    
    return "\n".join(parts)
