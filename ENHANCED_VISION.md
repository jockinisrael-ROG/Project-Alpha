# Enhanced Vision AI Integration

The project has been upgraded to support advanced computer vision models for comprehensive scene understanding and context-aware AI responses.

## Features

### Multiple AI Vision Models

- **CLIP** (Contrastive Language-Image Pre-training)
  - Image-text matching and scene understanding
  - Identifies mood, lighting, environment type, complexity
  
- **BLIP** (Bootstrapping Language-Image Pre-training)
  - Natural language image captioning
  - Generates descriptive text about the scene
  
- **YOLO** (You Only Look Once)
  - Real-time object detection
  - Identifies and counts objects in the scene
  
- **LLaVA** (Large Language and Vision Assistant)
  - Visual question answering
  - Answers specific questions about image content
  - Requires significant VRAM (8GB+)

## Usage

### Enabling Enhanced Vision

1. **Set in config/config.yaml:**
   ```yaml
   vision:
     enabled: true
     enhanced_models: true
     camera_index: 0
     sample_seconds: 0.15
   ```

2. **Install optional dependencies:**
   ```bash
   pip install -r vision-enhanced-requirements.txt
   ```

3. **Run the assistant:**
   ```bash
   python main_chat.py
   ```

### How It Works

When enabled, the enhanced vision system:

1. **Captures** a frame from your camera
2. **Analyzes** using multiple AI models simultaneously:
   - BLIP generates a scene caption
   - YOLO detects objects and counts them
   - CLIP analyzes scene semantics
   - LLaVA answers specific questions
3. **Formats** structured visual context
4. **Sends** to the LLM for context-aware responses

### Example Output

```
[Vision] Scene analyzed with: BLIP, YOLO, CLIP, LLaVA
[Vision] Scene: A person sitting at a computer desk in an office environment
[User] How do I look today?
[Alpha] *blushes slightly* You look great! I can see you're at your desk in a professional setup. The lighting looks good today too! How's your day going so far?
```

## Installation Guide

### Basic Vision (Emotion Detection + Color Calibration)
Already included, no additional installation needed.

### Enhanced Vision Models

#### 1. CLIP + BLIP (Lightweight, Recommended)
```bash
pip install torch torchvision clip transformers pillow
```

#### 2. YOLO Object Detection
```bash
pip install ultralytics
```

#### 3. LLaVA (Advanced, High VRAM Required)
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```

Or use all-in-one:
```bash
pip install -r vision-enhanced-requirements.txt
```

## Performance Notes

- **CLIP + BLIP**: ~2-3 seconds per frame (GPU), slower on CPU
- **YOLO**: ~0.5-1 second per frame
- **LLaVA**: ~5-10 seconds per frame (requires 8GB+ VRAM)

### GPU Acceleration

For significant speedup, ensure you have:
- NVIDIA GPU with CUDA capability
- PyTorch with CUDA support installed
- CUDA toolkit 11.8+ (recommended)

### CPU-Only Mode

All models work on CPU but will be slower. Models gracefully degrade if not installed.

## Architecture

- **process/enhanced_vision.py**: Integration of all AI models
  - `extract_vision_context()`: Extracts comprehensive visual understanding
  - `capture_and_analyze()`: Gets camera frame and analyzes it
  - `format_vision_for_llm()`: Formats results for LLM consumption

- **process/llm.py**: Updated to accept vision context
  - `_build_user_message()`: Integrates vision data with user input
  - `get_response()` / `get_response_stream()`: Pass vision context to LLM

- **main_chat.py**: Orchestrates vision analysis in conversation loop

## Graceful Degradation

If any model fails to load or isn't installed:
- That specific model is skipped
- Other models continue to work
- System logs warnings for missing modules
- No crash - continues with available capabilities

## Configuration

In `config/config.yaml`:

```yaml
vision:
  enabled: true              # Enable vision entirely
  enhanced_models: true      # Use advanced AI models (default: false)
  camera_index: 0            # Webcam device (0 = default)
  sample_seconds: 0.15       # How long to sample video
  auto_calibration: true     # Auto color calibration on startup
  calibration_refresh_minutes: 120  # Auto-recalibrate every 2 hours
  calibration_sample_seconds: 1.8   # Sampling time for calibration
```

## Troubleshooting

### Models not found
```
pip install -r vision-enhanced-requirements.txt
```

### CUDA out of memory
- Reduce model sizes (e.g., use ViT-B/32 instead of larger models)
- Run on CPU only
- Close other GPU-using processes

### Slow inference
- Expected on CPU (2-10 seconds per frame)
- GPU reduces to <1 second per frame
- Cache models in memory to avoid reloading

### Camera not working
- Check camera is not in use by other applications
- Verify correct camera_index in config
- Test with: `python -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); print('OK' if ret else 'FAIL'); cap.release()"`

## API Reference

### extract_vision_context(frame, camera_index=0, sample_seconds=1.0)
Extract comprehensive visual context from an image frame.

**Returns:** Dictionary with keys:
- `available`: bool - Whether analysis succeeded
- `caption`: str - BLIP image caption
- `detected_objects`: list - YOLO detections
- `clip_concepts`: list - CLIP scene understanding
- `llava_qa`: list - LLaVA question-answer pairs
- `models_used`: list - Which models were used

### capture_and_analyze(camera_index=0, sample_seconds=1.0)
Capture frame from camera and analyze it.

### format_vision_for_llm(vision_context)
Format vision results into LLM-friendly text.
