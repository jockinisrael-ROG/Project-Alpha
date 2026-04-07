# Project Alpha

Project Alpha is a local AI assistant with voice, vision, automation, and FastAPI endpoints for chat and realtime interaction.

## Features

- Voice pipeline with STT, LLM, and TTS
- FastAPI server for chat, streaming, vision, and automation endpoints
- Optional enhanced vision models
- Local config-driven setup with logging and memory support

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- An OpenRouter API key if you want LLM responses

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Set `OPENROUTER_API_KEY` in your environment if you want LLM access.
4. Adjust `config/config.yaml` if you need to change voice, STT, or vision settings.

## Run

Start the FastAPI server with:

```bash
uvicorn server:app --reload
```

Or run the standalone launcher scripts provided in the repo if you prefer the desktop workflow.

## Security

- Do not commit API keys or other secrets into `config/config.yaml`.
- Use environment variables for credentials.
- If a secret was committed previously, rotate it with the provider and remove it from git history if needed.

## Project Layout

- `server.py`: FastAPI backend entry point
- `main.py` and `main_chat.py`: assistant runtime entry points
- `process/`: STT, TTS, LLM, vision, automation, and logging modules
- `utils/`: configuration and memory helpers
- `config/`: runtime configuration files