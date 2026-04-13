import sounddevice as sd
import numpy as np
import tempfile
import os
import sqlite3
from datetime import datetime
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import requests

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# =============================
# CONFIG
# =============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

MODEL = "openai/gpt-4o-mini"

samplerate = 16000
duration = 5  # seconds
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "alpha_memory.db")


# =============================
# MEMORY DATABASE
# =============================
def get_db_connection():
    return sqlite3.connect(MEMORY_DB_PATH)


def init_memory_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            label TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            user_text TEXT NOT NULL,
            assistant_text TEXT NOT NULL,
            audio_path TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS assistant_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_interactions_session_created
        ON interactions(session_id, created_at)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_assistant_memory_type_created
        ON assistant_memory(memory_type, created_at)
        """
    )

    conn.commit()
    conn.close()


def start_session(label="voice_session"):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (started_at, label) VALUES (?, ?)",
        (now, label),
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def end_session(session_id):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()


def save_interaction(session_id, user_text, assistant_text, audio_path=None):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interactions (session_id, created_at, user_text, assistant_text, audio_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, now, user_text, assistant_text, audio_path),
    )
    conn.commit()
    conn.close()


def upsert_user_memory(key, value):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_memory (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, value, now),
    )
    conn.commit()
    conn.close()


def save_assistant_memory(memory_type, content):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO assistant_memory (memory_type, content, created_at) VALUES (?, ?, ?)",
        (memory_type, content, now),
    )
    conn.commit()
    conn.close()

# =============================
# PERSONALITY (CUTE TSUNDERE)
# =============================
SYSTEM_PROMPT = """
You are a cute, playful, slightly tsundere AI assistant.

Personality rules:
- You act a little annoyed or sarcastic sometimes, but you actually care.
- You may tease the user lightly.
- You are expressive (use small emotions like "hmph", "ugh", "fine...", "whatever").
- Keep responses natural and not overly long.
- Do NOT be rude or offensive.
- Always give correct and helpful answers.

You are currently in ALPHA stage (not fully developed like Yuki).
"""

# =============================
# LOAD MODELS
# =============================
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
tts_engine = pyttsx3.init() if pyttsx3 is not None else None

# =============================
# RECORD AUDIO
# =============================
def record_audio():
    print("\n🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype='int16')
    sd.wait()
    return audio

# =============================
# SAVE AUDIO
# =============================
def save_audio(audio):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_file.name, samplerate, audio)
    return temp_file.name

# =============================
# SPEECH TO TEXT
# =============================
def speech_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ""
    for segment in segments:
        text += segment.text

    text = text.strip()
    print(f"🧠 You said: {text}")
    return text

# =============================
# OPENROUTER CALL
# =============================
def get_ai_response(user_text):
    if not OPENROUTER_API_KEY:
        return "OpenRouter key is missing. Set OPENROUTER_API_KEY in your environment."

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Alpha Assistant"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ API Error:", response.text)
        return "Ugh… something broke. Fix it."

    print(f"🤖 Alpha: {reply}")
    return reply

# =============================
# TEXT TO SPEECH
# =============================
def speak(text):
    if tts_engine is None:
        return
    tts_engine.say(text)
    tts_engine.runAndWait()

# =============================
# MAIN LOOP
# =============================
def main():
    init_memory_db()
    session_id = start_session("alpha_tsundere_voice")

    print("🔥 Alpha Assistant Started (Tsundere Mode)")
    print("Press CTRL+C to stop.\n")

    try:
        while True:
            audio = record_audio()
            audio_path = save_audio(audio)

            user_text = speech_to_text(audio_path)

            if not user_text:
                print("😶 No speech detected...")
                continue

            reply = get_ai_response(user_text)
            save_interaction(session_id, user_text, reply, audio_path)
            speak(reply)
    finally:
        end_session(session_id)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    main()