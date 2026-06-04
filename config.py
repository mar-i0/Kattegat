import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

MEDIUMS_DIR = BASE_DIR / "Mediums"
INVERSION_DIR = BASE_DIR / "Inversión"

STYLE_INDEX_DIR = BASE_DIR / "style_index"
SYSTEM_PROMPT_FILE = BASE_DIR / "system_prompt.txt"

CLOUD_API_URL = "https://ollama.com/api/chat"
CLOUD_MODEL = "gemma3:12b"
LOCAL_API_URL = "http://127.0.0.1:11434/api/generate"
LOCAL_MODEL = "qwen2.5:1.5b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_STYLE = 5

def load_api_key():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    key = os.getenv("OLLAMA_API_KEY")
    if not key:
        key_path = BASE_DIR / "key.txt"
        if key_path.exists():
            key = key_path.read_text().strip()
    return key
