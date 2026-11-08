"""Paths and settings. RE_AGENT_RUNTIME lets tests and the demo use isolated storage."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_FILE = ROOT / "data" / "properties.json"
KNOWLEDGE_DIR = ROOT / "knowledge"
MODEL = os.getenv("RE_AGENT_MODEL", "claude-opus-5-5")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def require_api_key() -> None:
    """Exit with setup instructions instead of a raw SDK error."""
    from .llm import provider

    if provider() is None:
        raise SystemExit(
            "No API key found. Copy .env.example to .env and set ANTHROPIC_API_KEY or GROQ_API_KEY."
        )


def runtime_dir() -> Path:
    path = Path(os.getenv("RE_AGENT_RUNTIME", ROOT / "runtime"))
    path.mkdir(parents=True, exist_ok=True)
    return path
