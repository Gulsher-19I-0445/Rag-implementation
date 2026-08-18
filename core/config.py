import os

from pathlib import Path

from streamlit import config

class Config:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    ROOT = Path(__file__).resolve().parents[1]
    DOCS_DIR = (ROOT / "data" / "docs").resolve()
    DOCSTORE_DIR = (ROOT / "data" / "docs").resolve()
    TAU=""
    CHUNK_SIZE    = 700
    CHUNK_OVERLAP = 100
    MERGE_FLOOR   = 250
    MIN_CHUNK     = 100
    TOP_K=5
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    CHROMA_DIR  = (ROOT / "data" / "chroma").resolve()
    TOP_K       = 3
    TAU         = 0.6
    MANIFEST_PATH = (ROOT / "data" / ".manifest.json").resolve()
    SCORE_LO = 0.50
    SCORE_HI = 0.85
    JUDGE_MODEL = "openai/gpt-oss-20b"
    PROVIDERS= "openai/gpt-oss-20b"

def rel_key(path: Path) -> str:
    return path.resolve().relative_to(config.DOCSTORE_DIR).as_posix()


config = Config()
