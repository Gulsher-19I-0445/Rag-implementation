# ingest/manifest.py
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import config, rel_key

SKIP_PREFIXES = ("~$", ".")
EXTENSIONS = {".pdf", ".docx"}


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def scan(root: Path) -> dict[str, str]:
    """Current disk state: {rel_key: sha256}."""
    out = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTENSIONS:
            continue
        if p.name.startswith(SKIP_PREFIXES):
            continue
        out[rel_key(p)] = file_hash(p)
    return out


def load() -> dict[str, dict]:
    """Stored manifest: {rel_key: {hash, chunks, indexed_at}}."""
    if not config.MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: manifest unreadable ({e}) — treating as empty")
        return {}


def save(entries: dict[str, dict]) -> None:
    """Atomic write — a crash mid-save must not corrupt the manifest."""
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(config.MANIFEST_PATH)


def entry(hash_: str, chunks: int) -> dict:
    return {
        "hash": hash_,
        "chunks": chunks,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def diff(old: dict[str, dict], new: dict[str, str]):
    """old = stored manifest, new = scan() output. Returns (added, deleted, modified)."""
    added = new.keys() - old.keys()
    deleted = old.keys() - new.keys()
    modified = {
        k for k in old.keys() & new.keys()
        if old[k].get("hash") != new[k]
    }
    return added, deleted, modified