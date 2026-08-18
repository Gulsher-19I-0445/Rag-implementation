# ingest/sync.py
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.config import config
from ingest import manifest
from ingest.chunker import chunk_blocks
from ingest.parser import parse
from ingest.store import Store


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    duration_s: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.deleted or self.modified)

    def summary(self) -> str:
        if not self.changed and not self.failed:
            return "up to date"
        bits = []
        if self.added:
            bits.append(f"+{len(self.added)} added")
        if self.modified:
            bits.append(f"~{len(self.modified)} modified")
        if self.deleted:
            bits.append(f"-{len(self.deleted)} deleted")
        if self.failed:
            bits.append(f"!{len(self.failed)} failed")
        return f"{', '.join(bits)} ({self.chunks_added} chunks, {self.duration_s:.1f}s)"


def _ingest_one(store: Store, rel: str) -> int:
    """Parse → chunk → upsert one file. Returns chunk count. Raises on failure."""
    path = config.DOCSTORE_DIR / rel
    chunks = chunk_blocks(parse(path, rel))
    store.upsert(chunks)
    return len(chunks)


def reconcile(store: Store) -> SyncResult:
    started = time.perf_counter()
    result = SyncResult()

    config.DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)

    old = manifest.load()
    new = manifest.scan(config.DOCSTORE_DIR)
    added, deleted, modified = manifest.diff(old, new)

    entries = dict(old)

    for rel in sorted(deleted):
        store.delete_by_source(rel)
        entries.pop(rel, None)
        result.deleted.append(rel)

    for rel in sorted(modified):
        store.delete_by_source(rel)

    for rel in sorted(added | modified):
        try:
            n = _ingest_one(store, rel)
        except Exception as e:
            result.failed.append((rel, str(e)))
            entries.pop(rel, None)
            continue
        entries[rel] = manifest.entry(new[rel], n)
        result.chunks_added += n
        (result.added if rel in added else result.modified).append(rel)

    manifest.save(entries)
    result.duration_s = time.perf_counter() - started
    return result


def audit(store: Store) -> list[str]:
    """Manifest vs store drift. Empty list means consistent."""
    entries = manifest.load()
    expected = {rel for rel, e in entries.items() if e.get("chunks", 0) > 0}
    actual = store.sources()

    problems = []
    for rel in sorted(expected - actual):
        problems.append(f"in manifest but not indexed: {rel}")
    for rel in sorted(actual - expected):
        problems.append(f"indexed but not in manifest: {rel}")
    return problems


if __name__ == "__main__":
    store = Store()
    r = reconcile(store)
    print(r.summary())
    for rel, err in r.failed:
        print(f"  FAILED {rel}: {err}")
    for p in audit(store):
        print(f"  DRIFT {p}")
    print(f"store: {store.count()} chunks across {len(store.sources())} sources")