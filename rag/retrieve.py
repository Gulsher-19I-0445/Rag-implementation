from dataclasses import dataclass

from core.config import config
from ingest.store import Store


@dataclass
class Retrieval:
    hits: list[dict]
    top_score: float
    grounded: bool

    def as_context(self) -> str:
        """Numbered blocks so the model can cite [1], [2]."""
        return "\n\n".join(
            f"[{i}] ({h['doc_title']}, {h['locator']})\n{h['text']}"
            for i, h in enumerate(self.hits, 1)
        )


def retrieve(store: Store, query: str, k: int | None = None) -> Retrieval:
    hits = store.search(query, k=k or config.TOP_K)
    if not hits:
        return Retrieval([], 0.0, False)

    top = hits[0]["score"]
    if top < config.TAU:
        return Retrieval(hits, top, False)

    keep = [h for h in hits if h["score"] >= max(config.TAU - 0.10, top - 0.12)]
    return Retrieval(keep, top, True)