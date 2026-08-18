# ingest/store.py
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from core.config import config
from ingest.chunker import Chunk

_embedder: TextEmbedding | None = None


def embedder() -> TextEmbedding:
    """Lazy singleton — model load is ~1s and must not repeat per call."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=config.EMBED_MODEL)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """fastembed yields numpy arrays; Chroma needs plain float lists."""
    return [v.tolist() for v in embedder().embed(texts)]


class Store:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},   # immutable after creation
        )


    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embed([c.text for c in chunks]),   # header INCLUDED
            documents=[c.raw_text for c in chunks],       # header EXCLUDED
            metadatas=[{
                "source": c.source,
                "locator": c.locator,
                "doc_title": c.doc_title,
            } for c in chunks],
        )
        return len(chunks)

    def delete_by_source(self, rel: str) -> None:
        self._col.delete(where={"source": rel})


    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Returns [{text, source, locator, doc_title, score}]; score is SIMILARITY."""
        k = k or config.TOP_K
        if self._col.count() == 0:
            return []

        res = self._col.query(
            query_embeddings=embed([query]),
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        return [
            {
                "text": doc,
                "source": meta.get("source", ""),
                "locator": meta.get("locator", ""),
                "doc_title": meta.get("doc_title", ""),
                "score": 1.0 - dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._col.count()

    def sources(self) -> set[str]:
        got = self._col.get(include=["metadatas"])
        return {m.get("source", "") for m in got["metadatas"] if m}

    def reset(self) -> None:
        """Drop everything. Only for testing — never call from sync."""
        self._client.delete_collection("docs")
        self._col = self._client.get_or_create_collection(
            name="docs", metadata={"hnsw:space": "cosine"},
        )

if __name__ == "__main__":
    from pathlib import Path
    from core.config import rel_key
    from ingest.chunker import chunk_blocks
    from ingest.parser import parse

    store = Store()
    p = config.DOCSTORE_DIR / "example2.docx"
    rel = rel_key(p)

    n = store.upsert(chunk_blocks(parse(p, rel)))
    print(f"upserted {n}, count={store.count()}, sources={store.sources()}")

    for hit in store.search("what is the total PO amount"):
        print(f"  {hit['score']:.3f} [{hit['locator']}] {hit['text'][:70]}...")

    before = store.count()
    store.upsert(chunk_blocks(parse(p, rel)))
    print(f"re-upsert: {before} -> {store.count()} (must be equal)")

    # store.delete_by_source(rel)
    # print(f"after delete: count={store.count()} (must be 0)")