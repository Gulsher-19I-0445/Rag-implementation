# core/chunker.py
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import config
from ingest.parser import Block
from docx.text.paragraph import Paragraph

@dataclass
class Chunk:
    id: str
    text: str
    raw_text: str
    source: str
    locator: str
    doc_title: str


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " | ", " ", ""],   # " | " keeps table rows intact
    length_function=len,
)



def _header(block: Block) -> str:
    """Provenance prepended to embedded text so section-level queries can match."""
    parts = [p for p in (block.doc_title, block.locator) if p]
    return f"[{' — '.join(parts)}]\n" if parts else ""


def _merge_small(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    buf: list[Block] = []

    def flush():
        if not buf:
            return
        head = buf[0]
        out.append(Block(
            text="\n".join(b.text for b in buf),
            source=head.source,
            locator=head.locator,
            doc_title=head.doc_title,
        ))
        buf.clear()

    for b in blocks:
        same_group = bool(buf) and (b.source, b.locator) == (buf[0].source, buf[0].locator)
        below_floor = sum(len(x.text) for x in buf) < config.MERGE_FLOOR

        if buf and not (same_group and below_floor):
            flush()
        buf.append(b)

    flush()
    return out


def _split_large(blocks: list[Block]) -> list[Block]:
    """
    Pass 2: split blocks over CHUNK_SIZE, carrying source/locator/title onto
    every piece. Returns Blocks still — IDs are assigned later, once.
    """
    # TODO:
    split_blocks=[]
    for b in blocks:
        if len(b.text)<=config.CHUNK_SIZE:
            split_blocks.append(b)
        else:
            text_chunks = _splitter.split_text(b.text)
            for n, piece in enumerate(text_chunks,1):
                loc = f"{b.locator} ({n}/{len(text_chunks)})" if b.locator else None
                split_blocks.append(Block(text=piece, source=b.source, locator=loc, doc_title=b.doc_title))

    return split_blocks
    #   for each block: if len <= CHUNK_SIZE, pass through unchanged
    #   else _splitter.split_text(block.text) → one Block per piece
    ...


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    """Blocks from one file → Chunks ready for the vector store."""
    processed = _split_large(_merge_small(blocks))

    chunks: list[Chunk] = []
    for i, b in enumerate(processed):          # index runs across the whole FILE
        raw = b.text.strip()
        if len(raw) < config.MIN_CHUNK:
            continue
        chunks.append(Chunk(
            id=f"{b.source}::{i}",
            text=_header(b) + raw,
            raw_text=raw,
            source=b.source,
            locator=b.locator or "",
            doc_title=b.doc_title or "",
        ))
    return chunks


if __name__ == "__main__":
    from pathlib import Path
    from core.config import rel_key
    from parser import parse

    for name in ("example.pdf", "example.docx"):
        p = config.DOCSTORE_DIR / name
        cs = chunk_blocks(parse(p, rel_key(p)))
        lens = sorted(len(c.raw_text) for c in cs)
        print(f"\n{name}: {len(cs)} chunks | min={lens[0]} max={lens[-1]} "
              f"median={lens[len(lens)//2]}")
        for c in cs[:2]:
            print(f"  {c.id} [{c.locator}] {c.raw_text[:80]}...")