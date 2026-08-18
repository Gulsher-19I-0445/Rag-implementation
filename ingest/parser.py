# core/parsers.py
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass
class Block:
    text: str
    source: str
    locator: str | None = None
    doc_title: str | None = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError(f"Block.text got {type(self.text).__name__}: {self.text!r}")


NEW_BLOCK = re.compile(
    r"^\s*("
    r"[●•▪◦\-–—*]"
    r"|\d+[\.\)]\s"
    r"|[A-Z][A-Za-z ]{0,30}:"
    r"|#{1,6}\s"
    r")"
)
SENTENCE_END = re.compile(r"[.!?:;]\s*$")

NOISE = re.compile(r"^\s*(?:None\s*)+$", re.MULTILINE)
def _key(line: str) -> str:
    """Normalized form used for BOTH counting and removing boilerplate."""
    return re.sub(r"\d+", "#", line.strip()).lower()


def join_wrapped_lines(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if out and out[-1] and SENTENCE_END.search(out[-1]):
                out.append("")          # real paragraph break — keep it
            continue 
    # for line in text.splitlines():
    #     stripped = line.strip()
    #     if not stripped:
    #         out.append("")
    #         continue
        if (out and out[-1]
                and not SENTENCE_END.search(out[-1])
                and not NEW_BLOCK.match(stripped)):
            out[-1] += " " + stripped
        else:
            out.append(stripped)
    return "\n".join(out)


def strip_boilerplate(text: str, boilerplate: set[str], window: int = 2) -> str:
    if not boilerplate:
        return text
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if len(idx) < 4:
        return text
    edges = set(idx[:window]) | set(idx[-window:])
    kept = [l for i, l in enumerate(lines)
            if not (i in edges and _key(l) in boilerplate)]
    return "\n".join(kept)


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)   
    text = join_wrapped_lines(text)                  
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_title(reader: PdfReader, rel: str) -> str:
    title = None
    if reader.metadata:                               # can be None
        title = (reader.metadata.title or "").strip() or None
    if title and (".doc" in title.lower() or title.lower().startswith("untitled")):
        title = None
    return title or Path(rel).stem


def parse_pdf(path: Path, rel: str) -> list[Block]:
    reader = PdfReader(path)
    title = _pdf_title(reader, rel)

    raw: list[str] = []
    for page in reader.pages:
        try:
            raw.append(NOISE.sub("", page.extract_text() or ""))
        except Exception as e:
            print(f"WARN: {rel} page extract failed: {e}")
            raw.append("")

    empty = sum(1 for t in raw if len(t.strip()) < 20)
    if raw and empty > len(raw) / 2:
        print(f"WARN: {rel} looks scanned ({empty}/{len(raw)} pages empty) — skipped")
        return []

    # NOISE = re.compile(r"^\s*(?:None\s*)+$", re.MULTILINE)
    # raw.append(NOISE.sub("", page.extract_text() or ""))

    boilerplate: set[str] = set()
    if len(raw) >= 5:
        counts = Counter()
        for t in raw:
            lines = [l.strip() for l in t.splitlines() if l.strip()]
            counts.update({_key(l) for l in lines[:2] + lines[-2:]})
        threshold = max(3, int(0.7 * len(raw)))
        boilerplate = {k for k, n in counts.items() if n >= threshold}

    blocks: list[Block] = []
    for i, t in enumerate(raw):
        t = clean_text(strip_boilerplate(t, boilerplate))
        if len(t) < 30:
            continue
        blocks.append(Block(text=t, source=rel, locator=f"page {i + 1}", doc_title=title))
    return blocks


from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(doc):
    """Yield Paragraph and Table objects in true document order."""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _serialize_table(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for c in row.cells:
            t = c.text.strip().replace("\n", " ")
            if cells and cells[-1] == t:      # merged cells repeat
                continue
            cells.append(t)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_heading(para: Paragraph) -> bool:
    name = para.style.name or ""
    if name.startswith("Heading") or name == "Title":
        return True
    text = para.text.strip()
    if not text or len(text) > 60:
        return False
    runs = [r for r in para.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def parse_docx(path: Path, rel: str) -> list[Block]:
    doc = Document(path)
    title = (doc.core_properties.title or "").strip() or Path(rel).stem

    blocks: list[Block] = []
    buf: list[str] = []
    heading: str | None = None

    def flush():
        if not buf:
            return
        text = clean_text("\n".join(buf))
        buf.clear()
        if len(text) < 30:
            return
        blocks.append(Block(
            text=text,
            source=rel,
            locator=f"§ {heading}" if heading else f"section {len(blocks) + 1}",
            doc_title=title,
        ))

    for item in iter_block_items(doc):
        if isinstance(item, Table):
            t = _serialize_table(item)
            if t:
                buf.append(t)
            continue

        text = item.text.strip()
        if _is_heading(item):
            flush()
            heading = text or heading
            if text:
                buf.append(text)
        elif text:
            buf.append(text)

    flush()
    return blocks


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx}


def parse(path: Path, rel: str) -> list[Block]:
    fn = PARSERS.get(path.suffix.lower())
    if not fn:
        return []
    try:
        return fn(path, rel)
    except Exception as e:
        print(f"ERROR: failed to parse {rel}: {e}")
        return []


if __name__ == "__main__":
    from core.config import config, rel_key

    p = config.DOCSTORE_DIR / "example2.docx"
    blocks = parse(p, rel_key(p))
    Path("debug_blocks.txt").write_text(
        "\n\n===\n\n".join(f"[{b.locator}] {b.source}\n{b.text}" for b in blocks),
        encoding="utf-8",
    )
    print(f"{len(blocks)} blocks -> debug_blocks.txt")