# Grounded RAG over a local document folder

A retrieval-augmented question-answering app that reads a directory of `.pdf` and
`.docx` files, keeps its vector index in sync as those files change, and refuses to
answer anything it cannot ground in the retrieved text. Every response is scored on
three metrics shown alongside the answer.

Built without a RAG framework: parsing, chunking, lifecycle sync, and retrieval are
implemented directly against Chroma. LangChain is used only for its recursive text
splitter, and RAGAS only for the faithfulness metric.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
```

Drop `.pdf` / `.docx` files into `data/docs/`, then:

```bash
streamlit run app.py
```

The index is built on first launch. Subsequent launches only process files that
changed.

### Running fully offline

Generation sits behind an OpenAI-compatible provider layer, so switching to a local
model is a config change:

```bash
ollama pull qwen2.5:3b-instruct
LLM_PROVIDER=ollama streamlit run app.py
```

Embeddings, chunking, and the vector store are local in both modes — only the
generation and judging calls leave the machine when using Groq.

---

## Architecture

| Layer | Choice | Why |
| --- | --- | --- |
| Parsing | `pypdf`, `python-docx` | Default loaders drop DOCX tables entirely |
| Splitting | `RecursiveCharacterTextSplitter` | Solved problem; standalone package, no framework runtime |
| Embeddings | `fastembed` + `bge-small-en-v1.5` | 384-dim, ONNX, CPU-fast, no torch install |
| Vector store | Chroma (persistent, cosine) | `delete(where=...)` makes lifecycle sync one call |
| Generation | Groq / Ollama via `openai` SDK | Both expose OpenAI-compatible endpoints |
| UI | Streamlit | Per the brief |

### Ingestion

```
data/docs/ ──► parse ──► chunk ──► embed ──► Chroma
               Block     Chunk     vector    (id, vector, document, metadata)
```

`parse` produces `Block(text, source, locator, doc_title)` — one per PDF page, one
per DOCX heading section. `chunk` merges undersized blocks, splits oversized ones,
and prepends a provenance header.

Chunks carry two text fields. `text` (with the `[doc title — locator]` header) is
what gets embedded; `raw_text` (without it) is what gets stored and shown. The header
makes a mid-document chunk retrievable by document-level and section-level queries
without polluting the prompt or the source panel.

### Query

```
question ──► retrieve ──► [gate A: score ≥ τ] ──► generate ──► [gate B: answerable]
                                │                                      │
                                └──────────► refuse ◄──────────────────┘
                                                │
                                          evaluate ──► answer + report card
```

### Document lifecycle

A manifest at `data/.manifest.json` records a SHA-256 content hash per file.
Comparing a fresh directory scan against it yields three sets:

| Change | Action |
| --- | --- |
| Added | parse → chunk → embed → upsert |
| Deleted | `delete(where={"source": rel})` |
| Modified | delete, then re-ingest |

Modification is not a special case — delete-then-add is correct by construction and
cannot leave orphaned chunks from the previous version. Chunk IDs are
`{relative_path}::{index}`, so re-ingesting unchanged content overwrites rather than
duplicates.

Hashing rather than mtime is deliberate: mtime changes on checkout, copy, and cloud
sync without the content changing, and Word can rewrite a file byte-identically on
save.

The manifest is written once, after all store operations succeed. A file that fails
to parse has its entry removed rather than recorded, so the next run retries it
instead of treating it as permanently done.

**Trigger model:** reconciliation runs on app start and via the sidebar button. A
filesystem watcher was considered and left out — a watcher alone silently drifts
whenever the app isn't running, so startup reconciliation is the load-bearing half.
Adding `watchdog` on top would reduce latency, not add correctness.

---

## Grounding: two independent gates

**Gate A — similarity threshold.** If the top retrieved chunk scores below τ, the
app refuses without calling the LLM. Cheap, fast, and impossible to talk around.

**Gate B — model answerability.** The prompt requires JSON with an `answerable`
flag, numbered-source citations, and an explicit refusal exemplar. Temperature 0.

Both are necessary. Measured on the test corpus:

| Question | Top score | Outcome |
| --- | --- | --- |
| Out-of-domain ("who won the 2022 World Cup") | 0.535 | Gate A refused — no LLM call |
| Near-miss ("what is the vendor's late-delivery penalty") | **0.771** | Gate B refused |

The near-miss retrieved *more* confidently than several questions the app answered
correctly. Retrieval similarity measures topical proximity, not answerability, so no
threshold could have caught it.

### Calibrating τ

Measured against the corpus, five questions each side:

| | Range |
| --- | --- |
| In-domain top-1 | 0.703 – 0.796 |
| Out-of-domain top-1 | 0.468 – 0.535 |

τ = 0.60, roughly the midpoint, biased slightly toward refusing.

Worth noting that `bge-small` compresses similarity into a narrow band: every
out-of-domain query still scored ~0.5 despite sharing nothing with the corpus. A
threshold copied from a tutorial (0.3 is common) would have accepted all of them.
Thresholds must be calibrated per embedding model.

---

## Evaluation

Three metrics per response, shown as a report card.

| Metric | How | Cost |
| --- | --- | --- |
| **Grounding** | Top-1 retrieval similarity, rescaled from the measured 0.50–0.85 band to 0–1 | Free — already computed |
| **Relevance** | Cosine between question and answer embeddings | One embedding call |
| **Faithfulness** | RAGAS: decompose the answer into claims, verify each against the retrieved context | 2–20s, toggleable |

Faithfulness is skipped when the model refuses — a correct refusal has no claims to
support, and scoring it 0.00 would paint the app's best behaviour as its worst.

### Known limitations

**Faithfulness catches fabrication, not misattribution.** In one test the model
answered that a person worked at "AI-Augmented Test Engineering" — a phrase from a
CV's title line, not an employer. Faithfulness scored 1.00, because every token *was*
traceable to the retrieved context; the model had recombined real text into a false
statement. Claim-level verification checks provenance, not structural correctness.

**Self-judging skews optimistic.** The judge runs on the same model that produced the
answer. `JUDGE_MODEL` in `config.py` points the judge at a different model on the
same endpoint; the numbers below assume it isn't set.

**Relevance is sensitive to question length.** A three-word question against a
full-sentence answer scores lower than a well-formed question against the same
answer, independent of quality.

**Heading detection is heuristic.** DOCX files without semantic heading styles fall
back to a bold-and-short check. This degrades citation granularity, not answer
quality.

### RAGAS integration notes

RAGAS required three workarounds and is used for one metric as a result:

1. `langchain-community` ≥ 0.4.2 removed `chat_models.vertexai`, which RAGAS imports
   unconditionally at module load. Pinned to 0.4.1 (upstream issues #2741, #2745,
   #2753). Installing `langchain-google-vertexai` does not help — the import path is
   inside `langchain_community`.
2. RAGAS requests multiple completions per call (`n > 1`); Groq rejects this.
   Resolved with `bypass_n=True` on the LLM wrapper.
3. `answer_relevancy` generates synthetic questions via extra LLM calls to
   approximate a similarity computable in one embedding call, and its telemetry
   conflicts with `fastembed`. Replaced with a direct cosine.

---

## Tuning

Parameters are coupled. Changing an early one invalidates everything measured after
it, so sweep in this order:

| Order | Parameter | Measure | Requires re-index |
| --- | --- | --- | --- |
| 1 | `EMBED_MODEL` | in-domain mean, in/out gap | yes |
| 2 | `CHUNK_SIZE`, `CHUNK_OVERLAP` | top-1 score, correct-source hit rate | yes |
| 3 | `MERGE_FLOOR`, `MIN_CHUNK` | chunk-length histogram | yes |
| 4 | `TOP_K`, tail margin | how many retrieved chunks get cited | no |
| 5 | `TAU` | refusal accuracy on near-miss + out-of-domain | no |
| 6 | Prompt | near-miss refusal rate, citation correctness | no |
| 7 | `SCORE_LO`, `SCORE_HI` | display only | no |

τ must come last — it is a cut point on a distribution that steps 1–3 move.

Watch the **gap** between in-domain and out-of-domain scores, not the absolute level.
A change that raises both equally has bought nothing.

---

## Layout

```
core/config.py        paths, tunables, calibration constants
ingest/parsers.py     pdf/docx → Block
ingest/chunker.py     Block → Chunk
ingest/store.py       Chroma wrapper: upsert, delete_by_source, search
ingest/manifest.py    content hashing and three-way diff
ingest/sync.py        reconcile loop
rag/llm.py            provider layer (Groq / Ollama)
rag/retrieve.py       search + gate A
rag/generate.py       prompt + gate B
rag/evaluate_ragas.py three metrics
app.py                Streamlit UI
```
