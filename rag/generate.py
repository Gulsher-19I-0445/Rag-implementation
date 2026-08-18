import json

from core.config import config
from rag.llm import LLM
from rag.retrieve import Retrieval

SYSTEM = """You answer questions strictly from the numbered sources provided.

Rules:
- Use ONLY the sources. Never use outside knowledge. Never guess.
- Every factual sentence must end with a citation like [2].
- If the sources do not contain the answer, set answerable to false and say you don't know.
- Partial information is not an answer. If the sources only touch the topic, say so.

Respond with JSON only:
{"answerable": true|false, "answer": "...", "citations": [1, 2]}

Example:
Sources: [1] The office opens at 9am.
Question: What is the CEO's salary?
{"answerable": false, "answer": "I don't know — the documents don't cover this.", "citations": []}"""

REFUSAL = "I don't know — I couldn't find anything relevant in the documents."


def answer(llm: LLM, query: str, r: Retrieval) -> dict:
    if not r.grounded:
        return {"answerable": False, "answer": REFUSAL, "citations": [], "gate": "retrieval"}

    user = f"Sources:\n{r.as_context()}\n\nQuestion: {query}"
    raw = llm.complete(SYSTEM, user, json_mode=True)

    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"answerable": False, "answer": REFUSAL, "citations": [], "gate": "parse_error"}

    if not out.get("answerable"):
        out["answer"] = out.get("answer") or REFUSAL
        out["gate"] = "model"
    else:
        out["gate"] = None
    return out