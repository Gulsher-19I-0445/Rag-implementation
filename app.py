# app.py
import streamlit as st

from core.config import config
from ingest.store import Store
from ingest.sync import audit, reconcile
from eval.evaluate_ragas import evaluate
from rag.generate import answer
from rag.llm import LLM
from rag.retrieve import retrieve

st.set_page_config(page_title="Grounded RAG", page_icon="📄", layout="wide")


@st.cache_resource
def get_store() -> Store:
    return Store()


@st.cache_resource
def get_llm() -> LLM:
    return LLM()


store, llm = get_store(), get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "synced" not in st.session_state:
    with st.spinner("Syncing documents…"):
        st.session_state.sync = reconcile(store)
    st.session_state.synced = True


with st.sidebar:
    st.subheader("Index")
    r = st.session_state.sync
    st.caption(r.summary())
    c1, c2 = st.columns(2)
    c1.metric("Chunks", store.count())
    c2.metric("Documents", len(store.sources()))

    if st.button("Re-sync", use_container_width=True):
        with st.spinner("Scanning…"):
            st.session_state.sync = reconcile(store)
        st.rerun()

    with st.expander("Indexed files"):
        for s in sorted(store.sources()):
            st.caption(s)

    for problem in audit(store):
        st.warning(problem, icon="⚠️")

    st.divider()
    st.subheader("Settings")
    judge = st.toggle("Run faithfulness check", value=True,
                      help="Adds an LLM judging pass — slower, but scores each claim")
    st.caption(f"{llm.name} · {llm.cfg.model}")
    st.caption(f"τ = {config.TAU}  ·  k = {config.TOP_K}")


def _colour(v: float | None) -> str:
    if v is None:
        return "off"
    return "normal" if v >= 0.8 else "off" if v >= 0.5 else "inverse"


def render_report(rep, out):
    a, b, c = st.columns(3)
    a.metric("Grounding", f"{rep.grounding:.2f}",
             help="How well the retrieved passages match the question")
    b.metric("Relevance", f"{rep.answer_relevancy:.2f}" if rep.answer_relevancy else "—",
             help="Similarity between the question and the answer")
    if rep.faithfulness is None:
        c.metric("Faithfulness", "—",
                 help="Not scored — the model declined to answer")
    else:
        c.metric("Faithfulness", f"{rep.faithfulness:.2f}",
                 help="Share of claims supported by the retrieved passages")
    if rep.error:
        st.caption(f"eval error: {rep.error}")


def render_sources(r):
    with st.expander(f"Sources ({len(r.hits)})"):
        for i, h in enumerate(r.hits, 1):
            st.markdown(f"**[{i}]** {h['doc_title']} — {h['locator']}  ·  `{h['score']:.3f}`")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


st.title("Ask your documents")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("report"):
            render_report(m["report"], m["out"])
            render_sources(m["retrieval"])

if q := st.chat_input("Ask a question about the indexed documents…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            r = retrieve(store, q)
        with st.spinner("Answering…"):
            out = answer(llm, q, r)

        st.markdown(out["answer"])
        if not out["answerable"]:
            reason = {"retrieval": "nothing relevant was retrieved",
                      "model": "the retrieved passages don't cover this",
                      "parse_error": "the model returned malformed output"}
            st.info(f"Not answered — {reason.get(out.get('gate'), 'unknown')}", icon="🚫")

        with st.spinner("Evaluating…"):
            rep = evaluate(q, out["answer"], r,
                           judge=judge)

        render_report(rep, out)
        render_sources(r)

    st.session_state.messages.append({
        "role": "assistant", "content": out["answer"],
        "report": rep, "out": out, "retrieval": r,
    })