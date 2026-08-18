# rag/evaluate_ragas.py
from dataclasses import dataclass

from datasets import Dataset
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
import numpy as np
from ingest.store import embed
from ragas.metrics import faithfulness

from core.config import config
from rag.retrieve import Retrieval
from rag.llm import LLM

_judge = None
_emb = None


_judge_llm: LLM | None = None


def _relevance(question: str, answer: str) -> float:
    q, a = embed([question, answer])
    q, a = np.array(q), np.array(a)
    return max(0.0, float(q @ a / (np.linalg.norm(q) * np.linalg.norm(a))))

def _wrappers():
    global _judge, _emb, _judge_llm
    if _judge is None:
        _judge_llm = LLM()                      # reads LLM_PROVIDER as usual
        cfg = _judge_llm.cfg
        _judge = LangchainLLMWrapper(ChatOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=config.JUDGE_MODEL or cfg.model,
            temperature=0,
            n=1,
        ),
        bypass_n=True
        )
        # _emb = LangchainEmbeddingsWrapper(
        #     FastEmbedEmbeddings(model_name=config.EMBED_MODEL)
        # )
        fe = FastEmbedEmbeddings(model_name=config.EMBED_MODEL)
        fe.model = config.EMBED_MODEL          # what the telemetry event wants
        _emb = LangchainEmbeddingsWrapper(fe)
    return _judge, _emb


@dataclass
class Report:
    grounding: float
    faithfulness: float | None
    answer_relevancy: float | None
    error: str | None = None


def _grounding(r: Retrieval) -> float:
    lo, hi = config.SCORE_LO, config.SCORE_HI
    return max(0.0, min(1.0, (r.top_score - lo) / (hi - lo)))


def evaluate(question: str, answer: str, r: Retrieval, judge: bool = True, answerable: bool = True) -> Report:
    g = _grounding(r)
    rel = _relevance(question, answer)

    if not r.grounded or not answerable or not judge:
        return Report(grounding=g, faithfulness=None, answer_relevancy=rel)

    llm, emb = _wrappers()
    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [[h["text"] for h in r.hits]],
    })

    try:
        # res = ragas_evaluate(
        #     ds, metrics=[faithfulness, answer_relevancy],
        #     llm=llm, embeddings=emb, raise_exceptions=False,
        # )
        # df = res.to_pandas()
        res = ragas_evaluate(ds, metrics=[faithfulness], llm=llm, embeddings=emb,
                         raise_exceptions=False)
        df = res.to_pandas()
        return Report(
            grounding=g,
            faithfulness=_num(df, "faithfulness"),
            answer_relevancy=_relevance(question, answer),
        )
    except Exception as e:
        return Report(grounding=g, faithfulness=None,
                      answer_relevancy=None, error=str(e))


def _num(df, col: str) -> float | None:
    """RAGAS writes NaN, not an exception, when a metric fails."""
    if col not in df.columns:
        return None
    v = df[col].iloc[0]
    return None if v != v else float(v)      # NaN != NaN

