# scripts/ask.py
import sys

from ingest.store import Store
from eval.evaluate_ragas import evaluate
from rag.generate import answer
from rag.llm import LLM
from rag.retrieve import retrieve

QUERIES = sys.argv[1:] or [
    "who is gulsher khan",
    "what does the procurement team receive",
    "what is the vendor's late-delivery penalty",
    "What happens whene either party terminates this contract",
]

store, llm = Store(), LLM()
print(f"provider={llm.name} model={llm.cfg.model}  |  {store.count()} chunks\n")

for q in QUERIES:
    r = retrieve(store, q)
    out = answer(llm, q, r)
    # rep = evaluate(llm, q, out["answer"], r)

    print(f"Q: {q}")
    print(f"   gate={out.get('gate')}  answerable={out['answerable']}  top={r.top_score:.3f}")
    print(f"   A: {out['answer'][:160]}")
    rep = evaluate(q, out["answer"], r)
    print(f"   grounding={rep.grounding:.2f}  "
          f"faithfulness={rep.faithfulness}  relevancy={rep.answer_relevancy}")
    if rep.error:
        print(f"   EVAL ERROR: {rep.error}")
    print(f"   sources: {[f'{h[chr(39)]}' for h in []] or [h['locator'] for h in r.hits]}\n")