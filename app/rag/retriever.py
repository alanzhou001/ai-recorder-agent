from __future__ import annotations
from typing import List, Dict, Any
from .index import RAGIndex, tokenize

def retrieve(index: RAGIndex, query: str, k: int = 6) -> List[Dict[str, Any]]:
    q = tokenize(query)
    scores = index.bm25.get_scores(q)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in ranked:
        d = dict(index.docs[i])
        d["score"] = float(scores[i])
        out.append(d)
    return out
