from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def tokenize(s: str) -> List[str]:
    # 中文先用简单字符/空格混合；后续可换 jieba
    s = s.strip()
    if " " in s:
        return [t for t in s.split() if t]
    return [c for c in s if c.strip()]

@dataclass
class RAGIndex:
    bm25: BM25Okapi
    docs: List[Dict[str, Any]]   # each: {start,end,text}

def build_index(docs: List[Dict[str, Any]]) -> RAGIndex:
    corpus = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus)
    return RAGIndex(bm25=bm25, docs=docs)
