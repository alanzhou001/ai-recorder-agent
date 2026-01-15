# app/rag/qa.py
from __future__ import annotations

from typing import Any, Dict, List

from app.llm.providers.base import LLMProvider


QA_SYSTEM = """Answer ONLY using the provided excerpts.

Rules:
- Use ONLY information present in EXCERPTS.
- If EXCERPTS do not contain the answer, say you don't know.
- Do not invent names, numbers, or context.
- Add citations in the answer using [t=START-END] after each claim.
- Return STRICT JSON:
  {
    "answer": "...",
    "citations": [{"start": 0.0, "end": 1.0, "quote": "..."}]
  }
"""

def _format_excerpts(excerpts: List[Dict[str, Any]]) -> str:
    lines = []
    for e in excerpts:
        lines.append(f"[t={float(e['start']):.3f}-{float(e['end']):.3f}] {e['text']}")
    return "\n".join(lines)


async def answer_question(
    provider: LLMProvider,
    *,
    question: str,
    excerpts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    user = f"QUESTION: {question}\n\nEXCERPTS:\n{_format_excerpts(excerpts)}\n\nReturn JSON only."
    out = await provider.chat_json(system=QA_SYSTEM, user=user)

    # Minimal hardening
    answer = str(out.get("answer", "")).strip()
    citations = out.get("citations", [])
    if not answer:
        answer = "I don't know."

    return {"answer": answer, "citations": citations}
