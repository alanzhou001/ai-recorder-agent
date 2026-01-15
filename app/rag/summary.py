# app/rag/summary.py
from __future__ import annotations

from typing import Any, Dict, List

from app.llm.providers.base import LLMProvider


SUMMARY_SYSTEM = """You are a meeting summarizer.

Rules (must follow):
- Use ONLY information in EXCERPTS.
- Do NOT invent names, numbers, decisions, or context.
- If uncertain or missing, say so explicitly.
- Provide citations after each bullet/claim using [t=START-END].
- Return STRICT JSON with keys:
  {
    "title": string,
    "highlights": [string],
    "action_items": [{"item": string, "owner": string|null, "due": string|null, "cite": "[t=...-...]"}],
    "open_questions": [string],
    "citations": [{"start": number, "end": number, "quote": string}]
  }
"""

def _format_excerpts(excerpts: List[Dict[str, Any]]) -> str:
    lines = []
    for e in excerpts:
        start = float(e["start"])
        end = float(e["end"])
        text = str(e["text"])
        lines.append(f"[t={start:.3f}-{end:.3f}] {text}")
    return "\n".join(lines)


async def summarize_session(
    provider: LLMProvider,
    *,
    excerpts: List[Dict[str, Any]],
    guidance: str | None = None,
) -> Dict[str, Any]:
    """
    Summarize from provided excerpts only.
    guidance: optional user instruction, e.g. "focus on decisions" / "technical summary"
    """
    g = f"\nGUIDANCE: {guidance}\n" if guidance else ""
    user = f"{g}\nEXCERPTS:\n{_format_excerpts(excerpts)}\n\nReturn JSON only."
    out = await provider.chat_json(system=SUMMARY_SYSTEM, user=user)

    # Minimal hardening
    title = str(out.get("title", "")).strip() or "Session Summary"
    highlights = out.get("highlights", [])
    action_items = out.get("action_items", [])
    open_questions = out.get("open_questions", [])
    citations = out.get("citations", [])

    if not isinstance(highlights, list):
        highlights = []
    if not isinstance(action_items, list):
        action_items = []
    if not isinstance(open_questions, list):
        open_questions = []
    if not isinstance(citations, list):
        citations = []

    return {
        "title": title,
        "highlights": highlights,
        "action_items": action_items,
        "open_questions": open_questions,
        "citations": citations,
    }
