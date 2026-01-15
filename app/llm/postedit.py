from __future__ import annotations
from typing import Dict, Any
from .prompts import POSTEDIT_SYSTEM, POSTEDIT_USER_TEMPLATE
from .providers.base import LLMProvider

async def postedit_segment(provider: LLMProvider, seg: Dict[str, Any]) -> Dict[str, Any]:
    user = POSTEDIT_USER_TEMPLATE.format(start=seg["start"], end=seg["end"], text=seg["text"])
    out = await provider.chat_json(system=POSTEDIT_SYSTEM, user=user)
    clean = str(out.get("clean_text", "")).strip()
    if not clean:
        # 保守：LLM 输出异常则回退原文
        clean = seg["text"]
    return {
        "start": seg["start"],
        "end": seg["end"],
        "text_raw": seg["text"],
        "text_clean": clean,
        "changes": out.get("changes", []),
        "notes": out.get("notes", ""),
    }
