# app/llm/providers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    """
    Minimal provider interface for JSON-producing chat calls.
    """

    @abstractmethod
    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a parsed JSON object (dict). Provider must ensure the model returns valid JSON.
        """
        raise NotImplementedError
