# app/llm/providers/openai_compat.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx

from .base import LLMProvider


class OpenAICompatProvider(LLMProvider):
    """
    Talks to OpenAI-compatible Chat Completions endpoints:
      POST {LLM_BASE_URL}/v1/chat/completions

    Env:
      LLM_BASE_URL: e.g. https://api.openai.com  OR other vendor's compat base
      LLM_API_KEY:  token
      LLM_MODEL:    model name
    """

    def __init__(self) -> None:
        self.base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY") or ""
        self.model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_S") or "120")

        if not self.base_url:
            raise RuntimeError("Missing env LLM_BASE_URL")
        if not self.api_key:
            raise RuntimeError("Missing env LLM_API_KEY")

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            # Widely supported JSON mode in compat endpoints:
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Expected structure: choices[0].message.content is JSON string
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e

        try:
            return json.loads(content)
        except Exception as e:
            # Preserve the raw content for debugging
            raise RuntimeError(f"Model did not return valid JSON. content={content!r}") from e
