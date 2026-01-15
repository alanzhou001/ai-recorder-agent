# app/storage/session_state.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from app.storage.session_store import get_session_paths, locked_session


DEFAULT_STATE = {
    "buffer_text": "",     # 当前滚动中的未完成句子
    "final_count": 0,      # 已稳定输出的句子数（可用于客户端增量拉取）
    "last_end": 0.0,       # 最近一次处理到的时间（可选）
}


def state_path(session_id: str) -> Path:
    return get_session_paths(session_id).root / "state.json"


def load_state(session_id: str) -> Dict[str, Any]:
    p = state_path(session_id)
    if not p.exists():
        return dict(DEFAULT_STATE)
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(session_id: str, state: Dict[str, Any]) -> None:
    p = state_path(session_id)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
