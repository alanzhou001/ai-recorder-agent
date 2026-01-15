# app/storage/artifacts.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.storage.session_store import get_session_paths


def artifacts_dir(session_id: str) -> Path:
    """
    data/sessions/<session_id>/artifacts/
    """
    p = get_session_paths(session_id).root / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, limit: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out
