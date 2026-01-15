# app/storage/session_store.py
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "sessions"
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_G = threading.Lock()


def _get_lock(session_id: str) -> threading.Lock:
    with _LOCKS_G:
        if session_id not in _LOCKS:
            _LOCKS[session_id] = threading.Lock()
        return _LOCKS[session_id]


@dataclass
class SessionPaths:
    root: Path
    chunks_dir: Path
    meta_path: Path
    transcript_jsonl: Path


def get_session_paths(session_id: str) -> SessionPaths:
    root = SESSIONS_ROOT / session_id
    return SessionPaths(
        root=root,
        chunks_dir=root / "chunks",
        meta_path=root / "meta.json",
        transcript_jsonl=root / "transcript.jsonl",
    )


def create_session(model: str = "small", language: Optional[str] = None) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex
    paths = get_session_paths(session_id)
    paths.chunks_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "model": model,
        "language": language,
        "next_offset": 0.0,   # 服务端累计时间轴（秒）
        "chunk_count": 0,
    }
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_meta(session_id: str) -> Dict[str, Any]:
    paths = get_session_paths(session_id)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(paths.meta_path.read_text(encoding="utf-8"))


def save_meta(session_id: str, meta: Dict[str, Any]) -> None:
    paths = get_session_paths(session_id)
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def append_transcript(session_id: str, records: List[Dict[str, Any]]) -> None:
    """
    追加写入 transcript.jsonl，一行一个 JSON 记录。
    """
    paths = get_session_paths(session_id)
    with open(paths.transcript_jsonl, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_transcript(session_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    paths = get_session_paths(session_id)
    if not paths.transcript_jsonl.exists():
        return []
    lines = paths.transcript_jsonl.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    return [json.loads(x) for x in lines if x.strip()]


def locked_session(session_id: str):
    """
    简单的 per-session 互斥锁上下文，用于并发 chunk 上传时保护 meta/写入。
    """
    lock = _get_lock(session_id)
    return lock
