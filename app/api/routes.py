# app/api/routes.py
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.asr.transcriber import transcribe_audio
from app.storage.session_store import get_session_paths, locked_session
from app.storage.artifacts import artifacts_dir, read_jsonl, write_jsonl

from app.llm.providers.openai_compat import OpenAICompatProvider
from app.llm.postedit import postedit_segment

from app.rag.index import build_index
from app.rag.retriever import retrieve
from app.rag.qa import answer_question
from app.rag.summary import summarize_session

router = APIRouter()


# ----------------------------
# File paths + transcript I/O
# ----------------------------
def _meta_path(session_id: str) -> Path:
    return get_session_paths(session_id).root / "meta.json"


def _transcript_path(session_id: str) -> Path:
    return get_session_paths(session_id).root / "transcript.jsonl"


def _load_meta(session_id: str) -> Dict[str, Any]:
    p = _meta_path(session_id)
    if not p.exists():
        raise FileNotFoundError
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(session_id: str, meta: Dict[str, Any]) -> None:
    p = _meta_path(session_id)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_transcript_items(session_id: str, items: List[Dict[str, Any]]) -> None:
    p = _transcript_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _read_transcript_items(session_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    p = _transcript_path(session_id)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _probe_duration_seconds_ffprobe(path: str) -> float:
    """
    Conservative: use ffprobe (works for m4a/wav). You can later optimize for wav.
    """
    import subprocess

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


# ----------------------------
# Request/Response models
# ----------------------------
class QARequest(BaseModel):
    question: str
    k: int = Field(default=6, ge=1, le=20)


class SummaryRequest(BaseModel):
    # limit excerpts passed to LLM to control token budget
    max_segments: int = Field(default=120, ge=10, le=1000)
    guidance: Optional[str] = None


class PostEditRequest(BaseModel):
    # how many segments to post-edit per call (incremental)
    batch_size: int = Field(default=40, ge=1, le=200)
    # whether to use raw transcript or only "stable/final" ones (if you store that separately later)
    source: str = Field(default="raw")  # currently only "raw" is implemented


# ----------------------------
# Basic health
# ----------------------------
@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


# ----------------------------
# ASR: single-shot
# ----------------------------
@router.post("/asr/transcribe")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Single-shot transcription endpoint (no session).
    """
    # store temp file
    tmp_dir = Path("data") / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".bin"
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    with open(tmp_path, "wb") as f:
        f.write(file.file.read())

    try:
        result = transcribe_audio(str(tmp_path), model_name=model, language=language)
        return result
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ----------------------------
# Sessions
# ----------------------------
@router.post("/sessions")
def create_session(
    model: str = Form("small"),
    language: Optional[str] = Form(None),
) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex
    paths = get_session_paths(session_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.chunks.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "model": model,
        "language": language,
        "created_at": time.time(),
        "next_offset": 0.0,
        "chunk_index": 0,
    }
    _save_meta(session_id, meta)

    return {"session_id": session_id, "model": model, "language": language}


@router.get("/sessions/{session_id}/transcript")
def get_transcript(
    session_id: str,
    after: float = 0.0,
    limit: int = 5000,
) -> Dict[str, Any]:
    """
    Returns transcript items with end > after (incremental pull).
    """
    with locked_session(session_id):
        items = _read_transcript_items(session_id, limit=limit)
    if after > 0:
        items = [x for x in items if float(x.get("end", 0.0)) > after]
    return {"session_id": session_id, "items": items}


@router.post("/sessions/{session_id}/chunks")
def upload_chunk(
    session_id: str,
    file: UploadFile = File(...),
    t_offset: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """
    Upload one audio chunk to a session.
    - Saves chunk
    - Transcribes using faster-whisper
    - Appends segments to transcript.jsonl with absolute timestamps
    """
    with locked_session(session_id):
        try:
            meta = _load_meta(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

        model = meta.get("model", "small")
        language = meta.get("language", None)

        chunk_index = int(meta.get("chunk_index", 0))
        paths = get_session_paths(session_id)
        paths.chunks.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "").suffix or ".bin"
        chunk_path = paths.chunks / f"{chunk_index:06d}{suffix}"

        # Save uploaded bytes
        with open(chunk_path, "wb") as f:
            f.write(file.file.read())

        # Determine offset: client override or server-accumulated
        if t_offset is None:
            t_offset = float(meta.get("next_offset", 0.0))
        else:
            t_offset = float(t_offset)

        # Duration (for next_offset)
        duration = _probe_duration_seconds_ffprobe(str(chunk_path))

        # ASR
        result = transcribe_audio(str(chunk_path), model_name=model, language=language)
        # Normalize: result["segments"] expected as list with start/end/text (relative)
        segments = result.get("segments", [])

        abs_items: List[Dict[str, Any]] = []
        for seg in segments:
            start = float(seg["start"]) + t_offset
            end = float(seg["end"]) + t_offset
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            abs_items.append(
                {
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "audio_path": str(chunk_path),
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        _append_transcript_items(session_id, abs_items)

        # Update meta
        meta["chunk_index"] = chunk_index + 1
        meta["next_offset"] = float(t_offset) + float(duration)
        # Persist language if ASR detected it and session language not set
        lang_det = result.get("language")
        if not meta.get("language") and lang_det:
            meta["language"] = lang_det
        _save_meta(session_id, meta)

    return {
        "session_id": session_id,
        "chunk_index": chunk_index,
        "t_offset": float(t_offset),
        "duration": float(duration),
        "model": model,
        "language": meta.get("language"),
        "segments": [
            {
                "id": int(s.get("id", 0)),
                "start": float(s["start"]) + float(t_offset),
                "end": float(s["end"]) + float(t_offset),
                "text": str(s.get("text", "")).strip(),
            }
            for s in segments
            if str(s.get("text", "")).strip()
        ],
    }


# ----------------------------
# LLM Post-edit (faithful)
# ----------------------------
@router.post("/sessions/{session_id}/llm/postedit")
async def llm_postedit(
    session_id: str,
    req: PostEditRequest,
) -> Dict[str, Any]:
    """
    Incrementally post-edit transcript items into artifacts/clean_transcript.jsonl.
    - Processes next N raw items that have not been post-edited yet
    - Writes JSONL rows with {start,end,text_raw,text_clean,changes,notes}
    """
    with locked_session(session_id):
        raw_items = _read_transcript_items(session_id, limit=50000)

    if not raw_items:
        raise HTTPException(status_code=400, detail="No transcript items found for this session")

    adir = artifacts_dir(session_id)
    clean_path = adir / "clean_transcript.jsonl"

    # Determine how many already processed
    done_rows = read_jsonl(clean_path, limit=100000)
    done_n = len(done_rows)

    # Select batch
    batch = raw_items[done_n : done_n + int(req.batch_size)]
    if not batch:
        return {"session_id": session_id, "processed": 0, "already_done": done_n, "path": str(clean_path)}

    provider = OpenAICompatProvider()

    out_rows: List[Dict[str, Any]] = []
    for it in batch:
        # each it: {start,end,text,...}
        seg = {"start": it["start"], "end": it["end"], "text": it["text"]}
        row = await postedit_segment(provider, seg)
        out_rows.append(row)

    # Append to clean transcript
    write_jsonl(clean_path, out_rows)

    return {
        "session_id": session_id,
        "processed": len(out_rows),
        "already_done": done_n,
        "path": str(clean_path),
    }


def _load_docs_for_rag(session_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    """
    Prefer clean transcript if available, else raw transcript.
    Returns docs: {start,end,text}
    """
    adir = artifacts_dir(session_id)
    clean_path = adir / "clean_transcript.jsonl"
    clean = read_jsonl(clean_path, limit=limit)
    if clean:
        docs = [{"start": r["start"], "end": r["end"], "text": r.get("text_clean", r.get("text_raw", ""))} for r in clean]
        docs = [d for d in docs if str(d["text"]).strip()]
        return docs

    raw = _read_transcript_items(session_id, limit=limit)
    docs = [{"start": r["start"], "end": r["end"], "text": r["text"]} for r in raw if str(r.get("text", "")).strip()]
    return docs


# ----------------------------
# RAG QA
# ----------------------------
@router.post("/sessions/{session_id}/qa")
async def rag_qa(session_id: str, req: QARequest) -> Dict[str, Any]:
    docs = _load_docs_for_rag(session_id, limit=8000)
    if not docs:
        raise HTTPException(status_code=400, detail="No transcript available")

    index = build_index(docs)
    excerpts = retrieve(index, req.question, k=int(req.k))

    provider = OpenAICompatProvider()
    out = await answer_question(provider, question=req.question, excerpts=excerpts)

    return {
        "session_id": session_id,
        "question": req.question,
        "answer": out["answer"],
        "citations": out.get("citations", []),
        "excerpts": excerpts,  # useful for debugging; remove later if you want
    }


# ----------------------------
# Session Summary (RAG-style)
# ----------------------------
@router.post("/sessions/{session_id}/summary")
async def rag_summary(session_id: str, req: SummaryRequest) -> Dict[str, Any]:
    """
    Summarize using ONLY transcript excerpts (with citations).
    Strategy:
    - Use up to max_segments segments from the *end* (recent context tends to include conclusions)
    - This is conservative & simple; you can later switch to topic-based sampling.
    """
    docs = _load_docs_for_rag(session_id, limit=20000)
    if not docs:
        raise HTTPException(status_code=400, detail="No transcript available")

    # take tail segments to control token budget
    excerpts = docs[-int(req.max_segments):]

    provider = OpenAICompatProvider()
    out = await summarize_session(provider, excerpts=excerpts, guidance=req.guidance)

    # persist summary artifact
    adir = artifacts_dir(session_id)
    summary_path = adir / "summary.json"
    summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "session_id": session_id,
        "summary": out,
        "path": str(summary_path),
    }
