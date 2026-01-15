# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import shutil

from app.asr.transcriber import transcribe_audio
from app.api.schemas import (
    TranscriptionResponse,
    CreateSessionResponse,
    ChunkTranscribeResponse,
)
from app.storage.session_store import (
    create_session,
    get_session_paths,
    load_meta,
    save_meta,
    append_transcript,
    read_transcript,
    locked_session,
)
from app.audio.probe import probe_duration_seconds

router = APIRouter()

RECORDING_DIR = Path(__file__).resolve().parents[2] / "data" / "recordings"
RECORDING_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/asr/transcribe", response_model=TranscriptionResponse)
def transcribe_once(
    file: UploadFile = File(...),
    model: str = Form("small"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = Path(file.filename).suffix or ".wav"
    file_path = RECORDING_DIR / f"upload{suffix}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = transcribe_audio(str(file_path), model_name=model)
    return result


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session_api(
    model: str = Form("small"),
    language: str | None = Form(None),
):
    meta = create_session(model=model, language=language)
    return {"session_id": meta["session_id"], "model": meta["model"], "language": meta["language"]}


@router.post("/sessions/{session_id}/chunks", response_model=ChunkTranscribeResponse)
def upload_chunk(
    session_id: str,
    file: UploadFile = File(...),
    # 可选：客户端显式给本 chunk 在 session 时间轴的起点（秒）
    t_offset: float | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    paths = get_session_paths(session_id)
    if not paths.root.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # per-session lock：保护 meta/累计 offset/写 transcript
    with locked_session(session_id):
        meta = load_meta(session_id)

        chunk_index = int(meta.get("chunk_count", 0))
        suffix = Path(file.filename).suffix or ".wav"
        chunk_path = paths.chunks_dir / f"{chunk_index:06d}{suffix}"

        with open(chunk_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # chunk 时长（用于 server-side 自动累计 offset）
        try:
            duration = probe_duration_seconds(str(chunk_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")

        # 选择 offset：客户端提供优先，否则用服务端累计 next_offset
        if t_offset is None:
            t_offset_used = float(meta.get("next_offset", 0.0))
        else:
            t_offset_used = float(t_offset)

        # ASR：使用 session 的 model/language
        model = meta.get("model", "small")
        language = meta.get("language", None)

        result = transcribe_audio(
            audio_path=str(chunk_path),
            model_name=model,
            language=language,
            t_offset=t_offset_used,
        )

        from app.summarizer.merger import merge_segments
        result["segments"] = merge_segments(result["segments"])

        # 写入 transcript（每个 segment 作为一条记录，附加 chunk 信息）
        records = []
        for seg in result["segments"]:
            records.append(
                {
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "audio_path": str(chunk_path),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )
        append_transcript(session_id, records)

        # 更新 meta：累计 offset（仅当客户端没给 t_offset 时）
        meta["chunk_count"] = chunk_index + 1
        if t_offset is None:
            meta["next_offset"] = float(meta.get("next_offset", 0.0)) + float(duration)
        save_meta(session_id, meta)

        return {
            "session_id": session_id,
            "chunk_index": chunk_index,
            "t_offset": t_offset_used,
            "duration": float(duration),
            "model": result["model"],
            "language": result["language"],
            "segments": result["segments"],
        }


@router.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str, after: float = 0.0, limit: int = 5000):
    try:
        items = read_transcript(session_id, limit=limit)
        # 只返回 end > after 的新增部分
        items = [x for x in items if float(x.get("end", 0.0)) > after]
        return {"session_id": session_id, "items": items}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

