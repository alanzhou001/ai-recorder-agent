# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import subprocess
import shutil
import uuid

from app.asr.transcriber import transcribe_audio
from app.api.schemas import TranscriptionResponse

router = APIRouter()

RECORDING_DIR = Path(__file__).resolve().parents[2] / "data" / "recordings"
RECORDING_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/asr/transcribe", response_model=TranscriptionResponse)
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = Path(file.filename).suffix
    file_id = f"{uuid.uuid4()}{suffix}"
    file_path = RECORDING_DIR / file_id

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = transcribe_audio(
        audio_path=str(file_path),
        model_name=model,
    )

    return result

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/diag/gpu")
def diag_gpu():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True)
        return {"nvidia_smi": out.strip()}
    except Exception as e:
        return {"nvidia_smi": None, "error": str(e)}
