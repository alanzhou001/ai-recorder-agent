# app/asr/transcriber.py
from typing import List, Dict, Any, Optional
from .model_manager import get_whisper_model


def transcribe_audio(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
    t_offset: float = 0.0,
) -> Dict[str, Any]:
    model = get_whisper_model(model_name)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
    )

    results: List[Dict[str, Any]] = []
    for seg in segments:
        results.append(
            {
                "id": seg.id,
                "start": float(seg.start) + float(t_offset),
                "end": float(seg.end) + float(t_offset),
                "text": seg.text.strip(),
            }
        )

    return {
        "model": model_name,
        "language": info.language,
        "segments": results,
    }
