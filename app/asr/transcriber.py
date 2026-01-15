# app/asr/transcriber.py
from typing import List, Dict, Any
from .model_manager import get_whisper_model


def transcribe_audio(
    audio_path: str,
    model_name: str = "small",
    language: str | None = None,
) -> Dict[str, Any]:
    """
    转写音频文件，返回 segments
    """
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
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
        )

    return {
        "model": model_name,
        "language": info.language,
        "segments": results,
    }
