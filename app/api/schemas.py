# app/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from typing import List

class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    model: str
    language: str
    segments: List[Segment]


class CreateSessionResponse(BaseModel):
    session_id: str
    model: str
    language: Optional[str] = None


class ChunkTranscribeResponse(BaseModel):
    session_id: str
    chunk_index: int
    t_offset: float
    duration: float
    model: str
    language: str
    segments: List[Segment]

    partial_text: str
    final_texts: List[str]


class ChunkSubtitle(BaseModel):
    partial_text: str
    final_texts: List[str]
