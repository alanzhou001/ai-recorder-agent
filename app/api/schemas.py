# app/api/schemas.py
from pydantic import BaseModel
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
