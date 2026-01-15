# app/main.py
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Recorder Agent",
        description=(
            "Real-time audio recording, ASR (Whisper), "
            "LLM post-editing, and RAG-based summary & QA."
        ),
        version="0.1.0",
    )

    # CORS: allow local dev (Windows client / browser / future UI)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # dev-friendly; restrict in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(
        api_router,
        prefix="",                    # all routes already absolute
        tags=["api"],
    )

    # Simple root endpoint
    @app.get("/")
    def root():
        return {
            "name": "ai-recorder-agent",
            "status": "running",
        }

    return app


# Uvicorn entrypoint
app = create_app()
