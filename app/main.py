from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Recorder Agent", version="0.2.0")
app.include_router(router)
