from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

ASR_MODEL_DEFAULT = env("ASR_MODEL_DEFAULT", "small")
ASR_DEVICE = env("ASR_DEVICE", "cuda")          # cpu / cuda / auto
ASR_COMPUTE_TYPE = env("ASR_COMPUTE_TYPE", "float16")  # int8 / float16 / auto
ASR_MODEL_DIR = PROJECT_ROOT / (env("ASR_MODEL_DIR", "data/models") or "data/models")
ASR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
