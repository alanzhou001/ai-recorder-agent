from faster_whisper import WhisperModel
from app.core.config import ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_MODEL_DIR

_MODEL_CACHE: dict[str, WhisperModel] = {}

def get_whisper_model(model_name: str) -> WhisperModel:
    if model_name not in _MODEL_CACHE:
        print(f"[ASR] Loading model={model_name} device={ASR_DEVICE} compute_type={ASR_COMPUTE_TYPE} root={ASR_MODEL_DIR}")
        _MODEL_CACHE[model_name] = WhisperModel(
            model_name,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
            download_root=str(ASR_MODEL_DIR),
        )
    return _MODEL_CACHE[model_name]
