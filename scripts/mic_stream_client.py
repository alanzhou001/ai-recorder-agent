import argparse
import io
import time
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


def create_session(base_url: str, model: str, language: Optional[str], timeout: float) -> str:
    url = f"{base_url}/sessions"
    data = {"model": model}
    if language is not None:
        data["language"] = language

    # 禁用代理，避免 http_proxy 把 127.0.0.1 请求劫持到 1080
    r = requests.post(url, data=data, timeout=timeout, proxies={"http": None, "https": None})
    r.raise_for_status()
    session_id = r.json()["session_id"]
    return session_id


def post_chunk(
    base_url: str,
    session_id: str,
    wav_bytes: bytes,
    filename: str,
    t_offset: Optional[float],
    timeout: float,
):
    url = f"{base_url}/sessions/{session_id}/chunks"
    files = {"file": (filename, wav_bytes, "audio/wav")}
    data = {}
    if t_offset is not None:
        data["t_offset"] = str(t_offset)

    r = requests.post(
        url,
        files=files,
        data=data,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    r.raise_for_status()
    return r.json()


def wav_bytes_from_float32(x: np.ndarray, sr: int) -> bytes:
    """
    x: shape (n,), float32 in [-1, 1]
    """
    bio = io.BytesIO()
    sf.write(bio, x, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base url")
    p.add_argument("--model", default="small", help="small / medium etc.")
    p.add_argument("--language", default=None, help="e.g. zh; or omit for auto-detect")
    p.add_argument("--sr", type=int, default=16000, help="sample rate")
    p.add_argument("--chunk-sec", type=float, default=1.5, help="chunk seconds")
    p.add_argument("--device", type=int, default=None, help="input device index (optional)")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    p.add_argument(
        "--mode",
        choices=["server_accumulate", "client_offset"],
        default="server_accumulate",
        help="server_accumulate: do not send t_offset; client_offset: send cumulative offset",
    )
    args = p.parse_args()

    session_id = create_session(args.base_url, args.model, args.language, args.timeout)
    print(f"[client] session_id={session_id}")

    sr = args.sr
    chunk_samples = int(sr * args.chunk_sec)

    # client-side timeline (only used when mode=client_offset)
    t_offset = 0.0

    buffer = np.zeros((0,), dtype=np.float32)
    chunk_idx = 0

    # 使用小块读取，累积到 chunk_samples
    blocksize = 1024

    print("[client] recording... Ctrl+C to stop")
    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=args.device,
        ) as stream:
            while True:
                data, _ = stream.read(blocksize)  # shape (blocksize, 1)
                buffer = np.concatenate([buffer, data[:, 0]])

                while buffer.shape[0] >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]

                    wav_bytes = wav_bytes_from_float32(chunk, sr)

                    send_offset = None
                    if args.mode == "client_offset":
                        send_offset = t_offset

                    t0 = time.time()
                    resp = post_chunk(
                        args.base_url,
                        session_id,
                        wav_bytes,
                        filename=f"chunk_{chunk_idx:06d}.wav",
                        t_offset=send_offset,
                        timeout=args.timeout,
                    )
                    dt = time.time() - t0

                    # 打印本次新增 segments（只打印 text，便于观察）
                    seg_texts = [s["text"] for s in resp.get("segments", []) if s.get("text")]
                    joined = " | ".join(seg_texts) if seg_texts else "(no speech)"
                    print(
                        f"[client] chunk={resp['chunk_index']} "
                        f"offset={resp['t_offset']:.3f} dur={resp['duration']:.3f} "
                        f"rt={dt:.2f}s -> {joined}"
                    )

                    chunk_idx += 1
                    if args.mode == "client_offset":
                        t_offset += args.chunk_sec

    except KeyboardInterrupt:
        print("\n[client] stopped")


if __name__ == "__main__":
    main()
