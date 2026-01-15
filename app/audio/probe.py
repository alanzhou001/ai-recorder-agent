# app/audio/probe.py
import subprocess


def probe_duration_seconds(path: str) -> float:
    """
    使用 ffprobe 读取音频时长（秒）。
    需要系统安装 ffmpeg/ffprobe。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)
