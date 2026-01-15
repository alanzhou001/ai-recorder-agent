# app/summarizer/merger.py
from __future__ import annotations

from typing import Dict, List


def _overlap(a: str, b: str, max_len: int = 30) -> int:
    """
    返回 a 的后缀与 b 的前缀最大重叠长度
    """
    a = a.strip()
    b = b.strip()
    n = min(len(a), len(b), max_len)
    for k in range(n, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def merge_segments(segments: List[Dict], min_text_len: int = 2) -> List[Dict]:
    """
    简单合并：
    - 把相邻短 segment 合并成一句（同一 chunk/相邻时间）
    - 跨段做重复前后缀消除
    """
    if not segments:
        return []

    out: List[Dict] = []
    for seg in segments:
        seg = dict(seg)
        seg["text"] = str(seg.get("text", "")).strip()
        if not seg["text"]:
            continue

        if not out:
            out.append(seg)
            continue

        prev = out[-1]
        # 若上一个非常短，且时间连续，直接拼接
        if len(prev["text"]) <= min_text_len and seg["start"] - prev["end"] < 0.6:
            prev["text"] = (prev["text"] + seg["text"]).strip()
            prev["end"] = max(prev["end"], seg["end"])
            continue

        # 去重：prev 后缀与 seg 前缀重叠
        k = _overlap(prev["text"], seg["text"])
        if k > 0:
            seg["text"] = seg["text"][k:].lstrip()

        out.append(seg)

    # 再过滤一次空文本
    return [s for s in out if s.get("text")]
