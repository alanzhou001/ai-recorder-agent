# app/asr/text_postprocess.py
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any


_PUNCT_RE = re.compile(r"[。！？!?…]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.strip()
    s = _SPACE_RE.sub(" ", s)
    return s


def overlap_suffix_prefix(a: str, b: str, max_len: int = 30) -> int:
    """
    返回 a 的后缀与 b 的前缀最大重叠长度（字符级）
    """
    a = a.strip()
    b = b.strip()
    n = min(len(a), len(b), max_len)
    for k in range(n, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def dedup_join(prev: str, new: str) -> str:
    prev = normalize_text(prev)
    new = normalize_text(new)
    if not prev:
        return new
    if not new:
        return prev

    # 完全包含/相等
    if new == prev:
        return prev
    if new in prev:
        return prev
    if prev in new:
        return new

    k = overlap_suffix_prefix(prev, new, max_len=40)
    if k > 0:
        return (prev + new[k:]).strip()

    # 简单兜底：加空格拼接（中文也不影响）
    return (prev + " " + new).strip()


def split_into_final_and_partial(buf: str) -> Tuple[List[str], str]:
    """
    按句末标点切分：返回 (final_sentences, remaining_partial)
    """
    buf = normalize_text(buf)
    if not buf:
        return [], ""

    finals: List[str] = []
    start = 0
    for m in _PUNCT_RE.finditer(buf):
        end = m.end()
        sent = buf[start:end].strip()
        if sent:
            finals.append(sent)
        start = end

    partial = buf[start:].strip()
    return finals, partial


def merge_segments_text(segments: List[Dict[str, Any]]) -> str:
    """
    将 ASR segments 拼成一段文本（同 chunk 内）
    """
    parts = []
    for s in segments:
        t = normalize_text(str(s.get("text", "")))
        if t:
            parts.append(t)
    return " ".join(parts).strip()
