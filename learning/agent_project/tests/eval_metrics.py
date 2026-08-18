# -*- coding: utf-8 -*-
"""
SPEC-008 检索指标(纯函数,可离线单测)。

- hit@k   : top-k 内存在相关块的问题占比(单题取 0/1)
- MRR     : 首个相关块排名的倒数(单题;无命中为 0)
- 相关判定: 块原文含任一关键词(大小写不敏感)——比 chunk_id 稳健,重建库不失效

自测(AC2):python tests/eval_metrics.py
"""
from pathlib import Path


def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    low = chunk_text.lower()
    return any(k.lower() in low for k in keywords)


def hit_at_k(ranked_chunks: list[str], keywords: list[str], k: int = 3) -> float:
    return 1.0 if any(is_relevant(c, keywords) for c in ranked_chunks[:k]) else 0.0


def reciprocal_rank(ranked_chunks: list[str], keywords: list[str]) -> float:
    for i, c in enumerate(ranked_chunks, start=1):
        if is_relevant(c, keywords):
            return 1.0 / i
    return 0.0


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


if __name__ == "__main__":
    # AC2 手算用例:排名 [无关, 相关, 无关] → hit@3=1.0, MRR=1/2=0.5
    chunks = ["苹果", "香蕉 牛奶", "橙子"]
    assert hit_at_k(chunks, ["香蕉"], k=3) == 1.0
    assert reciprocal_rank(chunks, ["香蕉"]) == 0.5
    # 无命中
    assert hit_at_k(chunks, ["不存在"], k=3) == 0.0
    assert reciprocal_rank(chunks, ["不存在"]) == 0.0
    # 首位命中
    assert reciprocal_rank(["相关词", "x", "y"], ["相关词"]) == 1.0
    # k 截断:相关块在第 4 位, hit@3 不算
    assert hit_at_k(["a", "b", "c", "香蕉"], ["香蕉"], k=3) == 0.0
    assert reciprocal_rank(["a", "b", "c", "香蕉"], ["香蕉"]) == 0.25
    print("AC2 ✓ hit@k / MRR 指标函数正确(手算用例全部一致)")
