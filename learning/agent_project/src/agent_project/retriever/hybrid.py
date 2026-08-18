# -*- coding: utf-8 -*-
"""
混合检索 —— 向量(语义) + BM25(字面) 两路召回,RRF 融合。

为什么需要两路:
  - 向量检索只认"语义相近":换种说法也搜得到;但对专有名词/函数名/配置项等
    **字面精确匹配**类查询,含该术语的块可能因整体语义不贴近而落榜
  - BM25 全文检索只认"词项重合":字面精确匹配强;但"换种说法"完全搜不到
  - 两路互补 → RRF(Reciprocal Rank Fusion,倒数排名融合)合并各自排名

RRF 裸机制: score(d) = Σ_各路 1/(k_rrf + rank_i(d))
  - 只用**排名**,不用原始分 —— 向量余弦和 BM25 分数量纲不同,排名才可比
  - k_rrf=60 是业界标准值:越大,单路冠军的优势越被"削平"(融合越民主)
  - 例:块X 在路A第1、路B第2 → 1/61 + 1/61 ≈ 0.0328,两路都认可,分最高
"""
from pathlib import Path

import numpy as np

from agent_project.retriever.searcher import search

K_RRF = 60

# ---- 库级缓存:同一份 .json 的 chunks/metas/BM25 只构建一次(SPEC-002 AC6) ----
_lib_cache = {}  # {meta_path_str: {"chunks", "metas", "bm25", "corpus"}}


def _tokenize(text: str) -> list[str]:
    """jieba 分词 + 小写归一 + 滤掉纯标点 token。

    关键约束:查询与文档必须用**同一个**分词器 —— BM25 按词项匹配,
    两边切法不同就等于各说各话。小写归一让 PcapNg/pcapng 落到同一词项。
    """
    import jieba
    tokens = jieba.lcut(text.lower())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def _get_lib(meta_path):
    """按 meta_path 缓存:首次加载 .json + 全库分词 + 建 BM25;之后全走缓存。"""
    key = str(Path(meta_path).resolve())
    if key not in _lib_cache:
        import json
        from rank_bm25 import BM25Okapi
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        chunks, metas = data["chunks"], data["metas"]
        corpus = [_tokenize(c) for c in chunks]
        _lib_cache[key] = {
            "chunks": chunks, "metas": metas,
            "bm25": BM25Okapi(corpus), "corpus": corpus,
        }
    return _lib_cache[key]


def _bm25_search(query, meta_path, n):
    """BM25 单路:返回 [(chunk_idx, bm25分)] 降序前 n。

    0 分(词项完全不重合)的块不参与 —— 进了排名也只是噪音。
    """
    lib = _get_lib(meta_path)
    scores = lib["bm25"].get_scores(_tokenize(query))
    order = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def _rrf_fuse(rank_lists: list[list], k_rrf: int = K_RRF) -> dict:
    """
    RRF 融合(纯函数,tests/test_rrf.py 的测试对象)。
    :param rank_lists: 每路一个 [候选id, ...],各自按相关性降序
    :return: {候选id: rrf_score}
    """
    scores = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return scores


def hybrid_search(query, index_path, meta_path, k=20, n_per_route=None):
    """
    混合检索:向量 + BM25 两路召回,RRF 融合,输出与 search() 兼容。

    :param query: 用户问题
    :param index_path / meta_path: build_index 落盘的两份产物
    :param k: 融合后保留的候选数(rag_answer 里作为粗排输出,= recall_k)
    :param n_per_route: 每路各自取前 n 参与融合(默认 = k)
    :return: [{"chunk", "meta", "score"(=rrf,下游兼容), "rrf_score",
               "vector_score"(未入向量top为 None), "bm25_score"(同理)}, ...]
    """
    if n_per_route is None:
        n_per_route = k

    # 路1:向量(语义) —— 复用现有 search(),拿 id 与余弦分
    vec_hits = search(query, index_path, meta_path, k=n_per_route)
    vec_ids = [h["meta"]["chunk_idx"] for h in vec_hits]
    vec_map = {h["meta"]["chunk_idx"]: h["score"] for h in vec_hits}

    # 路2:BM25(字面)
    bm25_pairs = _bm25_search(query, meta_path, n=n_per_route)
    bm25_ids = [i for i, _ in bm25_pairs]
    bm25_map = dict(bm25_pairs)

    # RRF 融合两路排名 → 按融合分降序取前 k
    fused = _rrf_fuse([vec_ids, bm25_ids])
    order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    # 用 id 回查原文(与 searcher.search 同款机制,只是 chunks 走缓存)
    lib = _get_lib(meta_path)
    chunks, metas = lib["chunks"], lib["metas"]

    hits = []
    for idx, rrf in order:
        if 0 <= idx < len(chunks):
            hits.append({
                "chunk": chunks[idx],
                "meta": metas[idx],
                "score": rrf,                      # 下游兼容:rerank/prompt 读这个排序分
                "rrf_score": rrf,
                "vector_score": vec_map.get(idx),   # 来源分:该路原始分,未入该路 top 为 None
                "bm25_score": bm25_map.get(idx),
            })
    return hits
