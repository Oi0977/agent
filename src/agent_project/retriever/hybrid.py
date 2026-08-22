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

from agent_project.path_manager import PathManager
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


# ========== 多文档检索(SPEC-005)==========

# 向量路相似度地板:cosine 低于此值视为"无语义信号",不参与融合排名(卫生项)。
# 单文档 hybrid_search(SPEC-002 契约)不受影响。
VEC_FLOOR = 0.35

# 全局 BM25 库缓存:键 = 排序后的全部 meta 路径元组(入库集合变了自动重建)
_global_lib_cache = {}


def _get_global_lib(pairs):
    """
    跨文档全局 BM25 库:合并所有文档的块统一建词法索引。

    为什么必须全局:rank_bm25 的 IDF 是**库内**统计 —— 各库分别建 BM25 时,
    3 块小库里 0.09 分的噪音匹配也拿满"第1名",与 465 块大库 13.2 分的第1名
    在 RRF 里同权(SPEC-005 修订记录:实测弱匹配双路双计后反压真结果)。
    向量分数同一嵌入模型天然跨库可比,可各库分路;BM25 分数不跨库可比,必须全局。
    """
    key = tuple(sorted(m for _, m in pairs))
    if key not in _global_lib_cache:
        import json
        from rank_bm25 import BM25Okapi
        chunks, keys = [], []          # keys[i] = (doc_id, chunk_idx)
        for doc_id, (_, mpath) in enumerate(pairs):
            with open(mpath, encoding="utf-8") as f:
                d = json.load(f)
            for i, c in enumerate(d["chunks"]):
                chunks.append(c)
                keys.append((doc_id, i))
        corpus = [_tokenize(c) for c in chunks]
        _global_lib_cache[key] = {
            "bm25": BM25Okapi(corpus), "keys": keys, "chunks": chunks,
        }
    return _global_lib_cache[key]


def discover_docs(output_dir=None) -> list[tuple[str, str]]:
    """
    扫描 output 目录,返回 [(index_path, meta_path)] 产物对。

    只认"成对存在"的(.index, .json) —— 缺伴的 .index 是残骸,不参与检索。
    按文件名排序保证 doc_id 分配稳定(同名覆盖重入库后顺序不变)。
    """
    d = Path(output_dir) if output_dir else PathManager().OUTPUT_DIR
    pairs = []
    for idx in sorted(d.glob("*.index")):
        meta = idx.with_suffix(".json")
        if meta.exists():
            pairs.append((str(idx), str(meta)))
    return pairs


def hybrid_search_all(query, k=20, n_per_route=10, output_dir=None):
    """
    多文档混合检索(D+1 路 RRF 融合):

    - 向量路 × D(每文档一路):余弦分出自同一嵌入模型,跨库可比,各库各取前 n
      (cosine < VEC_FLOOR 的弱匹配不参与排名)
    - BM25 路 × 1(全局一路):合并全部文档块统一建库 —— IDF 必须全局统计,
      否则小库的噪音匹配也拿满"第1名"排名分、与真结果同权(SPEC-005 修订记录)

    候选 key = (doc_id, chunk_idx) 元组 —— _rrf_fuse 只要求可哈希,原样复用;
    命中的 meta.source 记录来源文档文件名,下游可区分。

    :param k: 融合后保留的候选数
    :param n_per_route: 各路取前 n 参与融合
    :param output_dir: 知识库目录(默认 PathManager.OUTPUT_DIR)
    """
    pairs = discover_docs(output_dir)
    if not pairs:
        raise FileNotFoundError(
            "知识库为空:请先 python -m agent_project.ingest <文件> 入库")

    rank_lists = []
    for doc_id, (ipath, mpath) in enumerate(pairs):
        vec_hits = [h for h in search(query, ipath, mpath, k=n_per_route)
                    if h["score"] >= VEC_FLOOR]
        rank_lists.append([(doc_id, h["meta"]["chunk_idx"]) for h in vec_hits])

    # 全局 BM25 一路:统一 IDF,弱词法匹配沉到长尾(排名贡献趋近于零)
    glib = _get_global_lib(pairs)
    g_scores = glib["bm25"].get_scores(_tokenize(query))
    g_order = np.argsort(g_scores)[::-1][:n_per_route]
    rank_lists.append([glib["keys"][int(i)] for i in g_order if g_scores[i] > 0])

    fused = _rrf_fuse(rank_lists)
    order = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    libs = [None] * len(pairs)  # 惰性取库缓存:只给真正进入 top-k 的文档取
    hits = []
    for (doc_id, idx), rrf in order:
        if libs[doc_id] is None:
            libs[doc_id] = _get_lib(pairs[doc_id][1])
        lib = libs[doc_id]
        chunks, metas = lib["chunks"], lib["metas"]
        if 0 <= idx < len(chunks):
            hits.append({
                "chunk": chunks[idx],
                "meta": metas[idx],        # meta.source = 来源文档文件名
                "score": rrf,
                "rrf_score": rrf,
            })
    return hits
