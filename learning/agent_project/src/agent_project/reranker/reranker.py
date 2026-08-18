# -*- coding: utf-8 -*-
"""
重排序 —— Cross-Encoder 精排(rerank)。

为什么需要重排:嵌入检索(双塔)和重排(交叉编码器)是两种精度/速度取舍:
  - 双塔(bi-encoder,如 BGE 嵌入模型):query 和 passage 各自独立编码成向量,
    再算余弦相似度。passage 向量可离线预计算,检索极快(毫秒级扫全库);
    但两边编码时互相"看不见",query 里的词无法和文档里的词直接交互,精度有上限。
    → 适合粗排:从几百块里快速召回 top-20
  - 交叉编码器(cross-encoder,如 BGE reranker):query 和 passage 拼接后
    整体过模型,逐 token 交叉注意,直接输出这个 (query, passage) 对的相关性分数。
    精度显著更高;但必须查询时逐对实时计算,无法预计算,慢。
    → 适合精排:把 top-20 重新打分,精选 top-3

两段式设计("快而广的召回 + 慢而准的重排")是业界 RAG 的标配链路。
"""
_model = None


def _resolve_model_path() -> str:
    """
    同 encoder.py 的做法:优先解析本地快照目录,彻底离线加载。

    HuggingFace 缓存结构:
      S:/huggingface_cache/models--BAAI--bge-reranker-base/snapshots/<hash>/
    """
    from pathlib import Path
    base = Path(r"S:\huggingface_cache\models--BAAI--bge-reranker-base\snapshots")
    if base.exists():
        snaps = sorted(base.iterdir())
        if snaps:
            return str(snaps[0])
    return "BAAI/bge-reranker-base"  # 无本地缓存时退回 repo-id(需联网)


def _get_model():
    """懒加载 CrossEncoder 单例(首次调用才加载,之后复用)。"""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_resolve_model_path())
    return _model


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """
    对初检结果精排:用 cross-encoder 逐对打分,按相关性降序取 top_k。

    :param query: 用户问题
    :param hits: search() 返回的初检结果 [{"chunk", "meta", "score"}, ...]
    :param top_k: 精排后保留的块数
    :return: 精排后的 hits(新增 "rerank_score" 字段,0~1,越大越相关;
             原 "score" 保留,便于对照粗排/精排的差异)
    """
    if not hits:
        return []

    # 1. 组装 (query, chunk) 对:cross-encoder 的输入是"问题+候选"拼在一起
    pairs = [[query, hit["chunk"]] for hit in hits]

    # 2. 逐对打分:模型输出原始 logits(可正可负,越大越相关)
    model = _get_model()
    raw_scores = model.predict(pairs)

    # 3. sigmoid 归一化到 0~1,便于人读(不改变排序,只改变数值尺度)
    import numpy as np
    probs = 1 / (1 + np.exp(-np.asarray(raw_scores)))

    # 4. 按精排分数降序取 top_k;保留原双塔余弦分,方便观察两种模型的分歧
    ranked = sorted(zip(hits, probs), key=lambda x: x[1], reverse=True)
    results = []
    for hit, p in ranked[:top_k]:
        results.append({
            "chunk": hit["chunk"],
            "meta": hit["meta"],
            "score": hit["score"],       # 双塔余弦分(粗排依据)
            "rerank_score": float(p),     # 交叉编码器分(精排依据)
        })
    return results
