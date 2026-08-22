# -*- coding: utf-8 -*-
"""
阶段3·检索 —— 在线检索(search)。

给一个问题,返回 top-k 最相关的文档块。流程:
  读 FAISS 向量库 + 读原文 JSON
    → query 嵌入(自动加 BGE 前缀)
    → FAISS 搜出 (相似度, 向量id)
    → 用 id 回查 JSON 里的原文/元数据。

关键认知:FAISS 的 search 只返回 id,不返回原文;
"id 怎么映射回那段文字"是这里自己做的 —— 这正是 FAISS "只存向量" 的本色。
"""
import json

import faiss
import numpy as np

from agent_project.embedder import embed_query


def search(query, index_path, meta_path, k=3):
    """
    :param query: 用户提问
    :param index_path: build_index 产出的 .index 路径(向量库)
    :param meta_path:  build_index 产出的 .json 路径(原文+元数据)
    :param k: 取前 k 个最相关块
    :return: [{"chunk":原文, "meta":元数据, "score":相似度}, ...] 按相似度降序
    """
    # 1. 读两份产物:向量库(FAISS) + 原文/元数据(JSON)
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks, metas = data["chunks"], data["metas"]

    # 2. query 嵌入(自动加 BGE 前缀)→ float32
    q_vec = np.asarray(embed_query(query), dtype="float32")

    # 3. FAISS 搜索:返回 (scores, ids),形状均为 (1, k)
    scores, ids = index.search(q_vec, k)

    # 4. 用 id 回查原文/元数据。
    #    id == -1 表示 k 超过了库大小,FAISS 用 -1 占位,需过滤掉。
    results = []
    for i, s in zip(ids[0], scores[0]):
        i = int(i)
        if 0 <= i < len(chunks):
            results.append({"chunk": chunks[i], "meta": metas[i], "score": float(s)})
    return results
