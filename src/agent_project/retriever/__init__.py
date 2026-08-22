# -*- coding: utf-8 -*-
"""
retriever 包入口(薄门面):只暴露公开接口。
  build_index    —— 离线建库(文档 → 向量库)
  search         —— 在线向量检索(单路,语义)
  hybrid_search  —— 在线混合检索(向量 + BM25 + RRF,两路融合)
"""
from .builder import build_index
from .searcher import search
from .hybrid import hybrid_search

__all__ = ["build_index", "search", "hybrid_search"]
