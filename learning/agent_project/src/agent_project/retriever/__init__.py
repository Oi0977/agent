# -*- coding: utf-8 -*-
"""
retriever 包入口(薄门面):只暴露公开接口。
  build_index —— 离线建库(文档 → 向量库)
  search      —— 在线检索(问题 → top-k 文档块)
"""
from .builder import build_index
from .searcher import search

__all__ = ["build_index", "search"]
