# -*- coding: utf-8 -*-
"""
reranker 包入口(薄门面):只暴露公开接口。
  rerank —— 对初检结果精排(cross-encoder 重排序)
"""
from .reranker import rerank

__all__ = ["rerank"]
