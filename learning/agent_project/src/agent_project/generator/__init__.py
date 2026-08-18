# -*- coding: utf-8 -*-
"""
generator 包入口(薄门面):只暴露公开接口。
  rag_answer —— RAG 完整问答(检索 + 生成)
"""
from .answer import rag_answer

__all__ = ["rag_answer"]
