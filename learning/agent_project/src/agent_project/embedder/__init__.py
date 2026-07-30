# -*- coding: utf-8 -*-
"""
embedder 包入口（薄门面）：只暴露公开接口，实现在 encoder.py。
"""
from .encoder import embed_texts, embed_query

__all__ = ["embed_texts", "embed_query"]
