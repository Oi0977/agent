# -*- coding: utf-8 -*-
"""
preprocessor 包入口（薄门面）：只暴露公开接口，实现在 loader.py。
"""
from .loader import parse_document

__all__ = ["parse_document"]
