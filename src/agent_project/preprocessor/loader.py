# -*- coding: utf-8 -*-
"""
文档加载调度：按文件类型分派到对应解析器，再串接清洗。
具体解析/清洗实现见 document_parser 子包，本模块只负责编排。
"""
from pathlib import Path

from .document_parser.pdf_parser import parse_pdf
from .document_parser.md_parser import parse_md
from .document_parser.text_cleaner import clean_text


def parse_document(file_path: str) -> str:
    """根据文件扩展名自动选择解析器，返回清洗后的纯文本"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        raw_text = parse_md(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    return clean_text(raw_text)
