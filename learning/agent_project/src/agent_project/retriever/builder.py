# -*- coding: utf-8 -*-
"""
阶段3·检索 —— 离线建库(build_index)。

把一份文档变成可检索的向量库,落盘成两份独立文件:
  - <stem>.index : FAISS 向量库(只存向量,不认原文)
  - <stem>.json  : 原文 + 元数据(chunks / metas),与向量靠"下标 id"一一对应

关键认知:FAISS 本身不知道某条向量代表哪段文字,这里显式维护一份 JSON
作为"id → 原文"的映射。两份产物各自独立,检索时用 FAISS 返回的 id 回查 JSON。
"""
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from agent_project.preprocessor import parse_document
from agent_project.chunker.smart_chunking import smart_chunk_text
from agent_project.embedder import embed_texts
from agent_project.path_manager import PathManager

# bge-small-zh-v1.5 固定输出 512 维向量
EMBED_DIM = 512


def _ascii_name(stem: str) -> str:
    """
    把文档名转成 ASCII 安全的产物文件名。

    原因:Windows 上 faiss.write_index 走 C 标准库 fopen,不支持中文等非 ASCII 文件名,
    会报 "Illegal byte sequence"。所以 .index 文件名必须是纯 ASCII。
    做法:保留 stem 中的 ASCII 字符,再补一个短哈希避免不同中文文档撞名;
    原始中文文件名仍完整记录在 .json 的 source 字段里,可追溯。
    """
    ascii_part = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_"))
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_part or 'doc'}_{digest}"


def build_index(file_path, chunk_size=500, chunk_overlap=50, output_dir=None):
    """
    解析 → 分块 → 嵌入 → 建 FAISS 库 → 落盘。

    :param file_path: 待建库的文档路径(.pdf / .md)
    :param chunk_size: 分块大小(字符)
    :param chunk_overlap: 块间重叠(字符)
    :param output_dir: 产物目录(默认 PathManager.OUTPUT_DIR;测试可注入临时目录)
    :return: (index_path, meta_path, 块数)
    """
    # 1. 解析 + 分块
    text = parse_document(file_path)
    chunks = smart_chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"文档解析/分块后无任何文本块: {file_path}")

    # 2. 嵌入(passage 不加前缀)→ float32 连续数组(FAISS 强制要求 float32)
    vecs = np.asarray(embed_texts(chunks), dtype="float32")

    # 3. 建 FAISS 索引:IndexFlatIP = 内积。
    #    embedder 已做 L2 归一化,故内积 = 余弦相似度,值域 [-1, 1],越大越相似。
    index = faiss.IndexFlatIP(EMBED_DIM)
    # FAISS 要求输入为 C-contiguous 的 float32 二维数组
    vecs = np.ascontiguousarray(vecs, dtype="float32")
    index.add(vecs)   # 只把向量喂给 FAISS —— 它只认向量

    # 4. 元数据:每块来自哪个文件、第几块(将来按来源过滤、拼接上下文用)
    source_name = Path(file_path).name
    metas = [{"source": source_name, "chunk_idx": i} for i in range(len(chunks))]

    # 5. 落盘:向量 → FAISS 的 .index;原文+元数据 → JSON。两份各自独立,靠下标 id 连接。
    #    文件名用 ASCII:FAISS 的 write_index 在 Windows 不支持中文文件名(见 _ascii_name)。
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        pm = PathManager()
        pm.init_all_dirs()
        out = pm.OUTPUT_DIR
    name = _ascii_name(Path(file_path).stem)
    index_path = out / f"{name}.index"
    meta_path = out / f"{name}.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metas": metas}, f, ensure_ascii=False, indent=2)

    return index_path, meta_path, len(chunks)
