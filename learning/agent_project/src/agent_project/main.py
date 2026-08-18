# -*- coding: utf-8 -*-
"""
文档解析入口（演示用）
parse_document 的实现已收拢到 preprocessor 包，这里只负责调用与展示。
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"S:\huggingface_cache"
from agent_project.preprocessor import parse_document
from agent_project.chunker.step_1_chunking import simple_chunk_text
from agent_project.chunker.smart_chunking import smart_chunk_text
from agent_project.embedder import embed_texts

# ========== 运行测试 ==========
if __name__ == "__main__":
    # 替换为你的文档路径
    file_path = r"S:\学习资料\Wireshark数据包分析实战(第3版)-2018-中文版.pdf"  # 或 .md
    cleaned_text = ""

    try:
        cleaned_text = parse_document(file_path)
        print("=" * 50)
        print("清洗后的纯文本内容：")
        print("=" * 50)
        print(cleaned_text)
        print("=" * 50)
        print(f"总字符数: {len(cleaned_text)}")
    except Exception as e:
        print(f"解析失败: {e}")

    # 测试文本分块
    chunk_size = 500
    overlap = 50
    chunks_simple = simple_chunk_text(cleaned_text, chunk_size, overlap)
    print("=" * 50)
    for i, chunk in enumerate(chunks_simple):
        print(f"块 {i + 1} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 50)

    # 测试智能分块
    chunks_smart = smart_chunk_text(cleaned_text, chunk_size, overlap)
    print("=" * 50)
    for i, chunk in enumerate(chunks_smart):
        print(f"块 {i + 1} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 50)

    print(f"总块数 (简单分块): {len(chunks_simple)}")
    print(f"总块数 (智能分块): {len(chunks_smart)}")

    # 阶段二验收：向量嵌入（智能分块结果）
    print("\n>>> 向量嵌入...")
    embeddings = embed_texts(chunks_smart)
    print(f"向量数量: {len(embeddings)}, 维度: {len(embeddings[0])}")

    # ========== 阶段三：检索（建库 + 检索）==========
    from agent_project.retriever import build_index, search

    print("\n" + "=" * 50)
    print(">>> 阶段三：建库（FAISS）...")
    index_path, meta_path, n_chunks = build_index(file_path)
    print(f"建库完成: {n_chunks} 个块")
    print(f"  向量库(.index): {index_path}")
    print(f"  原文+元数据(.json): {meta_path}")

    print("\n>>> 阶段三：检索（top-3）...")
    queries = [
        "Wireshark 解密 HTTPS",                       # 字面重合
        "怎么把浏览器加密的网页流量还原成明文",          # 语义相关 / 字面不重合
    ]
    for q in queries:
        print("-" * 50)
        print(f"query: {q}")
        for hit in search(q, index_path, meta_path, k=3):
            preview = hit["chunk"][:60].replace("\n", " ")
            print(f"  [score={hit['score']:.4f}] {preview}...")

    # ========== 阶段四:生成(RAG 问答,含精排)==========
    from agent_project.generator import rag_answer

    print("\n" + "=" * 50)
    print(">>> 阶段四:RAG 问答(粗排召回 + 精排重排 + 生成)...")
    q = "Wireshark 里怎么解密 HTTPS 流量?"
    result = rag_answer(q, index_path, meta_path, k=3, recall_k=20)
    print(f"问题: {q}")
    print("-" * 50)
    print("精排后引用的文档块:")
    for i, hit in enumerate(result["hits"], 1):
        preview = hit["chunk"][:50].replace("\n", " ")
        print(f"  【{i}】rerank={hit['rerank_score']:.4f} | {preview}...")
    print("-" * 50)
    print("回答:")
    print(result["answer"])