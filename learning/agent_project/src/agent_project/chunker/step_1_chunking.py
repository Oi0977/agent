
# -*- coding: utf-8 -*-
def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    将文本分割为多个块，每块的大小为 chunk_size，块之间有 overlap 的重叠部分。
    
    :param text: 待分割的文本
    :param chunk_size: 每个块的最大字符数
    :param overlap: 块之间的重叠字符数
    :return: 分割后的文本块列表
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 必须大于或等于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    # 1. 防御性检查：防止死循环
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})，否则步长为0导致死循环。")
    
    text_length = len(text)
    if text_length <= chunk_size:
        return [text]
    
    chunks = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        
        if end == text_length:
            break
        
        start += chunk_size - overlap
    return chunks