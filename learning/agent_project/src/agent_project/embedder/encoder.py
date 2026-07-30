# -*- coding: utf-8 -*-
"""
向量编码器(BGE bge-small-zh-v1.5)。

BGE 这类非对称双塔检索模型有一个关键约定:
- 编码 **文档块/语料(passage)** 时,不加任何前缀。
- 编码 **用户提问(query)** 时,必须拼接官方 query instruction 前缀,
  才能和 passage 落在同一个可比的语义空间里算余弦相似度。

所以对外暴露两个语义明确的函数:
    embed_texts  —— 建库用(编码 passage,不加前缀)
    embed_query  —— 检索用(编码 query,自动加前缀)
底层共享一个私有 _encode,干活只此一份。
"""
_model = None

# BGE 官方要求:s2p(short query to long passage)检索时给 query 加这句前缀,passage 不加。
# 原文见模型卡 Model List 的 "query instruction for retrieval"。
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _get_model():
    """懒加载模型单例(首次调用才下载/加载,之后复用)。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            'BAAI/bge-small-zh-v1.5',
            cache_folder=r"S:\huggingface_cache",
        )
    return _model


def _encode(texts: list[str], add_instruction: bool) -> list[list[float]]:
    """
    底层共享实现:实际干活的只有这一个函数。
    :param texts: 待编码文本列表
    :param add_instruction: 是否给每条文本拼接 query 前缀(仅 query 需要)
    :return: 归一化后的向量列表(L2 norm=1,点积即余弦相似度)
    """
    model = _get_model()
    if add_instruction:
        texts = [_QUERY_INSTRUCTION + t for t in texts]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    编码文档块/语料(passage)——建库时用,**不加前缀**。
    :param texts: 文档块列表
    :return: 向量列表
    """
    return _encode(texts, add_instruction=False)


def embed_query(query: str | list[str]) -> list[list[float]]:
    """
    编码用户提问(query)——检索时用,**自动加 BGE 前缀**。
    支持传入单条字符串或字符串列表,统一返回向量列表。
    :param query: 单条提问(str)或提问列表(list[str])
    :return: 向量列表
    """
    q = [query] if isinstance(query, str) else list(query)
    return _encode(q, add_instruction=True)
