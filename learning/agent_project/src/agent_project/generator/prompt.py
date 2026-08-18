# -*- coding: utf-8 -*-
"""
阶段4·生成 —— Prompt 模板。

RAG 生成的核心动作:把检索到的文档块拼成"参考资料",和问题一起喂给 LLM。
模板单独成文件:prompt 是 RAG 效果最直接的旋钮,值得独立出来反复调。
"""

PROMPT_TEMPLATE = """你是一个严谨的技术文档问答助手。请只依据下面的参考资料回答问题,不要使用资料之外的知识。

## 参考资料
{context}

## 问题
{question}

## 回答要求
- 只基于参考资料作答,资料里没有的不要编造
- 如果参考资料不足以回答,直接说明"参考资料中没有相关内容"
- 条理清晰,涉及步骤时按顺序分点说明
"""


def build_prompt(question: str, hits: list[dict]) -> str:
    """
    把 search() 的命中结果拼成完整 prompt。
    :param question: 用户问题
    :param hits: search() 返回的 [{"chunk":..., "meta":..., "score":...}, ...]
    :return: 完整 prompt 字符串
    """
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["meta"]
        # 精排后的结果带 rerank_score(0~1),粗排结果只有向量相似度
        if "rerank_score" in hit:
            rel = f"相关性 {hit['rerank_score']:.3f}"
        else:
            rel = f"相似度 {hit['score']:.3f}"
        header = f"【资料{i}】(来源: {meta['source']},第 {meta['chunk_idx']} 块,{rel})"
        parts.append(f"{header}\n{hit['chunk']}")
    context = "\n\n".join(parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)
