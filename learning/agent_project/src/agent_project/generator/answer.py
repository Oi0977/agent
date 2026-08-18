# -*- coding: utf-8 -*-
"""
阶段4·生成 —— RAG 完整问答(rag_answer)。

把检索(混合粗排 + 精排)与生成串成完整链路:
    问题 → hybrid_search 混合粗排 top-recall_k(向量+BM25 两路,RRF 融合)
         → rerank 精排 top-k(交叉编码器)
         → 拼 prompt → LLM 生成答案
检索链路:语义泛化(向量)+ 字面精确(BM25)互补召回,交叉编码器精选。
"""
from agent_project.generator.llm_client import chat
from agent_project.generator.prompt import build_prompt
from agent_project.retriever import hybrid_search
from agent_project.reranker import rerank


def rag_answer(question: str, index_path, meta_path, k: int = 3, recall_k: int = 20) -> dict:
    """
    RAG 完整链路:混合粗排 + 精排 + 生成。
    :param question: 用户问题
    :param index_path: build_index 落盘的 .index(向量库)
    :param meta_path: build_index 落盘的 .json(原文+元数据)
    :param k: 最终喂给 LLM 的块数(精排后保留)
    :param recall_k: 粗排融合后保留的候选数(两路各取前 recall_k 参与融合)
    :return: {"answer": LLM 答案, "hits": 精排后的命中(含 rerank_score)}
    """
    hits = hybrid_search(question, index_path, meta_path, k=recall_k)  # 粗排:两路融合
    hits = rerank(question, hits, top_k=k)                             # 精排:慢而准
    prompt = build_prompt(question, hits)                              # 拼装:块 + 问题 → prompt
    answer = chat(prompt)                                              # 生成:prompt → 答案
    return {"answer": answer, "hits": hits}
