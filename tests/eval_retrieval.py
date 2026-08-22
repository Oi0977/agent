# -*- coding: utf-8 -*-
"""
SPEC-008 检索基线评测:三配置对比(纯向量 / 混合RRF / 混合+精排)。

    python tests/eval_retrieval.py

指标:hit@3(线上 rag_answer/agent 用 k=3,对齐)+ MRR;按题型分桶。
相关判定:块原文含金标关键词任一(见 eval_metrics)。
确定性:FAISS 精确检索 + cross-encoder 推理均无随机性,同库重跑数字一致。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from eval_metrics import hit_at_k, mean, reciprocal_rank  # noqa: E402


def vector_only(query, pairs, k=3, pool=5):
    """纯向量全局检索:各文档 top-pool 按余弦分合并(同一嵌入模型,分数跨文档可比)。"""
    from agent_project.retriever.searcher import search
    hits = []
    for ip, mp in pairs:
        hits.extend(search(query, ip, mp, k=pool))
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def main():
    from agent_project.retriever.hybrid import discover_docs, hybrid_search_all
    from agent_project.reranker import rerank

    golden = json.loads((HERE / "eval" / "golden.json").read_text(encoding="utf-8"))
    questions = golden["questions"]
    print(f"金标集:{len(questions)} 题(literal={sum(1 for q in questions if q['type']=='literal')}"
          f" / semantic={sum(1 for q in questions if q['type']=='semantic')})")

    pairs = discover_docs()
    if not pairs:
        sys.exit("知识库为空:先 python -m agent_project.ingest <文件>")
    print(f"知识库:{len(pairs)} 份文档\n")

    K = 3
    results = {name: [] for name in ("vector", "hybrid", "hybrid+rerank")}
    bucket = {(t, name): [] for t in ("literal", "semantic")
              for name in results}

    header = f"{'id':<5}{'题型':<10}{'hit@3(向量/混合/精排)':<24}{'MRR(向量/混合/精排)'}"
    print(header)
    print("-" * len(header))
    for q in questions:
        kws, typ = q["keywords"], q["type"]
        vec = [h["chunk"] for h in vector_only(q["question"], pairs, k=K)]
        hyb = [h["chunk"] for h in hybrid_search_all(q["question"], k=10)[:K]]
        rr = [h["chunk"] for h in rerank(q["question"], hybrid_search_all(q["question"], k=10), top_k=K)]

        hits, rrs = [], []
        for name, chunks in (("vector", vec), ("hybrid", hyb), ("hybrid+rerank", rr)):
            h, r = hit_at_k(chunks, kws, K), reciprocal_rank(chunks, kws)
            results[name].append((h, r))
            bucket[(typ, name)].append((h, r))
            hits.append(h)
            rrs.append(r)
        print(f"{q['id']:<5}{typ:<10}{hits[0]:.0f} / {hits[1]:.0f} / {hits[2]:.0f}"
              f"{'':<14}{rrs[0]:.2f} / {rrs[1]:.2f} / {rrs[2]:.2f}")

    print("\n===== 基线汇总(k=3)=====")
    summary = {}
    for name in results:
        h = mean([x[0] for x in results[name]])
        m = mean([x[1] for x in results[name]])
        summary[name] = (h, m)
        print(f"{name:<15} hit@3 = {h:.2%}   MRR = {m:.3f}")
    for t in ("literal", "semantic"):
        line = f"  · {t:<9}"
        for name in results:
            line += f" {name}={mean([x[0] for x in bucket[(t, name)]]):.0%}"
        print(line + "(hit@3)")

    dv, dh = summary["vector"][0], summary["hybrid"][0]
    print(f"\n混合检索较纯向量 hit@3:{(dh - dv):+.0%}"
          f"(MRR:{summary['hybrid'][1] - summary['vector'][1]:+.3f})")
    return summary


if __name__ == "__main__":
    main()
