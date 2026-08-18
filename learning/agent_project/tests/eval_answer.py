# -*- coding: utf-8 -*-
"""
SPEC-008 Should —— LLM-as-judge:rag_answer 答案忠实度评分(1-5,只依据资料)。

    python tests/eval_answer.py --limit 5
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

JUDGE_PROMPT = """你是评审。根据【检索资料】判断【回答】的忠实度,只依据资料、不依据你自己的知识。
评分 1-5:5=完全有资料支撑;3=部分支撑;1=基本无支撑/编造。
只输出一行:分数 + 一句理由。

【检索资料】
{context}

【回答】
{answer}
"""


def main(limit=5):
    from agent_project.generator import rag_answer
    from agent_project.generator.llm_client import chat
    from agent_project.retriever.hybrid import discover_docs

    golden = json.loads((HERE / "eval" / "golden.json").read_text(encoding="utf-8"))
    pairs = discover_docs()
    if not pairs:
        sys.exit("知识库为空,先 ingest")
    # 主文档 = 块数最多的那份(金标问题以它为准)
    pair = max(pairs, key=lambda p: len(json.load(open(p[1], encoding="utf-8"))["chunks"]))

    scores = []
    for q in golden["questions"][:limit]:
        r = rag_answer(q["question"], pair[0], pair[1], k=3)
        ctx = "\n---\n".join(h["chunk"][:300] for h in r["hits"])
        resp = chat([{"role": "user",
                      "content": JUDGE_PROMPT.format(context=ctx, answer=r["answer"])}],
                    temperature=0.1)
        out = (resp.choices[0].message.content or "").strip()
        first = out.splitlines()[0] if out else "0 无法解析"
        try:
            s = int(first[0])
        except (ValueError, IndexError):
            s = 0
        scores.append(s)
        print(f'{q["id"]}  {s}  | {first[:70]}')
    print(f"\n忠实度均分:{sum(scores) / len(scores):.1f}(n={len(scores)},1-5)")


if __name__ == "__main__":
    limit = 5
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    main(limit)
