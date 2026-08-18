# -*- coding: utf-8 -*-
"""
最小 Agent 循环 —— 裸写 while + tool_calls,零框架依赖。

本质:Agent 不是魔法,就是一个 while 循环 + LLM 的 function calling 能力。
LLM 每轮输出两种可能之一:
  ① 返回 tool_calls → 说明"我要调某个工具" → 执行工具 → 结果回注 → 再来一轮
  ② 返回 content    → 说明"我有答案了"     → 返回最终回答

代码只需判断"有没有 tool_calls",有就执行并回注,没有就返回。
"""
import json
from pathlib import Path

from agent_project.generator.llm_client import chat

# ---------- 上下文目录:search 需要索引路径 ----------
_pm = None


def _get_pm():
    global _pm
    if _pm is None:
        from agent_project.path_manager import PathManager
        _pm = PathManager()
    return _pm


def _find_index():
    """自动发现 output 目录下已有的 .index/.json 对。"""
    pm = _get_pm()
    pm.init_all_dirs()
    indexes = list(pm.OUTPUT_DIR.glob("*.index"))
    if not indexes:
        raise FileNotFoundError("data/output/ 下没有 .index 文件,请先 build_index")
    idx = indexes[0]
    meta = idx.with_suffix(".json")
    if not meta.exists():
        raise FileNotFoundError(f"索引文件 {idx.name} 缺少对应的 .json 元数据")
    return str(idx), str(meta)


# ========== 工具定义 ==========

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "从知识库中检索与问题相关的文档片段。"
                "当用户的问题涉及特定技术细节、概念解释、操作步骤时,"
                "调用此工具获取参考资料再组织回答。"
                "返回最相关的 20 个文档块。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "direct_answer",
            "description": (
                "直接回答用户问题,不需要从知识库检索。"
                "当问题与知识库内容无关(如问候、简单计算、闲聊)时,"
                "调用此工具直接给出回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "直接回答的内容"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


# ========== 工具执行 ==========

def _exec_tool(name: str, arguments: dict) -> str:
    """执行工具并返回字符串结果(供 LLM 阅读)。"""
    global hybrid_search
    if hybrid_search is None:
        hybrid_search = _import_hybrid()

    if name == "search":
        query = arguments.get("query", "")
        try:
            idx_path, meta_path = _find_index()
            hits = hybrid_search(query, idx_path, meta_path, k=10)
            lines = []
            for i, h in enumerate(hits, 1):
                preview = h["chunk"][:200].replace("\n", " ")
                rrf = h.get("rrf_score", 0)
                lines.append(f"【{i}】(rrf={rrf:.4f}) {preview}...")
            return f"检索到 {len(hits)} 个相关文档块:\n\n" + "\n\n".join(lines)
        except Exception as e:
            return f"检索失败: {e}"

    elif name == "direct_answer":
        return arguments.get("answer", "(空回答)")

    else:
        return f"未知工具: {name}"


# ========== Agent 主循环 ==========

def _import_hybrid():
    """延迟导入 hybrid_search(避免循环依赖,也避免 jieba 冷启动拖慢其他模块)。"""
    from agent_project.retriever import hybrid_search
    return hybrid_search


# 全局引用,延迟初始化
hybrid_search = None


def run(question: str, max_iterations: int = 5, verbose: bool = True) -> str:
    """
    最小 Agent 循环:LLM 自主决策调工具→结果回注→循环直到给出答案。

    :param question: 用户问题
    :param max_iterations: 最大循环轮次(防无限循环)
    :param verbose: 是否打印每轮决策过程
    :return: LLM 最终答案
    """
    global hybrid_search
    if hybrid_search is None:
        hybrid_search = _import_hybrid()

    sep = "=" * 56

    if verbose:
        print(sep)
        print(f"  用户: {question}")
        print(sep)

    system_prompt = (
        "你是技术文档问答助手。工具说明:\n"
        "- search: 从知识库检索文档片段。调用时传入最相关的查询词。\n"
        "- direct_answer: 不需要检索,直接回答。\n\n"
        "核心规则:\n"
        "1. 收到问题后,如果与知识库相关,先调用 search 检索一次。\n"
        "2. 收到 search 结果后,你必须立即基于结果组织回答并直接输出,不要再调任何工具。\n"
        "3. 如果问题简单/与知识库无关,用 direct_answer 直接回答。\n"
        "4. 严格限制:每次对话最多调用一次 search。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, max_iterations + 1):
        # 1. LLM 决策(带上消息历史 + 可用工具)
        response = chat(messages, tools=TOOLS)
        choice = response.choices[0]
        msg = choice.message

        # 2. 判断:有没有 tool_calls?
        tool_calls = msg.tool_calls  # 可能是 None 或 []

        if not tool_calls:
            # ② LLM 说"我有答案了" → 返回
            answer = msg.content or ""
            if verbose:
                print(f"\n  [轮次{iteration}] LLM 决定:直接回答")
                print(sep)
                print(f"  AI: {answer}")
                print(sep)
            return answer

        # ① LLM 说"我要调工具" → 逐个执行并回注
        # 先把 assistant 消息(含 tool_calls)加入历史
        messages.append(msg.model_dump())

        if verbose:
            names = [tc.function.name for tc in tool_calls]
            print(f"\n  [轮次{iteration}] LLM 决定:调用工具 {names}")

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if verbose:
                print(f"    → {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:80]})")

            # 执行工具
            result = _exec_tool(fn_name, fn_args)

            if verbose:
                print(f"    ← 返回 {len(result)} 字符")

            # 结果回注为 tool 角色消息
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 超过最大轮次:强制用已有搜索结果回答(兜底,防死循环时不留答案)
    if verbose:
        print(f"\n  ⚠ 已达最大轮次{max_iterations},强制用已有结果回答")
    # 把所有 tool_result 提取出来,拼成最终 prompt
    collected = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            collected.append(m["content"])
    if collected:
        fallback_prompt = (
            f"请基于以下检索到的资料回答用户问题。不要调用任何工具,直接回答。\n\n"
            f"用户问题: {question}\n\n"
            f"检索资料:\n{''.join(collected[:3])}"
        )
        messages.append({"role": "user", "content": fallback_prompt})
        response = chat(messages, tools=None)
        return response.choices[0].message.content or ""
    return f"(已达最大轮次{max_iterations},无检索结果可参考)"
