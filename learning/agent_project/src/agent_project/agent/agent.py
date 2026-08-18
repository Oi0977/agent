# -*- coding: utf-8 -*-
"""
最小 Agent 循环 —— 裸写 while + tool_calls,零框架依赖。

本质:Agent 不是魔法,就是一个 while 循环 + LLM 的 function calling 能力。
LLM 每轮输出两种可能之一:
  ① 返回 tool_calls → 说明"我要调某个工具" → 执行工具 → 结果回注 → 再来一轮
  ② 返回 content    → 说明"我有答案了"     → 返回最终回答

工具从注册表来(agent/tools.py):@tool 一处声明,这里自动取 schema、统一分发。

多轮记忆(SPEC-004):LLM API 无状态,"记忆"由客户端持 messages 列表跨调用实现 ——
run() 返回 (答案, 压缩历史),调用方把历史传回下一次 run()。
"""
import json

from agent_project.agent.tools import execute_tool, get_tool_schemas
from agent_project.generator.llm_client import chat

# ========== 轮间记忆(SPEC-004)==========

SYSTEM_PROMPT = (
    "你是技术文档问答助手。可用工具:\n"
    "- search: 从知识库(多文档)检索片段。问题涉及技术细节/概念/操作步骤时先用它。\n"
    "- direct_answer: 不需要检索,直接回答(问候/闲聊/常识)。\n"
    "- calculator: 数学计算。任何算术都用它,不要心算。\n"
    "- list_documents: 列出知识库里有哪些文档。\n\n"
    "核心规则:\n"
    "1. 与知识库相关的问题,先调用 search 检索一次。\n"
    "2. 收到 search 结果后,立即基于结果组织回答并直接输出,不要再调任何工具。\n"
    "3. 简单/与知识库无关的问题用 direct_answer;算术用 calculator。\n"
    "4. 严格限制:每次对话最多调用一次 search。"
)


def _build_history(prev, question, answer, max_turns=10):
    """
    轮间历史构造(纯函数,可离线单测 —— SPEC-004 AC2/AC5 的落点)。

    轮内工作列表(含 assistant 的 tool_calls 意图与 tool 结果)用完即弃,从不进入历史:
    一次 search 回注约 2500 字符,是历史体积的大头;压成 [user, assistant] 原子对后,
    窗口截断只需按对切片,永不拆散 OpenAI 协议要求的 tool_calls/tool 配对。

    :param prev: 上一轮返回的压缩历史([system] + N 个 [user, assistant] 对);首轮传 None
    :param question: 本轮原始问题。显式传入而非从工作列表提取 ——
        兜底路径中追加的是 fallback prompt,不是用户的原始问题
    :param answer: 本轮最终答案(正常回答或兜底答案)
    :param max_turns: 最多保留的轮数(system 不计)
    :return: [system] + 最近 ≤max_turns 个 [user, assistant] 对
    """
    if prev:
        system = [dict(m) for m in prev if m.get("role") == "system"]
        users = [m for m in prev if m.get("role") == "user"]
        assistants = [m for m in prev if m.get("role") == "assistant"]
        pairs = list(zip(users, assistants))  # 压缩历史严格成对,zip 安全
    else:
        system = [{"role": "system", "content": SYSTEM_PROMPT}]
        pairs = []

    pairs = pairs + [({"role": "user", "content": question},
                      {"role": "assistant", "content": answer})]
    if len(pairs) > max_turns:
        pairs = pairs[-max_turns:]

    history = list(system)
    for u, a in pairs:
        history.append({"role": "user", "content": u.get("content") or ""})
        history.append({"role": "assistant", "content": a.get("content") or ""})
    return history


# ========== Agent 主循环 ==========

def run(question: str, history=None, max_iterations: int = 5,
        max_turns: int = 10, verbose: bool = True) -> tuple:
    """
    最小 Agent 循环 + 多轮记忆:LLM 自主决策调工具→结果回注→循环直到给出答案。

    返回 (答案, 压缩历史),调用方持有历史跨轮传递:
        ans1, hist = run("第一个问题")
        ans2, hist = run("针对答案的追问", history=hist)   ← 这就是"记忆"

    :param question: 用户问题
    :param history: 上一轮返回的压缩历史;None/空列表 = 新会话(SPEC-003 行为)
    :param max_iterations: 单轮内 Agent 循环最大轮次(防无限循环)
    :param max_turns: 返回历史最多保留的对话轮数(system 不计)
    :param verbose: 是否打印每轮决策过程
    :return: (answer, history) —— history 可直接作为下一轮的 history 参数
    """
    TOOLS = get_tool_schemas()

    sep = "=" * 56

    # 新会话:[system, user] 起步(SPEC-003 行为,AC4 向后等价);
    # 续聊:历史 + 新 user(system 已在历史头部,不重复添加)
    if history:
        messages = list(history) + [{"role": "user", "content": question}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    if verbose:
        turn_no = 1 + (sum(1 for m in history if m.get("role") == "user") if history else 0)
        n_hist = len(history) if history else 0
        print(sep)
        print(f"  第{turn_no}轮对话(历史{n_hist}条消息)")
        print(f"  用户: {question}")
        print(sep)

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
            return answer, _build_history(history, question, answer, max_turns)

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

            # 执行工具(注册表统一分发,失败转字符串回注)
            result = execute_tool(fn_name, fn_args)

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
        answer = response.choices[0].message.content or ""
        return answer, _build_history(history, question, answer, max_turns)
    answer = f"(已达最大轮次{max_iterations},无检索结果可参考)"
    return answer, _build_history(history, question, answer, max_turns)
