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

# ========== Token 估算(SPEC-006)==========

def _estimate_tokens(text: str) -> int:
    """
    粗估 token 数(纯函数,离线可测)。

    启发式(理论见详解 08):GLM 是中文优化词表 → 中文≈1字1token;
    非中文(英文/数字/符号)≈4字符/token。误差 ±20~30%,只用于发送前的
    预算决策;**记账以 API 返回的 usage 为准(真数)**。
    注意:该比率是"中文优化词表"相对的 —— 换老英文词表模型(字节回退)即失准。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _messages_tokens(messages) -> int:
    """一组 messages 的估算 token 数(只算 content;聊天模板开销不计,见 08 §3.2)。"""
    return sum(_estimate_tokens(m.get("content") or "") for m in messages)


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


def _build_history(prev, question, answer, max_turns=10, max_history_tokens=8192):
    """
    轮间历史构造(纯函数,可离线单测 —— SPEC-004 AC2/AC5、SPEC-006 AC2/AC5 的落点)。

    轮内工作列表(含 assistant 的 tool_calls 意图与 tool 结果)用完即弃,从不进入历史:
    一次 search 回注约 2500 字符,是历史体积的大头;压成 [user, assistant] 原子对后,
    截断只需按对切片,永不拆散 OpenAI 协议要求的 tool_calls/tool 配对。

    双闸截断(SPEC-006):
    - max_turns:轮数闸(防"轮数多但都短"的会话无限长)
    - max_history_tokens:token 预算闸(防"单轮超长"撑爆预算)—— 拼接本轮后
      若估算超预算,从最旧轮开始整对丢弃;system 固定开销不计入预算、永不丢;
      至少保留最近 1 轮(哪怕它自身超预算)

    :param prev: 上一轮返回的压缩历史([system] + N 个 [user, assistant] 对);首轮传 None
    :param question: 本轮原始问题。显式传入而非从工作列表提取 ——
        兜底路径中追加的是 fallback prompt,不是用户的原始问题
    :param answer: 本轮最终答案(正常回答或兜底答案)
    :param max_turns: 最多保留的轮数(system 不计)
    :param max_history_tokens: 历史问答对的 token 预算(估算值)
    :return: [system] + 截断后的 [user, assistant] 对
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

    # token 预算闸:超预算从最旧轮整对丢弃(估算只看问答对内容,system 不计)
    def _pairs_tokens(pl):
        return sum(_estimate_tokens(u.get("content") or "")
                   + _estimate_tokens(a.get("content") or "") for u, a in pl)

    while len(pairs) > 1 and _pairs_tokens(pairs) > max_history_tokens:
        pairs = pairs[1:]

    history = list(system)
    for u, a in pairs:
        history.append({"role": "user", "content": u.get("content") or ""})
        history.append({"role": "assistant", "content": a.get("content") or ""})
    return history


# ========== Agent 主循环 ==========

def run(question: str, history=None, max_iterations: int = 5,
        max_turns: int = 10, max_history_tokens: int = 8192,
        verbose: bool = True) -> tuple:
    """
    最小 Agent 循环 + 多轮记忆 + token 记账:LLM 自主决策调工具→结果回注→循环直到给出答案。

    返回 (答案, 压缩历史, 本轮统计),调用方持有历史跨轮传递:
        ans1, hist, st1 = run("第一个问题")
        ans2, hist, st2 = run("针对答案的追问", history=hist)   ← 这就是"记忆"

    :param question: 用户问题
    :param history: 上一轮返回的压缩历史;None/空列表 = 新会话(SPEC-003 行为)
    :param max_iterations: 单轮内 Agent 循环最大轮次(防无限循环)
    :param max_turns: 返回历史最多保留的对话轮数(system 不计)
    :param max_history_tokens: 历史 token 预算(估算);超出从最旧轮整对丢弃(SPEC-006)
    :param verbose: 是否打印每轮决策过程与 token 统计
    :return: (answer, history, stats)
        stats = {"llm_calls", "tool_calls", "prompt_tokens",
                 "completion_tokens", "history_turns"}(本轮各 LLM 调用的 usage 之和)
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
        if history:
            print(f"  [token] 历史重发约 {_messages_tokens(history)} tok"
                  f"(估算,预算 {max_history_tokens})")
        print(f"  用户: {question}")
        print(sep)

    # ---- token 记账(SPEC-006):每次 LLM 调用捕获 API 白送的 usage ----
    stats = {"llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "history_turns": 0}

    def _record_usage(response):
        """记一次调用的 usage;个别兼容层不返回 usage 时记 0 并提示,不炸主流程。"""
        usage = getattr(response, "usage", None)
        p = getattr(usage, "prompt_tokens", None) if usage else None
        c = getattr(usage, "completion_tokens", None) if usage else None
        if p is None and c is None:
            if verbose:
                print("    [token] ⚠ 本次响应未携带 usage,记账记 0")
            return
        stats["llm_calls"] += 1
        stats["prompt_tokens"] += p or 0
        stats["completion_tokens"] += c or 0
        if verbose:
            print(f"    [token] prompt {p or 0} / completion {c or 0}"
                  f"(本轮累计 {stats['prompt_tokens']}/{stats['completion_tokens']})")

    def _finish(answer):
        """统一出口:构造压缩历史 + 回填统计,返回三元组。"""
        new_history = _build_history(history, question, answer,
                                     max_turns, max_history_tokens)
        stats["history_turns"] = sum(1 for m in new_history if m.get("role") == "user")
        return answer, new_history, stats

    for iteration in range(1, max_iterations + 1):
        # 1. LLM 决策(带上消息历史 + 可用工具)
        response = chat(messages, tools=TOOLS)
        _record_usage(response)
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
            return _finish(answer)

        # ① LLM 说"我要调工具" → 逐个执行并回注
        # 先把 assistant 消息(含 tool_calls)加入历史
        messages.append(msg.model_dump())

        if verbose:
            names = [tc.function.name for tc in tool_calls]
            print(f"\n  [轮次{iteration}] LLM 决定:调用工具 {names}")

        for tc in tool_calls:
            stats["tool_calls"] += 1
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
        _record_usage(response)
        answer = response.choices[0].message.content or ""
        return _finish(answer)
    answer = f"(已达最大轮次{max_iterations},无检索结果可参考)"
    return _finish(answer)
