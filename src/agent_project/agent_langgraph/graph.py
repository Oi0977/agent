# -*- coding: utf-8 -*-
"""
LangGraph 版 Agent —— 裸写版(agent/agent.py)的等价重写,只换"编排层"。

对照表(裸写 → LangGraph,逐行语义映射见详解 09):

    while 循环体                 → agent 节点(chat + tools,判 tool_calls)
    tool_calls 分支里的执行/回注  → tools 节点(execute_tool 分发 + 回注)
    for 循环耗尽后的兜底          → fallback 节点(强制不带 tools 回答)
    _finish() 的压缩历史构造      → compact 节点(_replace 整段重写)
    history 参数跨调用传递        → checkpointer 线程状态(thread_id 寻址)
    max_iterations for 上界       → tools 出边的条件判断(轮次耗尽 → fallback)

复用不改(对照才纯粹):
- 工具注册表 agent/tools.py(get_tool_schemas / execute_tool)
- LLM 调用层 generator/llm_client.chat(同一 system 提示词从裸写版导入)
- 轮间压缩纯函数 _build_history(SPEC-004/006 已测,压缩语义两版逐字相同)
"""
import json
import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_project.agent.agent import SYSTEM_PROMPT, _build_history
from agent_project.agent.tools import execute_tool, get_tool_schemas
from agent_project.agent_langgraph.state import AgentState
from agent_project.path_manager import PathManager


# ---------- 小工具:消息格式 ----------

def _wire(messages: list) -> list:
    """发往 LLM 前剥掉私有键(下划线开头,如 _ephemeral)——OpenAI 协议不认识它们。"""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def _msg_to_dict(m) -> dict:
    """LLM 返回的 message → 裸 dict(真机是 ChatCompletion,测试替身是 SimpleNamespace)。"""
    if isinstance(m, dict):
        return dict(m)
    if hasattr(m, "model_dump"):  # openai SDK 的 pydantic 对象
        return m.model_dump()
    return {"role": "assistant", "content": getattr(m, "content", None),
            "tool_calls": getattr(m, "tool_calls", None)}


def _usage_delta(response) -> dict:
    """从响应里抽 usage 记账增量(SPEC-006:真数以 API 返回为准;缺 usage 记空不炸)。"""
    usage = getattr(response, "usage", None)
    p = getattr(usage, "prompt_tokens", None) if usage else None
    c = getattr(usage, "completion_tokens", None) if usage else None
    if p is None and c is None:
        return {}
    return {"llm_calls": 1, "prompt_tokens": p or 0, "completion_tokens": c or 0}


def _turn_anchor(messages: list) -> int:
    """最后一个"真实用户消息"(非兜底伪 user)的下标 —— 本轮问题的锚点。"""
    return max(i for i, m in enumerate(messages)
               if m.get("role") == "user" and not m.get("_ephemeral"))


def _done_iterations(messages: list) -> int:
    """本轮已执行的"决策轮次" = 锚点之后带 tool_calls 的 assistant 消息数。"""
    anchor = _turn_anchor(messages)
    return sum(1 for m in messages[anchor:]
               if m.get("role") == "assistant" and m.get("tool_calls"))


# ---------- 默认 checkpointer(SQLite 落盘,跨进程记忆)----------

_default_saver = None


def _default_checkpointer():
    """进程内单例:SQLite checkpointer 落 data/checkpoints/agent.db。

    对照 SPEC-007 的 data/sessions/*.json:同样是"磁盘上的会话状态",
    区别是 checkpointer 存的是每一步的完整 checkpoint(可回放/可中断恢复),
    会话 JSON 只存压缩后的最终历史。
    """
    global _default_saver
    if _default_saver is None:
        from langgraph.checkpoint.sqlite import SqliteSaver
        d = PathManager().DATA_ROOT / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(d / "agent.db", check_same_thread=False)
        _default_saver = SqliteSaver(conn)
    return _default_saver


# ---------- 建图 ----------

def build_graph(chat_fn=None, checkpointer=None, approval: bool = False,
                max_iterations: int = 5, max_turns: int = 10,
                max_history_tokens: int = 8192, verbose: bool = True):
    """
    编排 LangGraph 版 Agent,返回编译好的图(compiled graph)。

    :param chat_fn: LLM 调用(可注入 → 离线可测);默认真实 llm_client.chat。
        签名 chat_fn(messages, tools=None) → 响应对象(.choices[0].message / .usage)
    :param checkpointer: 记忆载体;None = 进程内 MemorySaver(重启即失)。
        持久记忆传 SqliteSaver(sqlite3.connect(db)) —— 同一 db + thread_id
        跨进程共享对话状态
    :param approval: True 时 tools 节点先 interrupt() 请求人工审批,
        调用方 Command(resume="yes"/"no") 后继续(此时应直接用返回的图,
        不要用 run() 简化入口)
    :param max_iterations / max_turns / max_history_tokens:语义与裸写 run() 同名
        参数完全一致
    """
    if chat_fn is None:
        from agent_project.generator.llm_client import chat as chat_fn  # 延迟导入,离线测试不碰

    # ---- 节点:闭包捕获 chat_fn/参数(节点本身只是普通函数,无魔法) ----

    def agent_node(state):
        """循环体:带上完整消息历史 + 工具 schema 调 LLM,拿回决策。"""
        response = chat_fn(_wire(state["messages"]), tools=get_tool_schemas())
        msg = _msg_to_dict(response.choices[0].message)
        delta = _usage_delta(response)
        if verbose:
            if msg.get("tool_calls"):
                names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                print(f"  [graph] agent 节点:LLM 决定调工具 {names}")
            else:
                print("  [graph] agent 节点:LLM 决定直接回答")
            if delta:
                print(f"    [token] prompt {delta['prompt_tokens']} / "
                      f"completion {delta['completion_tokens']}")
        return {"messages": [msg], "stats": delta}

    def tools_node(state):
        """执行上一条 assistant 的 tool_calls 并回注结果(走裸写版同一注册表)。"""
        tcs = state["messages"][-1].get("tool_calls") or []
        if approval:
            decision = interrupt({"question": "是否允许执行以下工具?",
                                  "tools": [tc["function"]["name"] for tc in tcs]})
            if decision != "yes":
                if verbose:
                    print(f"  [graph] 人工审批:拒绝({decision})")
                return {"messages": [
                    {"role": "tool", "tool_call_id": tc["id"],
                     "content": "用户拒绝了工具调用,请直接基于已有信息回答。"}
                    for tc in tcs]}
        out, n = [], 0
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)  # 统一分发,失败转字符串不炸循环
            n += 1
            if verbose:
                print(f"  [graph] tools 节点:{name}({json.dumps(args, ensure_ascii=False)[:60]})"
                      f" ← {len(result)} 字符")
            out.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        return {"messages": out, "stats": {"tool_calls": n}}

    def fallback_node(state):
        """轮次耗尽兜底:收集已有 tool 结果,强制不带 tools 调一次 LLM(与裸写同文案)。"""
        messages = state["messages"]
        question = messages[_turn_anchor(messages)]["content"]
        collected = [m["content"] for m in messages if m.get("role") == "tool"]
        if not collected:
            if verbose:
                print(f"  [graph] ⚠ 已达最大轮次{max_iterations},且无检索结果")
            return {"messages": [{"role": "assistant",
                                  "content": f"(已达最大轮次{max_iterations},无检索结果可参考)"}]}
        prompt = (f"请基于以下检索到的资料回答用户问题。不要调用任何工具,直接回答。\n\n"
                  f"用户问题: {question}\n\n"
                  f"检索资料:\n{''.join(collected[:3])}")
        if verbose:
            print(f"  [graph] ⚠ 已达最大轮次{max_iterations},强制用已有结果回答")
        response = chat_fn(_wire(messages) + [{"role": "user", "content": prompt,
                                               "_ephemeral": True}], tools=None)
        answer = _msg_to_dict(response.choices[0].message)
        # 兜底 prompt 标记 _ephemeral:只在本次调用出现,compact 时排除,不进历史
        return {"messages": [{"role": "user", "content": prompt, "_ephemeral": True}, answer],
                "stats": _usage_delta(response)}

    def compact_node(state):
        """轮间压缩:轮内工作列表 → [system] + [user, assistant] 对(复用裸写纯函数)。

        _build_history(SPEC-004 压缩 + SPEC-006 双闸截断)直接复用 ——
        两版的历史语义逐字相同,这是"等价重写"最硬的一条保证。
        """
        messages = state["messages"]
        anchor = _turn_anchor(messages)
        question = messages[anchor]["content"]
        answer = messages[-1].get("content") or ""
        history = _build_history(messages[:anchor] or None, question, answer,
                                 max_turns, max_history_tokens)
        if verbose:
            kept = sum(1 for m in history if m.get("role") == "user")
            print(f"  [graph] compact 节点:历史压缩为 {kept} 轮问答对")
        return {"messages": [{"role": "_replace", "messages": history}]}

    # ---- 边:while 循环的判断条件变成两张路由表 ----

    def route_after_agent(state):
        """agent 出边:有 tool_calls → tools;没有 → 收尾压缩。"""
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") else "compact"

    def route_after_tools(state):
        """tools 出边:本轮决策轮次未耗尽 → 回 agent 继续;耗尽 → 兜底。

        裸写的 for 上界在这里落地:判断点在"工具已执行完"之后,
        与裸写"第 N 轮工具执行完、循环变量耗尽"的时机一致。
        """
        return "agent" if _done_iterations(state["messages"]) < max_iterations else "fallback"

    g = StateGraph(state_schema=AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("fallback", fallback_node)
    g.add_node("compact", compact_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "compact"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "fallback"])
    g.add_edge("fallback", "compact")
    g.add_edge("compact", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ---------- 高层入口(等价裸写 agent.run 的形状)----------

def run(question: str, thread_id: str = "default", max_iterations: int = 5,
        max_turns: int = 10, max_history_tokens: int = 8192,
        verbose: bool = True, checkpointer=None, chat_fn=None) -> tuple:
    """
    LangGraph 版一问一答:run(问题, thread_id) → (答案, 本轮统计)。

    与裸写 run(question, history) 的签名差异就是两版的本质差异:
    - 裸写:调用方持有并传回 history(记忆在调用方手里)
    - 图版:只给 thread_id,记忆在 checkpointer 里(记忆在框架手里)
    返回值不再有 history —— 想看/迁记忆用 build_graph().get_state()。

    stats 键与裸写完全一致:{llm_calls, tool_calls, prompt_tokens,
    completion_tokens, history_turns}(对比表直接对齐)。
    approval 流程(人机审批)不适用本入口 —— interrupt 需要调用方
    在图对象上 Command(resume=...),请直接用 build_graph()。
    """
    graph = build_graph(chat_fn=chat_fn,
                        checkpointer=checkpointer or _default_checkpointer(),
                        max_iterations=max_iterations, max_turns=max_turns,
                        max_history_tokens=max_history_tokens, verbose=verbose)
    config = {"configurable": {"thread_id": thread_id},
              # 保险丝:正常路径由 route_after_tools 的轮次判断先兜住,
              # recursion_limit 只防路由 bug 导致的死循环(每轮 ≈ agent+tools 两步)
              "recursion_limit": 2 * max_iterations + 6}

    if verbose:
        prior = graph.get_state(config).values.get("messages", [])
        n_users = sum(1 for m in prior if m.get("role") == "user")
        print("=" * 56)
        print(f"  第{n_users + 1}轮对话(thread={thread_id},历史{len(prior)}条消息)")
        print(f"  用户: {question}")
        print("=" * 56)

    result = graph.invoke(
        {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": question}],
         "stats": {"_reset": True}},
        config)

    if "__interrupt__" in result:  # 本入口未开 approval,出现即异常路径
        raise RuntimeError("图被 interrupt 暂停(approval 流程请用 build_graph() 驱动)")

    messages = result["messages"]
    # 键集对齐裸写(reducer 只累加出现过的键,这里补齐为 0,对比表才能逐键对上)
    stats = {"llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    stats.update(result["stats"])
    stats["history_turns"] = sum(1 for m in messages
                                 if m.get("role") == "user" and not m.get("_ephemeral"))
    answer = messages[-1].get("content") or ""
    if verbose:
        print("=" * 56)
        print(f"  AI: {answer}")
        print("=" * 56)
    return answer, stats
