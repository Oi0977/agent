# -*- coding: utf-8 -*-
"""
AgentState —— LangGraph 的"状态黑板"定义(TypedDict + reducer)。

LangGraph 的核心心智模型:**State 是唯一跨节点共享的东西**。
每个节点收到完整 state,返回"增量";reducer 决定增量怎么并入 state。
裸写版里这个角色由 run() 局部变量 messages/stats 承担(函数栈就是状态),
换成图之后节点彼此看不见对方的局部变量 —— 状态必须显式化,reducer 就是
"增量如何合并"的唯一事实来源。

本文件刻意**手写** reducer 而不用框架内置的 add_messages:
- append_messages ≈ add_messages 的极简版(追加 + system 排头 + _replace 整段重写)
- sum_stats        ≈ 数值累加 + _reset 轮间清零
教学目的:看清 reducer 不过是 merge(old, new) 函数,没有魔法。
代价与取舍见详解 09 §设计决策。
"""
from typing import Annotated, TypedDict


def append_messages(old: list, new: list) -> list:
    """
    messages 键的 reducer:定义"消息增量如何并入历史"。

    三条合并语义:
    1. 普通消息(user/assistant/tool)→ 追加到末尾(轮内循环靠它累积)
    2. system 消息 → 永远排头;重复发送即原位刷新(幂等)——
       run() 每轮都发 [system, user],旧线程头部已有 system 时刷新而非重复
    3. role="_replace" 的伪消息 → 携带完整列表整段重写 ——
       compact 节点做"轮间压缩"用它;这是内置 RemoveMessage(按 id 删)
       的粗粒度亲戚:删的少用 RemoveMessage,整段重写用 _replace 一步到位

    框架对照:langgraph 内置 add_messages 还多做两件事 —— 把入参转成
    langchain 消息对象、给每条消息补 id(供 RemoveMessage 精确删除)。
    我们保持裸 OpenAI dict(发给自有 chat() 即用),所以自己写。
    """
    out = list(old)
    for m in new:
        role = m.get("role")
        if role == "system":
            if out and out[0].get("role") == "system":
                out[0] = m
            else:
                out.insert(0, m)
        elif role == "_replace":
            out = list(m["messages"])
        else:
            out.append(m)
    return out


def sum_stats(old: dict, new: dict) -> dict:
    """
    stats 键的 reducer:数值键求和;_reset 标记 → 整体替换(轮间清零)。

    stats 是"本轮统计"(与裸写 run() 的局部 stats 同语义),所以每个新轮次
    的 invoke 输入带 {"_reset": True} 归零重计;节点返回的增量被逐键累加。
    history_turns 不进 reducer —— 它是派生量,由 run() 从最终消息数出来。
    """
    if new.get("_reset"):
        return {k: v for k, v in new.items() if k != "_reset"}
    merged = dict(old)
    for k, v in new.items():
        merged[k] = merged.get(k, 0) + v
    return merged


class AgentState(TypedDict):
    """跨节点共享的全部状态。刻意只留两块黑板:

    - messages:完整工作列表(轮内含 tool_calls/tool 消息;轮末被 compact 压缩)
    - stats:本轮 token/调用统计(usage 真数,键与裸写版完全一致便于对比)
    本轮问题/轮次计数都不设键 —— 可从 messages 派生,不冗余存储。
    """
    messages: Annotated[list, append_messages]
    stats: Annotated[dict, sum_stats]
