# -*- coding: utf-8 -*-
"""
SPEC-009 离线验收(AC1-AC7):LangGraph 版 Agent,全离线 —— chat_fn 注入测试替身,
不碰真 API / 知识库 / 本地模型。跑法:

    python tests/test_langgraph.py
"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent_project.agent.agent import SYSTEM_PROMPT
from agent_project.agent_langgraph.graph import build_graph, run


# ---------- 测试替身:脚本化 LLM ----------

class ScriptedLLM:
    """按脚本顺序吐响应,并记录每次收到的 (messages, tools) 供断言。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.responses.pop(0)


class InfiniteToolLLM:
    """带工具时永远要求调工具(测轮次兜底)。

    tools=None(兜底调用)时只能回纯文本 —— 真实 API 语义:
    请求不带 tools,模型就没有 tool_calls 可发。
    """

    def __init__(self):
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if tools is None:
            msg = SimpleNamespace(content="基于已检索资料的兜底回答", tool_calls=None)
        else:
            msg = SimpleNamespace(content=None, tool_calls=[
                {"id": f"t{len(self.calls)}",
                 "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10))


def resp(content=None, tool_calls=None, usage=(100, 20)):
    """造一个最小 LLM 响应对象(形状对齐 openai ChatCompletion 的取用面)。"""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=usage[0],
                                                 completion_tokens=usage[1]))


def calc_call(cid="t1", expr="(1+2)*3"):
    return {"id": cid, "function": {"name": "calculator",
                                    "arguments": json.dumps({"expression": expr})}}


def invoke_input(question):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": question}],
            "stats": {"_reset": True}}


# ---------- AC1 图结构与编译 ----------

def ac1():
    fake = ScriptedLLM([resp("答:好的")])
    graph = build_graph(chat_fn=fake, checkpointer=MemorySaver(), verbose=False)
    nodes = set(graph.get_graph().nodes)
    assert {"agent", "tools", "fallback", "compact"} <= nodes, f"节点缺失: {nodes}"
    ans, stats = run("你好", thread_id="ac1", chat_fn=fake,
                     checkpointer=MemorySaver(), verbose=False)
    assert ans == "答:好的", ans
    assert stats["history_turns"] == 1
    print("AC1 ✓ 图编译成功,四节点齐全;无工具路径 agent→compact→END 出答案")


# ---------- AC2 工具走同一注册表 ----------

def ac2():
    fake = ScriptedLLM([resp(tool_calls=[calc_call()]), resp("答案是 9")])
    ans, stats = run("算 (1+2)*3", thread_id="ac2", chat_fn=fake,
                     checkpointer=MemorySaver(), verbose=False)
    assert stats["tool_calls"] == 1, stats
    # 第一次调用带 tools(来自注册表 get_tool_schemas)
    first_msgs, first_tools = fake.calls[0]
    assert first_tools and first_tools[0]["function"]["name"] in (
        "search", "calculator", "direct_answer", "list_documents")
    # 第二次调用末尾是工具结果回注(tool 角色,内容来自 execute_tool 真实执行)
    second_msgs, _ = fake.calls[1]
    last = second_msgs[-1]
    assert last["role"] == "tool" and "(1+2)*3 = 9" in last["content"], last
    assert ans == "答案是 9"
    print("AC2 ✓ tools 节点经 execute_tool 分发,结果回注;stats.tool_calls=1")


# ---------- AC3 checkpointer 记忆 + 轮间压缩 ----------

def ac3():
    mem = MemorySaver()
    fake1 = ScriptedLLM([resp(tool_calls=[calc_call(expr="1+1")]), resp("第一轮答案")])
    run("第一轮问题", thread_id="ac3", chat_fn=fake1, checkpointer=mem, verbose=False)
    fake2 = ScriptedLLM([resp("第二轮答案")])
    run("第二轮追问", thread_id="ac3", chat_fn=fake2, checkpointer=mem, verbose=False)

    # 第二轮 LLM 收到的 = [system] + 第一轮 [user,assistant] 对 + 新 user
    msgs, _ = fake2.calls[0]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert not any(m["role"] == "tool" for m in msgs), "tool 消息漏进了历史"
    assert not any(m.get("tool_calls") for m in msgs), "tool_calls 意图漏进了历史"

    # 双闸截断闸门随图生效:max_turns=1 → 三轮后历史只剩 1 对
    fake = ScriptedLLM([resp("答1"), resp("答2"), resp("答3")])
    g = build_graph(chat_fn=fake, checkpointer=MemorySaver(), max_turns=1, verbose=False)
    cfg = {"configurable": {"thread_id": "ac3b"}}
    for q in ("问1", "问2", "问3"):
        g.invoke(invoke_input(q), cfg)
    st = g.get_state(cfg).values["messages"]
    n_users = sum(1 for m in st if m["role"] == "user")
    assert n_users == 1, f"max_turns=1 应只剩 1 对,实际 {n_users}"
    print("AC3 ✓ 记忆经 checkpointer 存活;轮间压缩等价(tool/tool_calls 不入历史);"
          "max_turns 闸在图上同样生效")


# ---------- AC4 token 记账 ----------

def ac4():
    fake = ScriptedLLM([resp(tool_calls=[calc_call(expr="1+1")]), resp("答案")])
    _, stats = run("算 1+1", thread_id="ac4", chat_fn=fake,
                   checkpointer=MemorySaver(), verbose=False)
    assert stats["llm_calls"] == 2, stats
    assert stats["prompt_tokens"] == 200, stats
    assert stats["completion_tokens"] == 40, stats
    assert set(stats) == {"llm_calls", "tool_calls", "prompt_tokens",
                          "completion_tokens", "history_turns"}, set(stats)
    print("AC4 ✓ usage 逐调用累加(2 次 → 200/40),键集与裸写 stats 完全一致")


# ---------- AC5 轮次兜底 ----------

def ac5():
    fake = InfiniteToolLLM()
    ans, stats = run("无限调工具", thread_id="ac5", chat_fn=fake,
                     checkpointer=MemorySaver(), max_iterations=2, verbose=False)
    final_msgs, final_tools = fake.calls[-1]
    assert final_tools is None, "兜底调用必须不带 tools"
    assert "不要调用任何工具" in final_msgs[-1]["content"]
    assert ans, "兜底答案非空"
    assert stats["llm_calls"] == 3, stats  # 2 轮决策 + 1 次兜底
    print("AC5 ✓ 轮次耗尽走 fallback 节点:强制 tools=None 回答,答案非空,图正常 END")


# ---------- AC6 人工审批门(interrupt) ----------

def ac6():
    fake = ScriptedLLM([resp(tool_calls=[calc_call(expr="1+1")]), resp("审批后的答案")])
    g = build_graph(chat_fn=fake, checkpointer=MemorySaver(), approval=True, verbose=False)
    cfg = {"configurable": {"thread_id": "ac6"}}
    r1 = g.invoke(invoke_input("算 1+1"), cfg)
    assert "__interrupt__" in r1, "应在 tools 节点前中断"
    inter = r1["__interrupt__"][0].value
    assert "calculator" in str(inter), inter
    r2 = g.invoke(Command(resume="yes"), cfg)
    assert r2["messages"][-1]["content"] == "审批后的答案"
    # 工具确已执行:第二次 LLM 调用收到的末尾是 calculator 的真实结果
    # (最终 state 里没有 tool 消息是正常的 —— compact 已把轮内消息压掉)
    second_msgs, _ = fake.calls[1]
    assert second_msgs[-1]["role"] == "tool" and "1+1 = 2" in second_msgs[-1]["content"]
    print("AC6 ✓ approval=True 时 interrupt 暂停;Command(resume='yes') 后工具执行并出答案")


# ---------- AC7 SQLite 跨实例持久化 ----------

def ac7():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "cp.db")
        cfg = {"configurable": {"thread_id": "ac7"}}
        # 实例1(模拟进程1):首轮问答后关闭连接
        fake1 = ScriptedLLM([resp("暗号是菠萝,已记住")])
        conn1 = sqlite3.connect(db, check_same_thread=False)
        g1 = build_graph(chat_fn=fake1, checkpointer=SqliteSaver(conn1), verbose=False)
        g1.invoke(invoke_input("记住:暗号是菠萝"), cfg)
        conn1.close()
        # 实例2(模拟进程2):全新连接,同 thread_id 读状态
        fake2 = ScriptedLLM([resp("暗号是菠萝")])
        conn2 = sqlite3.connect(db, check_same_thread=False)
        g2 = build_graph(chat_fn=fake2, checkpointer=SqliteSaver(conn2), verbose=False)
        run("暗号是什么?", thread_id="ac7", chat_fn=fake2,
            checkpointer=SqliteSaver(conn2), verbose=False)
        msgs, _ = fake2.calls[0]
        contents = [m["content"] for m in msgs]
        assert any("菠萝" in (c or "") for c in contents), contents
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"], roles
        conn2.close()
    print("AC7 ✓ 同一 db 两个独立图实例:记忆跨\"进程\"存活(首轮问答对在追问轮可见)")


if __name__ == "__main__":
    for fn in (ac1, ac2, ac3, ac4, ac5, ac6, ac7):
        fn()
    print("\n全部通过:SPEC-009 图结构/注册表/记忆压缩/记账/兜底/审批门/持久化 ✓")
