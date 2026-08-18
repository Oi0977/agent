# -*- coding: utf-8 -*-
"""
LangGraph 版真机演示/验收入口(SPEC-009 AC8):python -m agent_project.agent_langgraph

四幕:
  §1 单题对比   —— 同一问题,裸写版 vs 图版各答一次,并排 token 统计
  §2 多轮记忆   —— 同 thread_id 追问,记忆来自 checkpointer(不是 history 传参)
  §3 跨进程持久 —— 两次独立建图(各自 SqliteSaver、同一 db),记忆跨"进程"存活
  §4 节点级流式 —— graph.stream(stream_mode="updates"),节点完成一个报一个
"""
import sqlite3
import tempfile
import os

from agent_project.agent_langgraph.graph import build_graph, run
from agent_project.path_manager import PathManager


def _section(title):
    print("\n" + "#" * 60)
    print(f"# {title}")
    print("#" * 60)


def main():
    # 演示可重复跑:清掉上次会话留下的 checkpoint(记忆从头开始)
    cp_dir = PathManager().DATA_ROOT / "checkpoints"
    if cp_dir.exists():
        for f in ("agent.db", "demo_persist.db"):
            p = cp_dir / f
            if p.exists():
                p.unlink()

    # ---------- §1 单题对比 ----------
    _section("§1 同题对比:裸写 run() vs LangGraph run()")
    q = "Wireshark 里怎么解密 HTTPS 流量?"

    print("\n>>> [A] 裸写版 agent.run(history 传参记忆)")
    from agent_project.agent import run as bare_run
    ans_bare, hist, st_bare = bare_run(q, verbose=True)

    print("\n>>> [B] LangGraph 版 run(thread_id 寻址记忆)")
    ans_lg, st_lg = run(q, thread_id="demo-compare", verbose=True)

    print("\n----- 对比表 -----")
    rows = [("llm_calls", st_bare["llm_calls"], st_lg["llm_calls"]),
            ("tool_calls", st_bare["tool_calls"], st_lg["tool_calls"]),
            ("prompt_tokens", st_bare["prompt_tokens"], st_lg["prompt_tokens"]),
            ("completion_tokens", st_bare["completion_tokens"], st_lg["completion_tokens"])]
    print(f"{'统计键':<18}{'裸写':>8}{'LangGraph':>12}")
    for k, a, b in rows:
        print(f"{k:<18}{a:>8}{b:>12}")
    print(f"\n[裸写答案前200字] {ans_bare[:200]}")
    print(f"[图版答案前200字] {ans_lg[:200]}")

    # ---------- §2 多轮记忆(checkpointer) ----------
    _section("§2 多轮记忆:同 thread_id 追问(记忆在 checkpointer,无 history 传参)")
    q2 = "你说的第二步里的协议设置,在哪个菜单打开?"
    ans2, st2 = run(q2, thread_id="demo-compare", verbose=True)
    print(f"\n追问轮 stats: {st2}")
    print(f"[追问答案前200字] {ans2[:200]}")

    # ---------- §3 跨"进程"持久化 ----------
    _section("§3 跨进程持久化:两个独立 build_graph,同一 SQLite db")
    db = PathManager().DATA_ROOT / "checkpoints" / "demo_persist.db"
    tid = "demo-persist"

    print(">>> [进程1] 第一次 build_graph + 首轮问答")
    conn1 = sqlite3.connect(db, check_same_thread=False)
    from langgraph.checkpoint.sqlite import SqliteSaver
    g1 = build_graph(checkpointer=SqliteSaver(conn1))
    g1.invoke({"messages": [{"role": "user", "content": "我最喜欢的抓包工具是 Wireshark,记住这一点。"}],
               "stats": {"_reset": True}},
              {"configurable": {"thread_id": tid}})
    conn1.close()  # 模拟进程退出

    print(">>> [进程2] 全新 build_graph(只共享 db 文件)+ 追问验证")
    conn2 = sqlite3.connect(db, check_same_thread=False)
    g2 = build_graph(checkpointer=SqliteSaver(conn2))
    st = g2.get_state({"configurable": {"thread_id": tid}})
    print(f"    进程2 读到的历史消息数: {len(st.values.get('messages', []))}")
    ans3, st3 = run("我刚才说我最喜欢的抓包工具是什么?只报工具名。",
                    thread_id=tid, verbose=False,
                    checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    print(f"    进程2 答案: {ans3}")
    print(f"    (含 'Wireshark' → {'✓ 记忆跨进程存活' if 'wireshark' in ans3.lower() else '✗ 记忆丢失'})")
    print(f"    stats: {st3}")
    conn2.close()

    # ---------- §4 节点级流式 ----------
    _section("§4 节点级流式:stream(stream_mode='updates') —— 节点完成一个报一个")
    print("问题: (123+456)*2 等于几?(走 calculator 工具)")
    g = build_graph(verbose=False)
    config = {"configurable": {"thread_id": "demo-stream"}}
    for chunk in g.stream({"messages": [{"role": "user", "content": "(123+456)*2 等于几?"}],
                           "stats": {"_reset": True}}, config,
                          stream_mode="updates"):
        for node, payload in chunk.items():
            if node == "__interrupt__":
                print(f"  [stream] interrupt: {payload}")
                continue
            msgs = payload.get("messages", [])
            desc = ",".join(f"{m.get('role')}" for m in msgs)
            print(f"  [stream] 节点 {node} 完成 → 新增消息角色: {desc}")

    print("\n全部演示完毕。离线验收见 tests/test_langgraph.py(AC1-AC7)。")


if __name__ == "__main__":
    main()
