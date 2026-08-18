# -*- coding: utf-8 -*-
"""agent_langgraph —— 裸写 Agent(agent/)的 LangGraph 等价重写(SPEC-009)。

只重写编排层:工具注册表 / LLM 调用 / 轮间压缩纯函数全部复用裸写版,
两版的 diff 恰好就是"框架提供了什么"(对比见详解 10)。

对外接口:
  run(question, thread_id)          一问一答,记忆在 checkpointer(thread_id 寻址)
  build_graph(chat_fn, checkpointer) 拿编译好的图自己驱动(审批/流式/查看状态)
"""
from agent_project.agent_langgraph.graph import build_graph, run

__all__ = ["build_graph", "run"]
