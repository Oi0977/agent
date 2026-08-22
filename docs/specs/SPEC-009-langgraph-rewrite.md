# SPEC-009 LangGraph 等价重写与对比评测

- **编号**: SPEC-009
- **类型**: 功能(框架对照重写) · 影响面: 大(新增平行包;不改裸写版任何代码)
- **状态**: ✔ 已验收(2026-08-19)
- **创建**: 2026-08-19
- **关联**: SPEC-003(裸写循环,被对照方)、SPEC-004(多轮记忆)、SPEC-005(工具注册表,直接复用)、SPEC-006(token 记账)、SPEC-007(会话持久化,被对照方)、SPEC-008(评测基线)
- **确认方式**: 用户战役指令批量确认(2026-08-19:"接下来开始实现langgraph对比重写,直到重写完成并通过test验证,最后将重写的架构也分层次重新讲解……最后以这两个的对比文档为结束")

## 背景

裸写版(SPEC-003~007)已构成"麻雀虽小五脏俱全"的 MVP:while 循环 + tool_calls、
轮内完整/轮间压缩的多轮记忆、token 记账与预算、REPL 终端与会话 JSON 落盘、
16 题金标评测基线。路线图的下一格(README §五)是 LangGraph 生产级 Agent。

学习目标决定形态:**重写"编排层"而非全系统**——工具注册表、检索、LLM 客户端
全部复用,只把 while 循环换成 StateGraph,把"history 参数跨调用传递"换成
checkpointer 线程状态。这样两版的 diff 恰好就是"框架到底提供了什么",
对比文档(详解 10)才有干净的对照面。

## 目标 / 非目标(MoSCoW)

**Must**:
- 新包 `src/agent_project/agent_langgraph/`(与 `agent/` 并存,互不引用):
  - `state.py`:`AgentState`(TypedDict)+ 自定义 reducer —— messages
    (追加 + system 排头 + `_replace` 整段重写,即手写极简版 `add_messages`)
    与 stats(数值求和 + `_reset` 轮间清零)
  - `graph.py`:`build_graph(chat_fn=None, checkpointer=None, approval=False)`
    编排四节点 + 条件边;`run(question, thread_id=...)` 高层入口,
    返回 `(answer, stats)` —— **无 history 返回值**(记忆在 checkpointer 里,
    这是与裸写版最重要的签名差异)
  - `__main__.py`:真机演示/验收入口(单题对比、多轮记忆、跨进程持久化、流式)
- 图拓扑与裸写 while 语义一一对应:
  `START→agent→(条件边)→{tools→agent 循环 | fallback→compact | compact→END}`
  - agent 节点 = 循环体(chat + tools,判 tool_calls)
  - tools 节点 = execute_tool 统一分发 + 结果回注(**走同一注册表**)
  - fallback 节点 = 轮次耗尽强制回答(裸写的兜底,这里是一个显式节点)
  - compact 节点 = 轮间压缩(`_replace` 重写为 [system]+[user,assistant] 对,
    双闸截断 max_turns/max_history_tokens 语义与裸写一致)
- 多轮记忆 = checkpointer:同 `thread_id` 跨 invoke 记忆存活;
  SQLite checkpointer 跨进程存活(对照 SPEC-007 的会话 JSON)
- token 记账:每次 LLM 调用捕获 usage 累入 stats,键与裸写完全一致
  (`llm_calls/tool_calls/prompt_tokens/completion_tokens/history_turns`)
- 离线可测:`chat_fn` 可注入(与 chat.py 的 run 注入同一手法),
  全部 AC1-AC7 不碰真 API
- 人工审批门(Must,吸收 SPEC-004 Won't 桶 A):`approval=True` 时
  tools 节点 `interrupt()` 暂停图,调用方 `Command(resume=...)` 后继续

**Should**:
- `graph.stream(stream_mode="updates")` 节点级流式演示(模型 token 级流式
  受限于自有 chat() 非流式,不做)
- 真机对比表(≥3 题含 1 追问)落 spec 实现备注与详解 10

**Won't**:
- 不引入 langchain-openai/ChatOpenAI:保留自有 `llm_client.chat()`,
  两版共用同一 LLM 调用层,对比才纯粹(代价:非 langchain 消息生态,
  详见详解 09 设计决策)
- 不重写 chat.py 终端:记忆机制不同(history 传参 vs thread_id),
  接入另行评估
- 不动检索/评测层:SPEC-008 基线对检索层仍有效
- 不做 token 级流式、多 Agent 协作、subgraph

## 验收标准(Given-When-Then)

- [x] AC1 — 图结构与编译(离线)
  - Given build_graph()(注入 fake chat_fn + MemorySaver)
  - When compile
  Then 成功;`get_graph().nodes` 含 agent/tools/fallback/compact;
  无 tool_calls 时走 agent→compact→END(fake 无工具路径出答案)

- [x] AC2 — 工具走同一注册表(离线)
  - Given fake chat_fn 第一次返回 calculator 的 tool_calls
  - When run("算 (1+2)*3")
  Then tools 节点经 `execute_tool` 分发,结果回注为 tool 消息,
  fake 第二次收到的 messages 末尾含 `"(1+2)*3 = 9"`,stats.tool_calls=1

- [x] AC3 — checkpointer 记忆 + 轮间压缩等价(离线)
  - Given 同 thread_id 两次 run,MemorySaver
  - When 第一轮(触发一次工具)后第二轮 run("追问")
  Then 第二轮 agent 节点收到的 messages = [system] + 第一轮 [user,assistant]
  对 + 新 user —— **不含**第一轮的 tool_calls/tool 消息(压缩语义等价 SPEC-004)

- [x] AC4 — token 记账(离线)
  - Given fake chat_fn 每次返回 usage(prompt=100, completion=20)
  - When 一次 run 发生 2 次 LLM 调用
  Then stats = {llm_calls:2, prompt_tokens:200, completion_tokens:40, ...},
  键集与裸写 stats 相同

- [x] AC5 — 轮次兜底(离线)
  - Given fake chat_fn 永远返回 tool_calls(max_iterations=2)
  - When run()
  Then 走 fallback 节点强制回答(fake 收到的最终调用 tools=None),
  最终答案非空,图正常 END 不抛 GraphRecursionError

- [x] AC6 — 人工审批门(离线)
  - Given build_graph(approval=True),fake 首轮返回 tool_calls
  - When invoke
  Then 返回含 `__interrupt__`;再 `invoke(Command(resume="yes"))`
  Then 图继续,工具结果回注,最终出答案

- [x] AC7 — SQLite 跨实例持久化(离线)
  - Given 同一 db 文件,两次独立 build_graph(各自 SqliteSaver)
  - When 实例1 run 第 1 轮;实例2 同 thread_id run 追问
  Then 追问轮 messages 含第 1 轮问答对(记忆跨"进程"存活)

- [x] AC8 — 真机对比(真机)
  - Given 真实 GLM API + 已入库知识库
  - When 同一组问题(≥3 题含 1 个依赖上文的追问)分别跑裸写版与 LangGraph 版
  - Then 两版均给出可用答案;token 统计同量级;对比表落实现备注与详解 10

- [x] AC9 — 文档六件套
  - Then 详解 09(LangGraph 分层讲解)+ 10(两版对比)完成;
  specs README/00-总览/README/CLAUDE.md 结构树同步;requirements.txt 补依赖

## 涉及模块

- `src/agent_project/agent_langgraph/`(新):state.py / graph.py / __main__.py
- 复用不改:`agent/tools.py`(注册表)、`generator/llm_client.py`(chat)
- `tests/test_langgraph.py`(新):AC1-AC7(fake chat_fn,全离线)
- `requirements.txt`:补 langgraph / langgraph-checkpoint-sqlite
- `.gitignore`:补 data/checkpoints/(SQLite checkpointer 运行时产物)

## 风险与兜底

- **装包连带破坏学习环境**:装前 pip freeze 快照(temp/),装后 diff +
  跑 6 项离线回归绊网;实测仅新增 3 包(langgraph 1.0.7 与 langchain-core
  1.4.8 原已存在),零已有包变动,6/6 通过
- **API 版本漂移**(langgraph 1.x 与 0.x 教程差异大):动手前用探针脚本
  (temp/)验证自定义 reducer / interrupt+Command / SqliteSaver / get_graph
  四个关键面,全部通过才开写
- **非 langchain 消息格式**:state 用裸 OpenAI dict,自定义 reducer 替代
  add_messages(教学上反而暴露 reducer 本质);代价与理由记录在详解 09
- **429 限流**:仍走 chat() 的指数退避

## 实现备注(实现后回填)

- **版本**:langgraph 1.0.7 / langchain-core 1.4.8 / langgraph-checkpoint 4.1.0
  / langgraph-checkpoint-sqlite 3.1.1;装包仅新增 3 包,零已有包变动,
  装后 6 项离线回归绊网全过(快照与 diff 留 temp/)
- **AC1-AC7 离线**:`tests/test_langgraph.py` 全过(fake chat_fn 注入,
  MemorySaver / tmp SQLite,不碰真 API)
- **AC8 真机实测**(2026-08-19,`python -m agent_project.agent_langgraph` 四幕全通
  + 专项对比脚本;问题序列 Q1 独立 → Q3 追问 → Q2 算术,两版消息序列一致):

  | 题 | 裸写 llm/tool/prompt/completion | LangGraph 同序 | 说明 |
  |---|---|---|---|
  | Q1 解密 HTTPS | 2/1/2019/429 | 2/1/2028/491 | 等价;+9 tok 为检索词差异噪声 |
  | Q3 追问(依赖 Q1) | 2/1/2623/354 | **1/0/955/306** | 图版凭记忆直答(未再检索),答案同样正确 |
  | Q2 算术 | 2/1/2326/93 | 2/1/2508/93 | 等价,均 1158 |
  | 合计 | 6/3/6968/876 | 5/2/5491/890 | 同量级 |

  - 追问轮两版都正确做指代消解("拿到私钥之后的查看步骤"→右键 Follow SSL Stream)
  - **核心观察:token 成本由"检索决策次数"主导,不由编排层主导** —— 同一历史下
    模型这次选择"不再检索、凭记忆答"(省 1668 prompt tok),下次可能重搜;
    编排层本身零 token 开销
- **踩坑回填**(详见详解 09 §踩坑):
  1. fallback 判断必须在 **tools 出边**(工具执行完再判耗尽),放 agent 出边
     会少吃最后一轮检索结果
  2. 测试替身须遵循协议语义:tools=None 的调用不可能返回 tool_calls
     (AC5 首跑"假失败",修替身不修产品代码)
  3. 多阶段流水线断言要落在消息流经的截面上,不能只看最终 state
     (AC6:compact 后 state 无 tool 消息是设计结果)
  4. 免费 API 偶发请求挂起(ReadTimeout):对比脚本加外层重试;
     chat() 已有的 429 指数退避照常生效
