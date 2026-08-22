# 09 · LangGraph 重写详解(agent_langgraph)

> 解决的问题:裸写版(《07》)用一个 while 循环 + history 传参把 Agent 跑通了,
> 但"循环控制、跨轮记忆、断点恢复、人工审批"全靠调用方自己搬。
> LangGraph 把这些**编排关注点**从业务代码里抽出来,变成框架的原生能力。
> 本篇是 SPEC-009 的实现复盘:**只换编排层,工具/检索/LLM/压缩全部复用裸写版**,
> 所以两版的 diff 恰好就是"框架到底提供了什么"(对比见《10》)。

## 1. 核心机制:图执行模型(StateGraph 心智)

LangGraph 只有一个核心抽象:**有向图上的状态机**。四个概念撑起全部:

| 概念 | 是什么 | 本项目落点 |
|------|--------|-----------|
| **State** | 跨节点共享的"黑板"(TypedDict),每键配一个 **reducer** 定义增量怎么并入 | `AgentState`:`messages` + `stats` 两块黑板 |
| **Node** | 普通函数:收到完整 state,返回**增量**(partial dict) | agent / tools / fallback / compact 四节点 |
| **Edge** | 固定边(无条件)或**条件边**(函数返回下一节点名) | `agent→tools/compact`、`tools→agent/fallback` 两张路由表 |
| **Checkpointer** | 每个"超步"结束把整个 state 存档;`thread_id` 寻址 | MemorySaver(内存)/ SqliteSaver(data/checkpoints/) |

**执行模型**:invoke 一次输入 → 图按边推进,每经过一个节点就执行它、用 reducer
合并增量、存一次 checkpoint → 到 END 返回最终 state。节点内部**看不见彼此的局部
变量**——这就是为什么裸写版 run() 里的局部 `messages`/`stats` 必须升格成 State。

**while 循环去哪了**:裸写的循环控制(`for iteration in range(...)` + `if tool_calls`)
被拆成**拓扑 + 条件边**——循环变成图上的环(agent→tools→agent),`if` 变成
route 函数的返回值。控制流从"代码缩进"变成"数据结构",这是图框架的本质交易。

### 1.1 图拓扑与裸写循环的逐行映射

```
                    ┌────────────────────────── 裸写 agent.py ──────────────────────────┐
                    │ for iteration in 1..max_iterations:                                │
                    │     response = chat(messages, tools=TOOLS)          ← agent 节点   │
                    │     if not tool_calls: return _finish(answer)       ← route→compact│
                    │     执行工具,结果回注 messages                      ← tools 节点   │
                    │ (循环耗尽)强制不带 tools 回答                        ← fallback 节点│
                    │ _finish():_build_history 压缩历史                   ← compact 节点 │
                    └──────────────────────────────────────────────────────────────────┘

  START ──▶ agent ──(有 tool_calls)──▶ tools ──(轮次未耗尽)──▶ agent   ← 图上的环 = while
              │                          │
              │                          └──(轮次耗尽)──▶ fallback ──┐
              └──(无 tool_calls,最终答案)───────────────────────────▶ compact ──▶ END
```

| 裸写 | LangGraph | 语义是否等价 |
|------|-----------|------|
| while 循环体(chat + 判 tool_calls) | `agent` 节点 | ✓ |
| tool_calls 分支的执行与回注 | `tools` 节点 | ✓(走**同一个** execute_tool 注册表) |
| `for` 上界耗尽后的兜底 | `tools` 出边条件 → `fallback` 节点 | ✓(兜底文案逐字相同;判断时机同为"工具执行完之后") |
| `_finish()` 构造压缩历史 | `compact` 节点 | ✓(**同一个** `_build_history` 纯函数,含双闸截断) |
| history 参数跨调用传递 | checkpointer + thread_id | 语义等价,机制不同(§3.2) |
| stats 局部变量 + `_record_usage` 闭包 | State `stats` 键 + reducer 累加 | ✓(键集相同,usage 同源) |
| max_iterations=5 | `route_after_tools` 里 `< max_iterations` 判断 | ✓ |
| recursion 无(循环写死了界) | `recursion_limit` 保险丝 | 图版多一层防路由 bug |

## 2. 代码走读

### 2.1 state.py —— reducer:增量合并的唯一事实来源

```python
def append_messages(old, new):        # messages 键的 reducer
    out = list(old)
    for m in new:
        if m.get("role") == "system":     # system 永远排头,重复发送即原位刷新
            ...
        elif m.get("role") == "_replace": # 伪消息:整段重写(compact 节点用)
            out = list(m["messages"])
        else:                             # 普通消息追加(轮内循环靠它累积)
            out.append(m)
    return out

def sum_stats(old, new):              # stats 键的 reducer
    if new.get("_reset"):             # 新轮次 → 整体清零重计
        return {k: v for k, v in new.items() if k != "_reset"}
    ...数值键逐个求和...
```

三个要点:

- **reducer 就是 merge(old, new) 函数,没有魔法**。框架内置的 `add_messages`
  做的同件事还多做两步:把入参转成 langchain 消息对象、给每条消息补 id
  (供 `RemoveMessage` 按 id 精确删除)。我们保持裸 OpenAI dict(发给自有
  chat() 即用),所以手写 15 行极简版 —— 教学上反而把 reducer 的本质暴露出来了。
- **system 排头语义**:run() 每轮输入都带 [system, user];旧线程头部已有 system
  时 reducer 原位刷新而非重复追加。这样新会话/续会话走同一条输入路径,不用分支。
- **`_replace` ≈ 粗粒度 RemoveMessage**:compact 要"轮内完整 → 轮间压缩",
  删的消息占大头,按 id 一条条删啰嗦,整段重写一步到位。

### 2.2 graph.py —— 四节点两条件边

**agent 节点(循环体)**:

```python
def agent_node(state):
    response = chat_fn(_wire(state["messages"]), tools=get_tool_schemas())
    msg = _msg_to_dict(response.choices[0].message)
    return {"messages": [msg], "stats": _usage_delta(response)}
```

`_wire` 发送前剥掉私有键(`_ephemeral`,OpenAI 协议不认识);`_usage_delta`
从响应抽 usage 记账(SPEC-006 的"真数"原则原样搬过来)。对比裸写:同一件事,
裸写在循环体里直接改局部 `messages.append(...)`,图版**返回增量**交给 reducer。

**tools 节点(执行与回注)**:遍历上一条 assistant 的 tool_calls,经
`execute_tool` 统一分发(与裸写共用同一注册表,SPEC-005),结果逐条回注为
tool 消息。`approval=True` 时先 `interrupt()`:

```python
if approval:
    decision = interrupt({"question": "是否允许执行以下工具?", "tools": names})
    if decision != "yes":
        return {"messages": [回注"用户拒绝"的 tool 消息]}   # 拒绝也是"工具结果"
```

**fallback 节点(轮次兜底)**:收集已有 tool 结果,拼与裸写逐字相同的兜底
prompt,`chat_fn(..., tools=None)` 强制回答。兜底 prompt 标 `_ephemeral`:
只在本次调用出现,compact 时排除 —— 等价于裸写"兜底路径里追加的不是用户原始
问题"的处理(question 从"最后一个非 ephemeral user 消息"取,即用户原话)。

**compact 节点(轮间压缩)**:

```python
history = _build_history(messages[:anchor] or None, question, answer,
                         max_turns, max_history_tokens)
return {"messages": [{"role": "_replace", "messages": history}]}
```

直接 import 裸写版的 `_build_history`(SPEC-004 压缩 + SPEC-006 双闸截断,
已被 test_history.py 覆盖)—— **两版历史语义逐字相同**,这是"等价重写"最硬
的一条保证,也是对比评测(AC8)公平性的前提。

**两张路由表(while 的判断条件变成边)**:

```python
def route_after_agent(state):   # 有 tool_calls → tools;没有 → 收尾
    return "tools" if state["messages"][-1].get("tool_calls") else "compact"

def route_after_tools(state):   # 本轮决策轮次未耗尽 → 回 agent;耗尽 → 兜底
    return "agent" if _done_iterations(...) < max_iterations else "fallback"
```

注意 fallback 的判断点在 **tools 出边**(工具已执行完)而不是 agent 出边 ——
与裸写"第 N 轮工具执行完、循环变量耗尽"的时机一致,兜底才能用上最后一轮
检索结果(细节见 §5 踩坑 1)。

### 2.3 run() —— 高层入口,签名差异就是本质差异

```python
# 裸写:调用方持有并传回 history(记忆在调用方手里)
ans, hist, stats = bare_run(question, history=hist)

# 图版:只给 thread_id,记忆在 checkpointer 里(记忆在框架手里)
ans, stats = lg_run(question, thread_id="t1")
```

run() 内部:invoke 输入 `[system, user]` + `stats._reset`,config 带
`thread_id` 和 `recursion_limit`(保险丝:正常路径由 route_after_tools 的轮次
判断先兜住,它只防路由 bug);出口把 stats 键集补齐成与裸写完全一致
(reducer 只累加出现过的键,比如没调工具就没有 tool_calls 键)。

### 2.4 checkpointer —— 记忆的两种载体

- **MemorySaver**:进程内字典,重启即失 —— 测试与一次性演示。
- **SqliteSaver**(`data/checkpoints/agent.db`,默认单例):跨进程存活,
  `__main__.py` §3 演示两个独立 build_graph 只共享 db 文件,记忆照样接上。
  对照 SPEC-007 的 `data/sessions/*.json`:同样是磁盘会话状态,区别是
  checkpoint 存**每一步**的完整状态(可回放/可从任意步恢复),会话 JSON
  只存压缩后的最终历史。

## 3. 设计决策

| # | 决策 | 理由 |
|---|------|------|
| 1 | **只重写编排层**,工具/检索/LLM/压缩全复用 | 两版 diff = 框架价值本身;同 tool_calls、同 system、同压缩函数,对比评测才有意义 |
| 2 | 自定义 reducer,不用内置 `add_messages` | 保持裸 OpenAI dict 直通自有 chat();顺带把 reducer 本质(merge 函数)暴露在教学面上。代价:不能用框架的 RemoveMessage/消息类型生态 |
| 3 | 不引入 langchain-openai / ChatOpenAI | 引入它就要把消息在 langchain 对象和 dict 间来回转换,模型 I/O 层变厚,**模糊"图到底做了什么"**;生产项目会反过来选(见 §6) |
| 4 | fallback 是显式节点,不是 try/except GraphRecursionError | 轮次耗尽是**预期业务分支**,用条件边表达才能"图上可见";recursion_error 是异常路径,留给真正的 bug |
| 5 | stats 进 State(reducer 累加 + _reset 清零) | 裸写的局部变量+闭包在图里活不了;记账逻辑(usage 抽取)两版同源 |
| 6 | run() 不返回 history | 记忆在 checkpointer,返回 history 会诱导调用方绕过框架自己管记忆 —— 那等于白引入框架 |
| 7 | approval 走 interrupt 而不是 input() | interrupt 是**框架级暂停**:state 已存 checkpoint,进程死了也能恢复;input() 只是阻塞当前进程 |
| 8 | approval 流程不进 run() 简化入口 | resume 需要 Command 对象,交互形态和一问一答不同,硬塞会造出怪 API |

## 4. 踩坑记录

### 坑 1:fallback 判断时机错了会少一轮检索

第一版把"轮次耗尽 → fallback"放在 **agent 出边**(agent#5 返回 tool_calls 时
直接送 fallback)—— 但此时第 5 轮的工具**还没执行**,兜底只能用前 4 轮的
检索结果;裸写是"执行完第 5 轮工具、循环变量耗尽"才兜底。修法:判断挪到
**tools 出边**,工具先执行、再判耗尽。教训:**循环上界落在哪条边上,决定了
最后一次循环体有没有完整执行** —— 图把控制流变成拓扑后,这种细节不再有
缩进提醒你,只能靠 trace(或离线测试)验证。

### 坑 2:测试替身不真实,测出了"假 bug"

AC5(轮次兜底)首跑失败:兜底答案为空。排查发现是 InfiniteToolLLM 替身
**无视 tools=None 照样返回 tool_calls** —— 真实 API 在不带 tools 的请求里
根本没有 tool_calls 可发。修替身(`tools is None → 只能回纯文本`),
产品代码一行没动。教训:fake 的行为契约必须对齐真实依赖的**协议语义**,
否则测出的失败会把你引向不存在的 bug。(对照:《07》§4 的 tool-call loop
也是"协议行为"问题 —— 模型层和测试层都要按协议办事。)

### 坑 3:断言观察点选错 —— compact 之后 state 里没有 tool 消息是正常的

AC6 首跑失败:想在最终 state 里找工具结果,找不到。因为 compact 已经把轮内
tool 消息压掉了(这正是 SPEC-004 的设计!)。正确观察点是"**第二次 LLM 调用
收到过**工具结果"(fake.calls[1] 的末尾)。教训:多阶段流水线的断言要落在
**消息流经的截面**上,而不是只盯最终状态。

### 坑 4:0.x 教程 ≠ 1.x API

langgraph 1.0 与网上大量 0.x 教程的导入路径/行为有差异(如 MemorySaver 的
位置、Command 的语义)。动手前先跑探针脚本(temp/,四种关键面:自定义
reducer / interrupt+Command / SqliteSaver 跨实例 / get_graph)再写产品代码
—— 和装包后跑绊网回归是同一个思想:**先验证地基,再盖房子**。

## 5. 与 SPEC 的对应

- 图拓扑/节点职责 → SPEC-009 Must(图拓扑一一对应)
- 轮间压缩语义 → 复用 SPEC-004 的 `_build_history`(AC3 断言 tool 消息不入历史)
- token 记账 → SPEC-006 usage 真数原则(AC4 键集一致)
- 兜底 → SPEC-003 的 max_iterations 语义(AC5,含判断时机,见坑 1)
- 审批门 → SPEC-004 Won't 桶 A 的 interrupt(AC6)
- 持久化 → 对照 SPEC-007 会话 JSON(AC7 跨实例)

## 6. 业界选型与取舍

### LangGraph 在 Agent 框架生态里的位置(2026)

| 候选 | 模型 | 适合 | 劣势 |
|------|------|------|------|
| **裸写(本项目 agent/)** | while + tool_calls,~260 行 | 学习机制、简单可控场景 | 循环/记忆/恢复全自己搬 |
| **LangGraph(本项目 agent_langgraph/)** | 显式状态机图 | 复杂流程(多分支/人审/长任务/需要恢复) | 概念税(State/reducer/checkpoint);调试要看拓扑 |
| LangChain AgentExecutor | 预制 ReAct 循环 | 快速原型 | 已被官方标记 legacy,黑盒度高 |
| AutoGen / CrewAI | 多 Agent 对话/角色协作 | 多 Agent 协作场景 | 单 Agent 场景过重 |
| Dify / Coze 等 Workflow | 可视化节点编排 | 非代码人群、产品化交付 | 受平台能力边界约束 |

**取舍逻辑**:LangGraph 的甜蜜点是**"流程复杂到值得把控制流数据化"** ——
分支多、要人审、要断点恢复、要观测每一步。流程简单时,它的概念税大于收益
(本项目 4 节点已经能感觉到:同样的逻辑,裸写一眼看完,图版要在脑子里"编译"
一遍拓扑)。生产化路径:checkpointer 换 Postgres、stream 配 SSE、
部署用 LangGraph Platform —— 这些都是框架生态的延长线,裸写版每一条都要
自己造。

### 真实产品形态

- LangGraph Platform / langgraph-server:图即服务,自带 studio 可视化调试
- 检查点系统进数据库后,天然支持"回到任意一步重跑"(time travel)
- interrupt+Command 是人工审批/_HITL 的标准范式,与工作流引擎的
  用户任务节点(user task)同构

## 7. Q&A 自测

### Q1 · LangGraph 的 State 和普通函数的局部变量,本质区别是什么?为什么裸写的 messages 不用设计 reducer,图版必须设计?
**难度: 基础** · 考点: 状态显式化

> **你的回答**:



---

### Q2 · reducer 是什么?你手写的 append_messages 和框架内置的 add_messages 差在哪?各自适合什么场景?
**难度: 机制** · 考点: 增量合并

> **你的回答**:



---

### Q3 · 裸写的 for iteration in range(5) 上界,在图版里是怎么实现的?为什么 fallback 的判断要放在 tools 节点的出边而不是 agent 节点的出边?
**难度: 机制** · 考点: 控制流拓扑化(踩坑 1)

> **你的回答**:



---

### Q4 · 裸写版调用方传 history,图版只传 thread_id。两种记忆机制各自的优劣?如果要把图版接进 chat.py 终端,需要改什么?
**难度: 决策** · 考点: 记忆归属

> **你的回答**:



---

### Q5 · interrupt() 和直接在节点里 input() 等用户输入,都能"暂停等人",本质区别是什么?进程崩溃后各自的后果?
**难度: 面试** · 考点: 框架级暂停 vs 进程级阻塞

> **你的回答**:



---

### Q6 · 你的测试替身曾让 AC5 假失败。fake 的什么行为不真实?这个坑说明"测试替身"要遵循什么原则?
**难度: 面试** · 考点: 测试替身的协议保真

> **你的回答**:



---

### Q7 · checkpointer 存的和 SPEC-007 会话 JSON 存的有什么不同?"每一步都存"换来了什么能力?
**难度: 机制** · 考点: checkpoint vs 快照

> **你的回答**:



---

### Q8 · 什么信号出现时,你会把一个裸写 Agent 迁移到 LangGraph?说出至少三个,并解释为什么这些信号指向"控制流数据化"。
**难度: 面试** · 考点: 框架引入时机(高频面试题)

> **你的回答**:
