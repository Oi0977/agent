# 10 · 裸写 vs LangGraph 对比(战役收官)

> 前情:同一套 Agent,裸写版(《07》,SPEC-003~007)和 LangGraph 重写版
> (《09》,SPEC-009)并存,共用**同一个**工具注册表、检索栈、LLM 客户端、
> system 提示词、轮间压缩函数。本篇是两者的全面对照 —— 也是整个
> "裸写 → 框架"学习路线的收官结论。

## 0. 一句话结论

**LangGraph 没有让 Agent 变聪明,它让"编排"从代码里消失了** —— 循环变拓扑、
局部变量变 State、记忆传参变 checkpointer、"暂停等人"从进程阻塞变成框架能力。
代价是你必须接受它的心智模型(reducer/条件边/checkpoint),并在调试时
"在脑子里编译拓扑"。流程简单时裸写更透明,流程复杂时框架开始还本。

## 1. 总对比表

| 维度 | 裸写(agent/) | LangGraph(agent_langgraph/) | 评价 |
|------|---------------|------------------------------|------|
| **循环控制** | `for iteration in range(5)` 写死在函数里 | 图上的环(agent→tools→agent)+ 条件边判界 | 语义等价(判断时机须放 tools 出边,见 09 坑 1) |
| **状态** | run() 局部变量 messages/stats | State(TypedDict)+ reducer 增量合并 | 图版必须显式化:节点互相看不见局部变量 |
| **多轮记忆** | 调用方持 history 传参往返 | checkpointer + thread_id | **本质差异最大的一条**:记忆归属从调用方移到框架 |
| **持久化** | 自己写 JSON 存取(SPEC-007) | SqliteSaver 落盘,存每一步 checkpoint | 图版天然支持"回到任意一步重跑"(time travel) |
| **轮次兜底** | 循环耗尽后的 if 分支 | 显式 fallback 节点 + tools 出边条件 | 语义等价(文案逐字相同) |
| **人工审批** | 无(要自己发明暂停协议) | `interrupt()` + `Command(resume=...)` | 图版免费获得,且**进程崩溃后可恢复**(state 已存档) |
| **流式** | 无 | `stream(stream_mode="updates")` 节点级事件 | 模型 token 级流式两版都受限于非流式 chat() |
| **token 记账** | 闭包 `_record_usage` 改局部 stats | `_usage_delta` 返回增量,reducer 累加 | 键集与数值口径完全一致(AC4) |
| **轮间压缩** | `_build_history` 纯函数 | **同一个**函数,compact 节点调用 | 等价性最硬的保证:压缩语义逐字相同(AC3) |
| **可测试性** | 依赖注入(chat 可换但循环难拆) | chat_fn 注入 + 节点天然可单测 + 图结构可断言 | 图版多一层可测面:拓扑本身(get_graph) |
| **依赖** | 0 框架(纯 openai SDK) | langgraph + langchain-core + sqlite 三件 | 学习环境实测:新增 3 包,零已有包变动 |
| **代码量(编排层)** | agent.py **256 行** | state.py 75 + graph.py **359 行**(+40%) | 复杂度守恒:控制流没了,结构声明多了 |
| **心智负担** | 顺序读代码即是控制流 | 要先在脑内"画图"再读节点 | 裸写的最大优势 |

## 2. 同题实测:数字说了什么(SPEC-009 AC8,2026-08-19)

条件:同一知识库、同一 system 提示、同一压缩历史构造,问题序列
Q1(独立)→ Q3(追问 Q1)→ Q2(算术),两版消息序列完全一致。

| 题 | 裸写 llm/tool/prompt/completion | LangGraph 同序 | 差异解读 |
|---|---|---|---|
| Q1 解密 HTTPS | 2/1/2019/429 | 2/1/2028/491 | **等价**;+9 tok 是两次运行检索词微差的噪声 |
| Q3 追问 | 2/1/2623/354 | **1/0/955/306** | 图版凭记忆直答(没再检索),答案同样正确 |
| Q2 算术 | 2/1/2326/93 | 2/1/2508/93 | 等价,均答 1158 |
| **合计** | 6/3/6968/876 | 5/2/5491/890 | 同量级 |

三个结论:

1. **编排层零 token 开销**。两版发给模型的消息由同一函数构造,token 差
   全部来自模型自己的决策差异(检索词选择、要不要重搜)。
2. **token 成本由"检索决策次数"主导,不由框架主导**。Q3 一行代码没改,
   模型这次选了"不搜、凭记忆答"就省了 1668 prompt tokens;换一次运行
   可能又去搜。想控成本,优化点在提示词/工具描述/检索结果长度,
   不在换框架。
3. **答案质量等价**。追问轮两版都正确完成指代消解(拿到私钥之后的步骤
   → 右键 Follow SSL Stream),说明压缩记忆两版同样工作。

## 3. 逐机制对照(代码级)

### 3.1 循环:for → 环 + 条件边

```python
# 裸写:控制流在代码里(缩进即拓扑)
for iteration in range(1, max_iterations + 1):
    response = chat(messages, tools=TOOLS)
    if not response.choices[0].message.tool_calls:
        return _finish(response.choices[0].message.content)
    ...执行工具,结果回注...

# LangGraph:控制流在数据里(拓扑 + 路由函数)
g.add_conditional_edges("agent", route_after_agent, ["tools", "compact"])
g.add_conditional_edges("tools", route_after_tools, ["agent", "fallback"])
```

`if tool_calls` 变成 route_after_agent 的返回值;`for` 上界变成
route_after_tools 的 `<` 判断。**收益**:加一条分支路径(如"高危工具走人审")
只需加节点加边,不动现有节点;裸写则要在循环体里再插 if-else。
**代价**:排错时没有"第 37 行"可指,只有"tools 出边的路由"。

### 3.2 记忆:history 传参 → checkpointer

```python
# 裸写:记忆是调用方的行李,每轮背进背出
ans1, hist, st1 = run(q1)                    # hist = [system, 问1, 答1]
ans2, hist, st2 = run(q2, history=hist)      # 全量重发(无状态 API 的本质)

# LangGraph:记忆是框架的档案,thread_id 是档案号
ans1, st1 = lg_run(q1, thread_id="t1")
ans2, st2 = lg_run(q2, thread_id="t1")
```

两版底层做的事一模一样(无状态 API,记忆=重发 messages);区别是**谁持有**。
裸写的 history 是普通数据,可以 inspect、可以改、可以塞进自己的 JSON;
图版的记忆在 checkpointer 里,想看要 `get_state()`,想跨进程要共享 db。
裸写的自由度更高,图版的**一致性保证**更强(不会出现"忘了传 history"
这种 bug —— 没有 history 参数可忘)。

### 3.3 兜底:循环尾部 if → fallback 节点

两版兜底 prompt 逐字相同("请基于以下检索到的资料回答……不要调用任何工具");
裸写是 for 循环后的 if 分支,图版是显式节点。**判断时机**都必须在
"工具执行完之后"(图版=tools 出边),否则少吃最后一轮检索结果(09 坑 1)。

### 3.4 状态合并:直接改 → 返回增量

```python
# 裸写:命令式,直接改局部列表
messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
stats["tool_calls"] += 1

# LangGraph:声明式,返回增量,reducer 决定怎么并
return {"messages": [{"role": "tool", ...}],
        "stats": {"tool_calls": 1}}
```

命令式直观;声明式换来的是**框架能拦截每一次状态变更** —— checkpoint、
stream 事件、time travel 都建立在这个拦截点上。这是"控制流数据化"的
深层交易:你交出改状态的权力,换框架对所有变更的可见性。

## 4. 框架的价值与代价

### 价值(裸写要自己造的)

1. **checkpoint 体系**:每步存档 → 断点恢复 / time travel / 跨进程记忆。
   裸写只有"会话 JSON"这一种粗粒度快照。
2. **interrupt + Command**:人机审批是框架原语,进程死了也能恢复;
   裸写的 input() 只能阻塞活着的进程。
3. **流式事件**:stream() 按节点吐事件,接 SSE 就是现成的进度条。
4. **拓扑即文档**:`get_graph()` 可视化、结构可断言(AC1 把"图长什么样"
   写成了测试)。
5. **生态延长线**:Postgres checkpointer、LangGraph Platform、studio 调试,
   每一条裸写都要自己铺。

### 代价(裸写没有的)

1. **概念税**:State/reducer/条件边/checkpoint/config 四件套,学习曲线真实存在
   (本战役光 API 探针就跑了一轮)。
2. **代码量 +40%**(256 → 359 行):复杂度守恒,控制流没了,结构声明来了。
3. **调试距离变长**:裸写的 bug 在栈回溯里;图版的 bug 要先定位到节点+边,
   再进节点看。本战役的"fallback 少吃一轮"就是拓扑级 bug,栈回溯帮不上忙。
4. **版本漂移**:0.x→1.x 的教程失配是真实的坑(09 坑 4),框架越活跃,
   教程半衰期越短。

## 5. 选型建议(何时迁移)

出现以下**任一**信号,裸写的维护成本开始超过框架税:

| 信号 | 对应框架能力 |
|------|--------------|
| 流程出现第 3 条以上分支(如"检索失败走兜底问答""高危操作走人审") | 条件边比嵌套 if 可维护 |
| 需要"暂停等人/等外部系统",且进程可能重启 | interrupt + checkpoint 恢复 |
| 需要审计/回放"第 N 步当时的状态" | checkpoint time travel |
| 需要 streaming 进度反馈 | stream() 事件 |
| 多人协作,流程要可视化沟通 | 拓扑即文档 |

反之:**单循环 + 简单工具 + 记忆只要问答对** —— 裸写 256 行更透明,
面试讲得出每一行,出了 bug 栈回溯直达。

一句话:**先裸写后框架不是学习仪式,是真实的技术决策路径** —— 你得先知道
裸写的痛在哪,才知道框架在卖什么。本项目两版并存,就是这个结论的活体标本。

## 6. Q&A 自测

### Q1 · 两版发给 LLM 的消息序列完全一致,为什么?这个"一致"在对比实验里排除了什么变量?
**难度: 机制** · 考点: 对照实验设计

> **你的回答**:



---

### Q2 · 实测里 Q3 追问轮两版 token 差了 1668,这是框架差异吗?真正的差异来源是什么?想省 token 该优化哪里?
**难度: 机制** · 考点: 成本归因(实测核心观察)

> **你的回答**:



---

### Q3 · 裸写的 history 传参和图版的 checkpointer,记忆的"持有者"不同带来什么具体后果(各说两条)?
**难度: 决策** · 考点: 记忆归属

> **你的回答**:



---

### Q4 · "控制流数据化"是什么意思?你交出了什么,换回了什么?哪三个框架能力建立在这笔交易上?
**难度: 面试** · 考点: 框架本质(高频面试题)

> **你的回答**:



---

### Q5 · 你的项目现在两版并存。如果下一步要做"多 Agent 协作"(检索 Agent + 写作 Agent 互审),你会基于哪版演进?为什么?
**难度: 面试** · 考点: 选型判断

> **你的回答**:
