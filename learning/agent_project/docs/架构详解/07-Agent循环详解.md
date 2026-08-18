# 07 · Agent 循环(agent)

> 解决的问题:RAG 是"固定流水线"——每次提问都执行 search→rerank→generate,
> 不问不管。Agent 让 LLM **自主决策**:"这个问题要不要检索?检索什么?查几轮?"
> 从"程序驱动"进化为"模型驱动"。

## 1. 核心机制:while 循环 + function calling

Agent 不是魔法,本质上就是一个**while 循环**:

```
用户提问 → messages = [system, user]
while 未达最大轮次:
    response = LLM(messages, tools=[search, direct_answer])
    if response 有 tool_calls:        ← LLM 说"我要调工具"
        执行工具 → 结果回注 messages(tool角色) → 再来一轮
    else:
        return response.content        ← LLM 说"我有答案了"
```

**判断条件只有一种**:response 有没有 `tool_calls`。有就执行并回注,没有就返回。
LLM 的 function calling 能力是这整个机制的**地基**——它能输出结构化的工具调用指令,
代码只需要"读指令、执行、把结果喂回去"。

**和固定 RAG 的对比**:

```
固定 RAG:  问题 → [search] → [rerank] → [generate] → 答案  (固定顺序,不问不管)
Agent:     问题 → while LLM 还想调工具:                        (模型自主决策)
               └→ search / direct_answer / ...               (每轮可能不同)
```

## 2. 代码走读(agent.py)

### 工具定义(TOOLS)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "从知识库中检索与问题相关的文档片段...",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "direct_answer",
            "description": "直接回答用户问题,不需要从知识库检索...",
            ...
        }
    }
]
```

**为什么有 direct_answer 工具**:让 LLM 有"不需要检索"的选项。
没有它,LLM 可能对所有问题都调 search(即使问的是"1+1")。
有了它,LLM 可以自主判断:"这个问题和知识库无关,直接回答。"

### 系统提示(防 tool-call loop)

```python
system_prompt = (
    "核心规则:\n"
    "1. 收到问题后,如果与知识库相关,先调用 search 检索一次。\n"
    "2. 收到 search 结果后,你必须立即基于结果组织回答并直接输出,不要再调任何工具。\n"
    "3. 如果问题简单/与知识库无关,用 direct_answer 直接回答。\n"
    "4. 严格限制:每次对话最多调用一次 search。"
)
```

**这是整个 Agent 里最关键的一行设计**——没它,LLM 会陷入死循环(见 §4 踩坑)。

### 工具执行与结果回注

```python
# 执行工具
result = _exec_tool(fn_name, fn_args)

# 结果回注为 tool 角色消息
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,      # 必须与 tool_calls 里的 id 匹配
    "content": result,
})
```

**tool_call_id 的作用**:OpenAI 协议用它把"工具调用"和"工具结果"配对。
LLM 内部靠这个 id 知道"这个结果对应我刚才的哪个调用"。

### 兜底机制(轮次耗尽)

```python
# 轮次耗尽时:用已有搜索结果拼 prompt,强制调一次 LLM 回答(不带 tools)
fallback_prompt = f"请基于以下检索到的资料回答...{collected_results}..."
response = chat(messages, tools=None)  # 不带 tools → LLM 只能回答
return response.choices[0].message.content
```

**为什么需要**:万一 LLM 就是不停搜(真发生过,见 §4),至少把搜到的东西用上,
给用户一个答案而不是空返回。

## 3. 设计决策

| # | 决策 | 理由 |
|---|---|---|
| 1 | 裸写 while + tool_calls,不引入框架 | **理解 Agent 到底是什么**:循环+function calling;框架留到理解机制后评估 |
| 2 | 两个工具:search + direct_answer | direct_answer 让 LLM 有"不检索"选项,是 Agent 自主性的体现 |
| 3 | max_iterations=5 硬上限 | 防死循环;每轮≈5s,5 轮≈25s 是可接受上限 |
| 4 | 系统提示"严格最多调一次 search" | 防 tool-call loop(见 §4),是 Agent 可控性的关键 |
| 5 | 兜底:轮次耗尽时强制回答 | 防"搜了但没答"的空返回;利用已有搜索结果至少给用户一个答案 |
| 6 | 不改 rag_answer() | 两种模式并存:Agent 和固定 RAG 可对比 |

## 4. 踩坑记录

### tool-call loop(本模块最大的坑)

**现象**:LLM 调了一次 search,拿到结果,然后**又调了一次 search**(换了个 query),
然后又调了一次……直到 max_iterations 用完。从不停下来回答。

**根因**:GLM-4.7-flash(类似很多小模型)对搜索结果有"总觉得不够"的倾向——
它看结果时会想"再搜搜会不会更好",而不是"已经有这些了,我来组织回答"。

**解法(两层)**:
1. **系统提示约束**(主要):明确告诉 LLM "最多调一次 search,拿到结果后必须立即回答"
2. **兜底机制**(保险):轮次耗尽时,把已有结果拼成 prompt,强制不带 tools 调一次 LLM,
   让它只能回答不能调工具

**本质**:这是 Agent 可控性问题——让 LLM 自主决策 ≠ 放任它决策。
"严格限制调用次数"是人给 Agent 画的边界,和"自主"不矛盾。

### 429 限流(反复出现)

chat() 已加指数退避重试(1s/2s/4s/8s/16s,共 5 次)。
免费模型的固有限制,重试通常能扛住。

## 5. 业界选型与取舍

### Agent 框架对比

| 候选 | 特点 | 适合场景 | 劣势(对本项目) |
|------|------|----------|------|
| **裸写(本项目)** | 一个 while + tool_calls,~100 行 | **学习机制**:完全看穿每个环节 | 无状态管理/可视化/持久化 |
| LangChain | 高层封装,`@tool` 装饰器,AgentExecutor | 快速搭简单 Agent | 抽象层厚,学不到底层 |
| **LangGraph** | 状态机有向图,条件分支,人工介入 | **生产级复杂 Agent**(业界首选) | 学习曲线陡,裸写后再学更好 |
| LlamaIndex Workflows | 检索+节点工作流 | 重检索型 Agent | RAG 检索已有,Agent 编排是新需求 |
| OpenAI Agents SDK | 轻量 Agent 循环 | 简单场景 | 锁定 OpenAI 生态 |

**取舍逻辑**:学习期裸写 → 理解机制后上 LangGraph(生产级)。
裸写的价值在于:面试时你能讲清"Agent 就是 while + tool_calls",
而不只是"我用 LangGraph 搭了一个 Agent"。

### 真实产品的 Agent 形态

- Dify: Workflow 可视化编排(本质也是图,有向图的可视化)
- RAGFlow: Agent 模式(检索+工具调用+对话管理)
- FastGPT: Flow 编排(节点连线式)
- 企业级:LangGraph 做底(状态机+条件分支+持久化+人工审批)

所有产品的 Agent 内核都是本项目这个循环——只是上面套了不同厚度的工程壳。

## 6. 与 RAG 的关系(不是替代,是升级)

```
固定 RAG:    始终 search → rerank → generate
             (每次提问,同样的路径)
Agent:       LLM 自主决定路径
             ├── 简单问题 → 直接回答(不搜索)
             ├── 需要检索 → search → 回答
             └── 复杂问题 → search → 追问 → search → 回答
```

固定 RAG 是 Agent 的**子集**:Agent 可以退化为固定 RAG(每次都调 search)。
但 Agent 多了"不检索"和"多轮检索"的灵活性。
`rag_answer()` 保留作对照——面试时能对比两种模式。

## 7. 多轮对话记忆(SPEC-004)

### 机制:LLM API 无状态,"记忆"是客户端伪造的

`chat()` 就是一次普通 HTTP POST——服务器处理完即忘,不存任何对话状态。
网页版 ChatGPT"记得上文",底层是**客户端每次把从头到现在的完整 messages 重发一遍**。
所以记忆的实现只有一个要点:**让 messages 列表跨 `run()` 调用存活**:

```python
ans1, hist = run("Wireshark 怎么解密 HTTPS 流量?")          # hist = [system, 问1, 答1]
ans2, hist = run("你说的第二步在哪个菜单?", history=hist)     # messages = hist + [问2]
```

### 核心设计:轮内完整、轮间压缩

```
轮内(工作列表,用完即弃):  [system, 问, assistant(tool_calls), tool(≈2500字符), 答]
轮间(返回的历史):          [system, 问, 答]      ← 中间消息从不进入历史
```

- **为什么压缩**:一次 search 回注 ≈2500 字符,是历史体积的绝对大头;
  5 轮不压缩 ≈ 上万字符死 token
- **为什么压缩后截断是安全的**:每轮压成 `[user, assistant]` 原子对,
  窗口截断(max_turns=10)退化为按对切片——**永远不会**把
  assistant(tool_calls) 和对应 tool 消息拆散(OpenAI 协议硬约束,拆散即 400),
  不需要复杂的"配对感知截断"
- **实现落点**:`_build_history(prev, question, answer, max_turns)` 纯函数,
  可离线单测(tests/test_history.py);question/answer 显式传入——
  兜底路径中工作列表里追加的是 fallback prompt,不是用户原始问题,不能靠遍历提取

### 实测现象:记忆与检索互补

第 2 轮追问"你说的第二步里的 SSL 协议设置在哪个菜单",LLM **同时**:
- 用**记忆**做指代消解("第二步"→ 第 1 轮的解密步骤,不用复述背景)
- 调**新 search** 查具体菜单位置(答案给出 Edit→Preferences→Protocols→SSL)

记忆管"上文说了什么",检索管"细节在哪"——二者不打架。

### 升级路径(Won't 但值得知道)

| 方案 | 做法 | 何时需要 |
|------|------|----------|
| 滑动窗口(本项目) | 最近 N 轮,超了整对丢 | 默认够用 |
| 摘要压缩 | 旧轮 LLM 压成一段摘要替换原文 | 轮数多且早期上下文仍关键 |
| 检索式记忆 | 历史存向量库,按当前问题检索注入 | 跨会话长期记忆(MemGPT 思路) |

## 8. 工具注册表与多文档检索(SPEC-005)

### 注册表:函数与 schema 绑在同一处声明

SPEC-003 的工具是"手写 JSON schema 拼在 TOOLS 列表 + `_exec_tool` 里 if-else 分发"
——加一个工具要改两处,漂移迟早发生。SPEC-005 改为注册式(`agent/tools.py`):

```python
@tool(name="calculator", description="计算数学表达式...", parameters={...})
def tool_calculator(expression: str) -> str:
    ...

get_tool_schemas()   # → LLM 的 tools 参数,自动生成
execute_tool(name, arguments)  # → 统一分发;任何失败转字符串回注,循环不崩
```

对照业界:LangChain 的 `@tool` 装饰器、OpenAI 的 function schema,本质相同——
把"LLM 看的 schema"和"代码执行的函数"绑在**同一处**。新增工具 = 写函数 + 一处声明。

新工具两个:**calculator**(ast 白名单安全求值,只放行数字与算术节点,
`import os`/`__import__`/变量名一律拒绝——永不 `eval` 裸字符串)、
**list_documents**(列知识库清单,LLM 可先看库里有什么再决定怎么搜)。

### search 工具升级为两段式(真机踩坑倒逼)

多文档检索(SPEC-005)上线当天就踩出"小文档名次压缩"(详见 04 详解 §9):
纯 RRF 序下,"Wireshark 解密 HTTPS"的 top1 被 3 块的小笔记文档抢走。
修法:agent 的 search 工具与 `rag_answer` 对齐为**两段式**——
`hybrid_search_all` 召回 top-10 → `rerank` 交叉编码器精排 top-5,
精排失败退回 RRF 序(不中断)。教训一句话:**跨文档融合的排序仅供参考,
精排才是最终裁判**。

## 9. Q&A 自测

### Q1 · Agent 和 RAG 的本质区别是什么?RAG 是 Agent 吗?
**难度: 基础** · 考点: 概念区分

> **你的回答**:



---

### Q2 · while 循环的终止条件是什么?如果 LLM 永远不返回 content(一直返回 tool_calls)会怎样?靠什么兜底?
**难度: 机制** · 考点: Agent 循环的边界

> **你的回答**:



---

### Q3 · tool_call_id 是什么?如果不传会怎样?它在多工具并行调用时有什么作用?
**难度: 机制** · 考点: OpenAI function calling 协议

> **你的回答**:



---

### Q4 · direct_answer 工具为什么必要?没有它,LLM 对"1+1 等于几"会怎么处理?
**难度: 决策** · 考点: 工具集设计

> **你的回答**:



---

### Q5 · 你遇到了 tool-call loop,它是什么?为什么发生?你用哪两层解法解决的?第二层为什么必要?
**难度: 面试** · 考点: Agent 可控性(真实踩坑,必问)

> **你的回答**:



---

### Q6 · LangGraph 相比你现在的裸写,多解决了哪些问题?说出至少三个。什么时候值得引入?
**难度: 面试** · 考点: 框架价值判断

> **你的回答**:



---

### Q7 · LLM API 是无状态的,那 ChatGPT 网页版的"记得上文"是怎么实现的?你的 Agent 的多轮记忆和它本质上有区别吗?
**难度: 面试** · 考点: 记忆的本质(高频面试题)

> **你的回答**:



---

### Q8 · 你的多轮记忆为什么"轮内保留 tool 消息、轮间压缩丢弃"?如果直接全量保留所有历史会怎样?压缩为什么让窗口截断变安全了?
**难度: 机制** · 考点: 压缩/截断的协同设计

> **你的回答**:
