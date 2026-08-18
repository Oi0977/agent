# SPEC-003 最小 Agent 循环

- **编号**: SPEC-003
- **类型**: 功能 · 影响面: 完整
- **状态**: ✔ 已验收(2026-08-18)
- **创建**: 2026-08-18
- **关联**: SPEC-001(检索契约,search/hybrid_search 是工具来源)、详解 06(LLM 生成)

## 背景

RAG 流水线已完成(混合检索+精排+生成),但仍是**固定链路**:每次提问都执行同样的
search → rerank → generate,LLM 无法自主决策。Agent 化的核心一步是把检索函数
包装成 LLM 可调用的"工具",让模型根据问题自主决定是否需要检索、检索什么、
是否需要多轮——从"程序驱动"进化为"模型驱动"。

本 spec 定义**最小可行的 Agent 循环**:裸写 while + tool_calls,不引入任何框架。
目标是**理解 Agent 机制本身**,框架(LangGraph 等)留到下阶段。

## 目标 / 非目标(MoSCoW)

**Must(必须)**:
- LLM 接收问题后,能自主决定调用 search 工具
- 工具执行结果自动回注 LLM 上下文,LLM 据此组织最终回答
- 最大迭代轮次限制(防无限循环)
- 每轮的工具调用和结果打印到终端(可观测)

**Should(应该)**:
- 两个工具:search(向量+BM25 混合检索)和 direct_answer(不需要检索直接回答)
- 工具 schema 用 OpenAI function calling 格式(JSON Schema)

**Won't(不做)**:
- 不做多工具并行调用
- 不做流式输出
- 不做持久化会话记忆
- 不引入 LangChain/LangGraph
- 不改现有 `rag_answer()`(固定链路保留作对照)

## 核心机制(裸写 Agent 循环)

```
while 未达最大轮次:
    response = LLM(messages, tools=[search_schema, direct_answer_schema])
    if response 有 tool_calls:
        for each tool_call:
            result = 执行对应 Python 函数(tool_call.arguments)
            messages.append(tool_result(result))
    else:
        return response.content   # LLM 决定不再调工具,直接给出答案
```

**本质**:Agent 不是魔法,就是一个 **while 循环 + LLM 的 function calling 能力**。
LLM 每轮输出两个可能之一:①"我要调某个工具"(返回 tool_calls) 或 ②"我有答案了"(返回 content)。
代码只需判断"有没有 tool_calls",有就执行并回注,没有就返回。

## 验收标准(Given-When-Then)

- [x] AC1 — 单轮工具调用
  - Given Agent 可用工具:search
  - When 用户问"Wireshark 怎么解密 HTTPS 流量?"
  - Then LLM 返回 tool_calls,tool 为 search;执行后结果回注;LLM 最终返回文本答案

- [x] AC2 — 不需检索时直接回答
  - Given 用户问"1+1 等于几?"
  - Then LLM 不调用 search,直接返回文本答案(无需检索)

- [ ] AC3 — 多轮工具调用
  - Given 用户问"先搜一下 pcapng 文件格式,再搜一下 pcap 格式,然后比较两者区别"
  - Then LLM 至少执行两次 search(两轮不同 query),最终综合两次结果回答

- [ ] AC4 — 最大轮次限制
  - Given max_iterations=5
  - When LLM 连续 5 轮都返回 tool_calls
  - Then 第 5 轮强制终止并返回最后一条文本(或超时提示),不无限循环

- [ ] AC5 — 终端可观测
  - When Agent 运行
  - Then 终端依次打印:用户问题→LLM 思考/工具调用→工具结果→…→最终答案

- [ ] AC6 — 最小 Agent 循环纯函数(不依赖 LLM,可离线验证)
  - Given tools 配置 + mock LLM 响应
  - When 循环执行
  - Then 工具被正确调度、结果被正确回注、messages 最终状态正确

## 涉及模块

- `generator/llm_client.py`:扩展 chat() 支持 `tools` 参数+返回 tool_calls 解析
- `agent/agent.py`(新):最小 Agent 循环 + 工具注册
- `agent/__init__.py`(新)

## 风险与兜底

- **GLM-4.7-flash function calling 兼容性**:OpenAI SDK 调智谱走兼容接口;
  智谱支持 function calling;若工具 schema 太复杂被拒绝,简化到最小字段
- **工具执行失败**:search() 报异常时把错误信息作为 tool_result 返回给 LLM,
  让它自己判断是否重试或换策略(LLM 的鲁棒性)
- **死循环**:max_iterations 硬上限兜底
- **延迟叠加**:每轮 = 1 次 LLM 调用 + 1 次 search;3 轮 ≈ 10-15s

## 实现备注(实现后回填)

- **AC1 实测**(2026-08-18):LLM 收到"Wireshark 里怎么解密 HTTPS 流量?"→
  自主调 search(query="Wireshark 解密 HTTPS 流量")→ 拿到 2208 字符 →
  直接回答,输出完整 5 步解密流程(含 Edit→Preferences→SSL 路径)
- **AC2 实测**:收到"1+1 等于几?"→ 调 direct_answer → 返回"1+1 等于 2。"
- **踩坑:tool-call loop**(LLM 不停调 search 从不回答):GLM-4.7-flash 对搜索结果
  有"总觉得不够"的倾向,会反复换 query 搜索。解法:①更强的系统提示
  ("严格限制每次对话最多调用一次 search")②兜底机制:轮次耗尽时把已有
  搜索结果拼成最终 prompt 强制回答(不调工具)
- **GLM-4.7-flash function calling 兼容性**:完全兼容 OpenAI SDK 的 tools/tool_calls
  协议,无需任何适配层
- **max_retries=5(指数退避)**加到 chat():免费模型 429 限流偶发,重试可扛
