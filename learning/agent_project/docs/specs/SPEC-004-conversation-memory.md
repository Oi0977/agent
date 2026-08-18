# SPEC-004 多轮对话记忆(跨轮 messages 持有 + 轮间压缩)

- **编号**: SPEC-004
- **类型**: 功能 · 影响面: 中(改 `run()` 签名,破坏性变更需同步调用方)
- **状态**: ✔ 已验收(2026-08-18)
- **创建**: 2026-08-18
- **关联**: SPEC-003(最小 Agent 循环;本 spec 扩展其 `run()`,且其 Won't 项"持久化会话记忆"中"轮内记忆"部分由此实现;验收后 SPEC-003 加修订记录)、详解 07

## 背景

SPEC-003 的 `run()` 每次调用新建 `messages = [system, user]`,函数返回即销毁——
两次调用之间零记忆:用户问"Wireshark 怎么解密 HTTPS?"得到答案后追问
"你说的第二步在哪个菜单?",LLM 完全不知道"第二步"指什么。

**根因**:LLM API 是无状态的(HTTP 请求处理完即忘),服务器不存对话。
"记忆"只能由**客户端把 messages 列表跨调用持有并重发**来实现。
本 spec 给 Agent 补上这一层,并顺带解决"历史越背越重"的 token 成本问题。

## 目标 / 非目标(MoSCoW)

**Must(必须)**:
- `run()` 接受上一轮返回的消息历史,回答**依赖上文的追问**时正确利用历史
- `run()` 返回 `(答案, 新历史)`,由调用方持有列表跨轮传递(裸函数哲学:机制摊开,不用 Memory 类)
- **轮内完整、轮间压缩**:含 tool_calls/tool 的完整工作列表仅本轮存活;
  返回的历史中,每轮只保留 `[user, assistant 最终答案]` 两条
- 压缩保持 OpenAI 协议完整:历史中不得出现孤儿 tool 消息或孤儿 tool_calls
- 历史构造/窗口截断是**纯函数**,可离线单测(不调 LLM)

**Should(应该)**:
- `max_turns` 窗口:超过 N 轮只保留最近 N 轮(system 永远保留);
  压缩使每轮成为原子对 → 截断只需按对切片,天然不会拆散协议配对
- verbose 模式打印"第 X 轮对话(历史 N 条消息)"
- main.py 的 Agent 演示段改为两轮追问

**Won't(不做)**:
- 不做磁盘持久化(会话结束即失忆,进程内记忆)
- 不做摘要压缩/检索式长期记忆(是本方案的升级路径,留给后续 spec)
- 不引入任何框架 Memory 组件(LangChain Memory 等)

## 核心机制

```
调用方(持有列表)                     run(question, history=messages)
    │                                       │
    │── history(上轮压缩后) ──────────────▶│ messages = history + [user]
    │                                       │ (history 已含 system,不再重复添加)
    │                                       │
    │                                       │ while Agent 循环(SPEC-003,不变)
    │                                       │   轮内 messages 完整回注
    │                                       │   (assistant(tool_calls) + tool 结果都在)
    │                                       │
    │                                       │ answer = 最终 content
    │◀── (answer, new_history) ────────────│ new_history = _build_history(...)
    │                                       │   = history + [user, assistant(answer)]
    │                                       │   ↑ 工作列表用完即弃,tool 中间消息不进入历史
    │                                       │   ↑ 超过 max_turns 轮时丢弃最早若干轮
    │── run(追问, history=new_history) ──▶│ (下一轮)
```

**为什么轮间压缩(本 spec 的核心设计决策)**:
- 一次 search 回注 ≈ 2500 字符,是历史体积的绝对大头;5 轮不压缩 ≈ 上万字符死 token
- 每轮压成 `[user, assistant]` 原子对后,窗口截断退化为按对切片——
  **永远不会**把 assistant(tool_calls) 和对应 tool 消息拆散(OpenAI 协议硬约束,
  拆散即 400),也不需要复杂的"配对感知截断"逻辑

## 验收标准(Given-When-Then)

- [x] AC1 — 跨轮记忆生效(真机)
  - Given 第 1 轮问"Wireshark 怎么解密 HTTPS 流量?"已得到答案,持有返回的历史
  - When `run("你说的第二步里的 SSL 协议设置,具体在哪个菜单打开?", history=历史)`
  - Then 第 2 轮发给 LLM 的 messages 含第 1 轮的问答内容;答案能正确指代上文
    (答出 Edit→Preferences→SSL 路径类内容,无需用户复述任何背景)

- [x] AC2 — 轮间压缩(离线,纯函数)
  - Given 一轮含 search 的完整工作列表
    `[system, user, assistant(tool_calls), tool, assistant(answer)]`
  - When 构造返回历史
  - Then 历史末尾两条为 `[user, assistant(content=answer)]`;不含任何 `role=="tool"` 消息;
    不含带 tool_calls 的 assistant 消息;system 在头部且仅一条

- [x] AC3 — 协议完整性(真机)
  - Given 第 1 轮曾发生工具调用、已压缩的历史
  - When 以它为 history 发起第 2 轮真实调用
  - Then API 调用成功(无孤儿 tool_call_id 引发的 400),正常返回答案

- [x] AC4 — 无历史时向后等价
  - When `run(question)` 不传 history(或传空列表)
  - Then 行为与 SPEC-003 一致:新建 `[system, user]` 起步,完成同样的循环;
    返回 `(answer, history)`

- [x] AC5 — 窗口截断(离线,纯函数)
  - Given system + 12 轮历史 + 本轮新问答(共 13 轮),`max_turns=10`
  - When 构造返回历史(截断发生在"拼接本轮之后",返回历史永不超限)
  - Then 返回历史 = system + 最近 10 轮(21 条消息);最早 3 轮被整对丢弃;无残缺轮
  (注:初稿手算"丢 2 轮"有误,离线单测暴露后修正 —— 截断在拼接本轮之后)

- [x] AC6 — 终端可观测
  - When verbose=True 且带 history 运行
  - Then 打印当前是第几轮对话、历史含多少条消息

## 涉及模块

- `agent/agent.py`:`run()` 签名变更(question, history=None, max_iterations, max_turns, verbose)
  + 轮间历史构造纯函数 + 窗口截断。
  **破坏性变更:返回 `str` → `(str, list)`**,grep 全部调用方同步修改
- `tests/test_history.py`(新):AC2/AC5 离线单测(纯函数,不调 LLM)
- `main.py`:Agent 演示段改为两轮追问(演示记忆生效)
- SPEC-003:验收后末尾加修订记录(`run()` 返回值变更,见 SPEC-004)

## 风险与兜底

- **破坏性变更**:`run()` 返回类型变了 → 动手前 grep 调用方(main.py / tests / temp),
  全部同步改,不许留旧用法
- **指代消解依赖模型能力**:GLM-4.7-flash 对弱指代(只有"它呢?"这种)可能仍需重述;
  AC1 特意用带具体线索的指代("你说的第二步里的 SSL 协议设置")验收
- **token 增长**:max_turns 窗口兜底;单轮答案本身很长时历史仍可能偏大,学习项目可接受
- **429 限流**:chat() 已有指数退避(1/2/4/8/16s),两轮真机验收间隔加 sleep

## 实现备注(实现后回填)

- **AC1/AC3/AC4 实测**(2026-08-18,temp/test_multi_turn.py 真机两轮):
  - 第 1 轮"Wireshark 怎么解密 HTTPS 流量?"→ 自主 search → 返回五步解密流程
  - 第 2 轮追问"你说的第二步里的 SSL 协议设置,具体在哪个菜单打开?"→
    **正确指代上文**,答案区分了"查看 SSL 流(Follow Stream)"与"配置 SSL 协议"两种意图,
    给出 `Edit→Preferences→Protocols→SSL` 菜单路径,用户零背景复述
  - 第 2 轮返回历史恰好 5 条 `[system, 轮1问答, 轮2问答]`,压缩生效;
    含工具调用的历史二次真实调用无 400(AC3)
- **AC2/AC5 离线单测**:`tests/test_history.py` 全过(纯函数,不调 LLM)
- **AC5 手算错误**:spec 初稿写"丢 2 轮",单测跑出实际丢 3 轮 ——
  截断发生在**拼接本轮之后**(13 轮留 10),已修正 spec。这是继 SPEC-002 之后
  第二次"可执行验收暴露纸面算术错",证明"AC 要能跑"的价值
- **有趣观察**:第 2 轮 LLM **同时用了记忆**(理解"第二步"指什么)和**新 search**
  (查具体菜单位)—— 记忆负责指代消解,检索负责最新细节,二者互补不打架
- **实现要点**:`SYSTEM_PROMPT` 从 run() 内提为模块常量(_build_history 重建历史要用);
  `agent/__init__.py` 补薄门面导出 `run`(此前为空,导入会失败);
  兜底路径的历史由显式 question/answer 构造(工作列表里的 fallback prompt 不等于原始问题)
- **调用方影响**:grep 确认 `run()` 此前无外部调用方,破坏性变更零波及;
  main.py 补阶段六两轮演示(代码与已验证的 temp 脚本一致)
