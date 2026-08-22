# SPEC-006 Token 记账与上下文预算

- **编号**: SPEC-006
- **类型**: 功能 · 影响面: 中(llm_client 捕获 usage + agent 记账/预算 + run 签名)
- **状态**: ✔ 已验收(2026-08-19)
- **创建**: 2026-08-19
- **关联**: SPEC-004(历史构造 `_build_history` 在其上扩展预算截断)、详解 08(理论地基)

## 背景

SPEC-004 的窗口按**轮数**截断(max_turns=10),但 10 轮可能是 2k token 也可能是 20k token;
项目至今没有任何 token 数字:API 响应里**白送的精确 usage 一直被丢弃**。
无记账 ⇒ 成本不可见;无预算 ⇒ "何时截断"只能拍脑袋。
理论背景见详解 08:token 是模型私有单位,事前只能估算、事后必有真数。

## 目标 / 非目标(MoSCoW)

**Must**:
- `run()` 捕获每次 LLM 调用返回的 `usage`,累计会话统计并随返回值输出:
  `(answer, history, stats)`,stats = `{"llm_calls", "tool_calls", "prompt_tokens",
  "completion_tokens", "history_turns"}`(历史轮数)
- verbose 每次调用后打印:`[token] prompt P / completion C(会话累计 T)`
- `_estimate_tokens(text)` 纯函数:中文≈1字1token,非中文按字符数/4(GLM 中文优化
  词表下成立;启发式是分词器相对的,详解 08 §5)
- `max_history_tokens`(默认 8192):`_build_history` 在 max_turns 切片后估算历史体积,
  超预算从最旧轮开始**整对丢弃**直到回到预算内;system 永不丢;至少保留最近 1 轮;
  max_turns 保留为第二道闸
- 估算与预算截断均纯函数,离线单测

**Should**:
- verbose 打印"本轮为历史重复支付约 X token(估算)"

**Won't**:
- 摘要压缩(升级路径,后续 spec)
- 精确分词器(tiktoken/transformers)
- 金额换算(人民币成本)

## 验收标准(Given-When-Then)

- [x] AC1 — 估算函数(离线)
  - Given 纯中文 100 字 / 纯英文 400 字符 / 混合文本
  - When _estimate_tokens()
  - Then 中文≈100(±10%);英文≈100(±30%);混合值介于按比例叠加;
    任意文本加长估算单调不减

- [x] AC2 — 预算截断(离线)
  - Given system + 多轮超长历史(估算超 8192),max_history_tokens=8192
  - When _build_history()
  - Then 从最旧轮整对丢弃直至入预算;system 保留;至少留最近 1 轮;
    无残缺对;返回的历史估算值 ≤ 预算(或仅剩 1 轮)

- [x] AC3 — 记账累计(真机)
  - Given 两轮对话
  - When 每轮结束
  - Then stats.prompt_tokens 为本轮各次 LLM 调用 usage.prompt_tokens 之和;
    第 2 轮单次调用的 prompt_tokens > 第 1 轮(历史重发生效,可观测)

- [x] AC4 — 签名与调用方(离线 grep + 真机)
  - When run() 返回三元组
  - Then 所有调用方(main.py/tests/temp 脚本)同步更新,grep 无旧二元组用法残留

- [x] AC5 — 小预算端到端(离线单测)
  - Given max_history_tokens 设为极小值(如 50)
  - When 构造多轮历史
  - Then 返回历史仅 system + 最近 1 轮(证明预算闸真实生效,非摆设)

## 涉及模块

- `agent/agent.py`:usage 捕获与累计、stats 返回、`_estimate_tokens`、
  `_build_history` 加 max_history_tokens、verbose 输出
- `generator/llm_client.py`:无需改(usage 已在 ChatCompletion 响应中)
- `main.py`:阶段六演示适配三元组
- `tests/test_token_budget.py`(新):AC1/AC2/AC5

## 风险与兜底

- **GLM usage 字段缺失**:个别供应商兼容层可能不返回 usage → 捕获时 `getattr` 防御,
  缺失记 0 并打印提示,不影响主流程
- **估算偏差**:±20-30% 对预算阈值无碍(阈值本身留有余量);记账层不受影响(真数)
- **chat() 返回 None 的路径**:重试耗尽会 raise,run() 不捕获(现有行为不变)

## 实现备注(实现后回填)

- **AC1/AC2/AC5 离线**:`tests/test_token_budget.py` 全过(每轮 202 tok 的人造历史,
  手算 12轮+本轮 → 双闸后剩 4 轮,与代码一致)
- **AC3/AC4 实测**(2026-08-19,两轮真机):
  - 第1轮:llm_calls=2, prompt 621→1417(search 回注 ≈ +800 tok), 合计 2038/473
  - 第2轮:llm_calls=3, prompt 939→1707→2462(历史重发 + 每次工具结果再叠加), 合计 5108/476
  - 三元组返回正常;usage 字段 GLM-4.7-flash 稳定返回,无需防御分支触发
- **★ 实测校准:启发式偏保守约 1.9 倍** —— 历史估算 591 tok vs 实际重发 318 tok:
  GLM-4.7 中文常用词会合并成单 token(如"流量""协议"),实际 < 1字1token。
  对预算用途是**安全方向**(提前截断而非超限);记账用 API 真数不受影响。
  若换模型应重校准比率(见 08 详解 §6)
- verbose 输出:"[token] 历史重发约 N tok(估算,预算 8192)" + 每次调用
  "[token] prompt P / completion C(本轮累计 X/Y)" —— 全链路 token 可见
