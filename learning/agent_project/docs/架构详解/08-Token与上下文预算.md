# 08 · Token 与上下文预算(跨模块基础)

> 解决的问题:上下文窗口、API 计费、记忆截断时机、分块大小上限——**全都以 token 计量**。
> 没有 token 意识,上下文管理只能按轮数拍脑袋(SPEC-004 的 max_turns),成本不可见,
> "何时压缩"无从谈起。本篇是 SPEC-006(token 记账与上下文预算)的理论地基。

---

## 1. 核心机制:token 到底是什么

**一个 token = 文本被某个分词器(tokenizer)切出来的一个片段。**
分词器 = **词表(vocabulary)+ 切分算法**,两者都是**模型训练的产物**,不是国际标准。

### 1.1 词表是模型的"基因"

模型嵌入层有且只有 `vocab_size` 行,模型**只能读自己词表里的 token**:

| 模型 | 词表大小(约) |
|---|---|
| GLM-4 | 15 万 |
| Qwen2.5 | 15 万 |
| Llama 3 | 12.8 万 |
| GPT-4o(o200k_base) | 20 万 |

GLM 切出的 token id 拿去喂 GPT,对它就是乱码编号——**token 不可跨模型互通**。

### 1.2 确定性,但只"相对"于单一分词器

- 同一段文本 + 同一个分词器 → **永远同样的 token 数**(确定性,不波动)
- 同一段文本 + 不同分词器 → **不同 token 数**(相对性)

所以"这段话多少 token"这个问题天然不完整,完整形式是"这段话**在 XX 分词器下**多少 token"。

### 1.3 切分算法:BPE 家族一句话原理

主流分词器都是 BPE(Byte Pair Encoding)血统:**从字节起步,反复合并语料中最高频的
相邻对,学出词表**。变体(WordPiece/Unigram/SentencePiece)思路同族。效果:

- 英文 ≈ 4 字符/token(`Wireshark` 可能是 1~3 个 token)
- 中文在**中文优化词表**(GLM/Qwen)≈ 1 字/token(`解密` = 1 token)
- 中文在**老英文词表**(Llama2 类,词表里没汉字)走字节回退:1 汉字 = 3 UTF-8 字节
  = **3 个 token**(`解密` = 6 token)

```
"Wireshark 解密 HTTPS"
  中文优化词表:  ≈ 5 token        老英文词表:  ≈ 9 token
```

**同一段话,token 成本差一倍以上**——多模型路由和成本估算时的真实问题。

### 1.4 货币类比(方向容易搞反)

| | 货币世界 | token 世界 |
|---|---|---|
| 单位 | 美元/人民币,**统一定义** | 都叫 "token",但**每家一个定义** |
| 换算 | 全球统一汇率,随市场波动 | 文本→token **确定不波动**,但**各家比率不同** |
| 互通 | 可换汇 | GLM token 与 GPT token **不可互换** |

token 不是通用货币,是**每家模型自己的私有货币**,各家汇率(词表)不同。

## 2. 开源 vs 闭源:算法同源,发布方式不同

| | 开源模型 | 闭源 API |
|---|---|---|
| 分词器 | **随权重发布**(HF 上 `tokenizer.json`),服务器用的就是这份文件 | 锁在服务端,只在响应里报结果 |
| 本地精确计数 | ✅ 可以一字不差(同一份词表) | ❌ 只能启发式估算(OpenAI 的 `tiktoken` 是罕见例外:主动发布了) |
| 精确数字来源 | 本地算或响应 usage | 响应 `usage` 字段 |

趋势:各家词表在**趋同**(容量奔 10~25 万、字节回退成标配、多语言覆盖),
比率越来越接近,但永远不会相同——词表是各家语料和训练目标的产物。

## 3. usage:API 白送的精确账单

服务端推理的**第一步就是分词**(不分词没法喂模型),所以它"顺便"就知道确切数量,
返回 usage 零成本——这也是**所有 API 都返回 usage 的根本原因:计费依赖它**。

| 服务 | 位置 | 字段名 |
|---|---|---|
| OpenAI 兼容(智谱/DeepSeek/Kimi/vLLM…) | `response.usage` | `prompt_tokens` / `completion_tokens` |
| Anthropic | `response.usage` | `input_tokens` / `output_tokens` |
| Ollama 原生接口 | 响应体 | `prompt_eval_count` / `eval_count` |
| OpenAI 推理模型 | usage 细分 | 额外 `reasoning_tokens`(思考消耗) |

**自己部署开源模型也不用自己算**:只要通过推理框架 serving(vLLM/Ollama/
llama.cpp server/TGI),框架同样要分词才能推理,照样返回 usage。

### 3.1 真正需要客户端自己算的只有一类场景:**发送前**

API 的 usage 是"事后"数字,但"这轮历史会不会超预算、要不要先截断"必须**事前**决策
——这正是估算函数存在的理由。同理:成本预估、按嵌入模型上限切文本(本项目的
chunk_size=500 字符就是给 BGE 的 512 token 上限留余量,按"中文≈1字1token"校准)。

### 3.2 就算词表对了也有坑:聊天模板开销

角色标记、特殊符号也占 token,所以**裸文本计数永远略低于实际 prompt_tokens**。
⇒ 业界标准做法:**事前用估算做预算决策,事后以 API usage 为准记账**。

## 4. 代码走读(SPEC-006 落地)

三层各对应一段代码(全在 `agent/agent.py`):

**① 记账(真数)**——每次 `chat()` 后捕获 API 白送的 usage:

```python
def _record_usage(response):
    usage = getattr(response, "usage", None)     # 防御:兼容层可能不返回
    ...
    stats["prompt_tokens"] += p or 0             # 本轮各次调用累计
```

verbose 效果(2026-08-19 真机两轮实测):

```
第1轮: [token] prompt 621 / completion 70     ← system+问题
       → search(...)                          ← 工具回注 ≈ +800 tok
       [token] prompt 1417 / completion 403
第2轮: [token] 历史重发约 591 tok(估算,预算 8192)
       [token] prompt 939 / completion 64     ← 621+历史重发 318(真数)
       [token] prompt 1707 / ...              ← 又一次工具结果叠加
       [token] prompt 2462 / completion 365
```

读数即结论:**轮内成本大头是工具结果回注(+800),轮间成本是历史重发(+318)**
——这就是 SPEC-004"轮间压缩丢弃 tool 消息"省的钱。

**② 估算(事前)**——纯函数,预算决策用:

```python
def _estimate_tokens(text):
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")   # 中文≈1字1token
    return cjk + (len(text) - cjk + 3) // 4              # 其余≈4字符/token
```

**③ 预算闸**——`_build_history` 双闸截断(max_turns 切完再切 token):

```python
while len(pairs) > 1 and _pairs_tokens(pairs) > max_history_tokens:
    pairs = pairs[1:]        # 从最旧轮整对丢弃;system 不计不丢;至少留最近 1 轮
```

`run()` 返回三元组 `(answer, history, stats)`,stats 含
`llm_calls / tool_calls / prompt_tokens / completion_tokens / history_turns`。

## 5. 设计决策与理由

| # | 决策 | 理由 |
|---|---|---|
| 1 | 记账用 API usage,预算用本地启发式 | 事前只能估(闭源无本地词表),事后必有真数;两者职责不同 |
| 2 | 启发式:中文≈1字1token,非中文按字符数/4 | GLM 是中文优化词表,该近似成立;**换英文老词表模型即失准**——启发式是分词器相对的 |
| 3 | 不引入精确分词器(tiktoken/transformers) | 为一个预算阈值引依赖不值;换本地开源模型时可升级为精确(加载官方 tokenizer.json) |
| 4 | max_turns 保留,与 max_history_tokens 双闸 | 轮数闸防"轮数多但都短"的会话无限长;token 闸防"单轮超长"撑爆预算 |

**记忆的成本形状**:全量重发历史 ⇒ 第 n 轮要为前 n-1 轮内容再付一次费,
**会话累计成本随轮数近似平方增长**。预算截断就是给这个上闸——记账让这件事
从"感觉"变成"看见"。

## 6. 踩坑记录

- **★ 启发式偏保守约 1.9 倍(实测校准)**:历史估算 591 tok vs API 实际重发 318 tok。
  原因:GLM-4.7 的中文优化词表会把常用词合并成单 token("流量""协议"≈1 token/词),
  实际 < 1字1token。对**预算**用途是安全方向(宁可提前截断,不会超限);
  **记账**用 API 真数,不受影响。换模型必须重校准比率——这就是 §5 决策 2
  "启发式是分词器相对的"的实测证据。
- **chat 模板开销真实存在**:估算只算 content,角色标记/特殊符号不计,
  所以估算天然略低于实际 prompt(上面 591 vs 318 是反方向,因为词表合并更强)。
  两股偏差方向相反,预算阈值留余量即可,不必精修。
- **usage 防御**:个别 OpenAI 兼容层可能不返回 usage → `getattr` 判空记 0 并提示,
  不炸主流程(智谱 GLM-4.7-flash 实测稳定返回)。

## 7. 业界选型与取舍

| 候选 | 是什么 | 取舍 |
|---|---|---|
| **API usage(本项目)** | 服务端白送的真数 | 零成本、精确;但只能事后 |
| tiktoken | OpenAI 官方分词器库 | 仅 OpenAI 系精确;中文场景非其词表 |
| HF tokenizers / SentencePiece | 开源模型的分词器运行时 | 配本地开源模型可做到与服务器一字不差 |
| 各云厂商 counting API | 提交文本返回 token 数 | 多一次网络往返,适合精确预检 |
| LangChain usage callbacks | 框架层自动聚合各回调 | 方便但藏机制;裸写阶段先手搓一遍 |

**升级触发条件**:换本地开源模型(Ollama+Qwen)时,加载其官方分词器,
把估算升级为精确。

## 8. Q&A 自测

### Q1 · token 的定义是什么?为什么不同模型的 token 数不一样?"token 是通用单位"错在哪?
**难度: 基础** · 考点: 概念

> **你的回答**:



---

### Q2 · 同一段中文,为什么在 Llama2 类模型上的成本可能是在 Qwen/GLM 上的 2~3 倍?
**难度: 机制** · 考点: 字节回退

> **你的回答**:



---

### Q3 · "128k 上下文窗口"的 128k 是什么单位?字符、字还是 token?为什么?
**难度: 基础** · 考点: 窗口语义

> **你的回答**:



---

### Q4 · 客户端怎么做 token 预算?开源模型和闭源 API 的做法有什么本质差异?聊天模板开销指什么?
**难度: 决策** · 考点: 预算工程

> **你的回答**:



---

### Q5 · Agent 多轮对话全量重发历史,会话累计 token 成本随轮数怎么增长?怎么控制?
**难度: 面试** · 考点: 成本形状(高频)

> **你的回答**:



---

### Q6 · 为什么所有 LLM API 都返回 usage?自己部署的开源模型需要自己算 token 吗?
**难度: 机制** · 考点: usage 的来源

> **你的回答**:
