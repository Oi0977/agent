# 06 · LLM 生成(generator:prompt 拼装 + 调用 + 全链路)

> 解决的问题:把检索到的证据组织成**忠实于资料**的自然语言回答。
> RAG 的最后一站——前面五段都是"准备",这里才"说话"。

## 1. 全链路编排(answer.py)

```python
def rag_answer(question, index_path, meta_path, k=3, recall_k=20):
    hits = search(question, index_path, meta_path, k=recall_k)  # 粗排召回 top-20
    hits = rerank(question, hits, top_k=k)                      # 精排精选 top-3
    prompt = build_prompt(question, hits)                       # 拼 prompt
    answer = chat(prompt)                                       # 调 LLM
    return {"answer": answer, "hits": hits}                     # hits 一并返回供引用展示
```

系统从"能检索"进化为"能问答"的完整闭环。
`hits` 随答案一起返回:前端/演示要展示"引用了哪些资料、各多相关"。

## 2. Prompt 设计(prompt.py)

```
你是一个严谨的技术文档问答助手。请只依据下面的参考资料回答…

## 参考资料
【资料1】(来源: xxx.pdf,第 142 块,相关性 0.729)
<原文>
【资料2】…

## 问题
{question}

## 回答要求
- 只基于参考资料作答,资料里没有的不要编造
- 如果参考资料不足以回答,直接说明"参考资料中没有相关内容"
- 条理清晰,涉及步骤时按顺序分点说明
```

**设计要点**:
- **"只依据资料 + 不会就说没有"** = 反幻觉的两道闸。RAG 的可信度不靠模型自觉,
  靠 prompt 把"边界"画死——宁可拒答,不可编造
- **资料头带来源/块号/相关性分**:LLM 引用有据可查,答案可溯源(如"根据参考资料【资料1】")
- 模板独立成文件:prompt 是 RAG 效果**最直接的旋钮**,迭代频繁,值得单独一处改

## 3. 薄抽象(llm_client.py):两行换供应商

```python
# ---- 供应商配置:换 LLM 只改这里 ----
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"   # 智谱;Ollama: http://localhost:11434/v1
MODEL = "glm-4.7-flash"                              # 本地: qwen2.5:7b
```

`chat(prompt)` 内部用 OpenAI SDK 调用。**OpenAI 兼容接口已是行业事实标准**
(智谱/DeepSeek/Ollama/vLLM 全兼容)——薄抽象不是绕过 SDK,而是把"换供应商"
这个变化点收敛到两行常量。本地/云端的调用代码结构完全相同。

**temperature=0.3**:RAG 问答要"忠于资料",低温减少自由发挥;创作类任务才需要高温。

**max_retries=5**:SDK 内置退避重试。免费模型(GLM-4.7-Flash)高峰期限流 429,
实测偶发,重试可扛。

## 4. .env 的裸加载(不引 python-dotenv)

```python
def _load_dotenv():
    # 读文件 → 逐行拆 KEY=VALUE → os.environ.setdefault(不覆盖已有)
```

原理三行讲完,不为一个小功能加依赖——和全项目"裸机制优先"的哲学一致。
`.env` 存 API Key(已 gitignore),`.env.example` 是入库模板。

## 5. 踩坑记录

| 坑 | 现象 | 解法 |
|----|------|------|
| 429 限流 | `该模型当前访问量过大` 偶发(免费款高峰) | `max_retries=5` + 验收脚本加 sleep |
| Key 硬编码风险 | — | .env(不入库)+ .env.example(模板);`_get_client()` 里缺失时给出人话报错 |

## 6. 业界选型与取舍

### 选型一:LLM 接入方式

| 候选 | 特点 | 劣势(对本场景) |
|------|------|------|
| **智谱 GLM-4.7-flash(本项目)** | 免费、中文好、128K 上下文、OpenAI 兼容 | 高峰 429 限流;效果弱于旗舰 |
| DeepSeek / GLM 旗舰(API) | 效果更强 | 付费(虽便宜);同样依赖网络 |
| 本地 Ollama + qwen2.5:7b | 完全本地、免费、离线 | 8G 显存只能跑 7B 档,复杂综合有代差;占资源 |

**本项目的取舍逻辑(决策过程实录)**:学习目标是先打通 RAG 结构,所以选
"起步最快 + 零成本 + 建立效果上限基线"——云端免费 API。**本地 Ollama 不排除**,
路线图里作为"效果对比 + 离线能力"的下一步;`llm_client` 的薄抽象保证切换成本
= 改两行(BASE_URL + MODEL)。**取舍的关键不是二选一,是让选择可逆。**

### 选型二:调 SDK 的方式

OpenAI SDK + 兼容 base_url(胜出)vs 各家原生 SDK vs 裸 HTTP:
OpenAI 兼容接口是行业事实标准(智谱/DeepSeek/Ollama/vLLM 全兼容),
一套代码通吃所有供应商,薄抽象只收敛"会变的两行"。

### 业界生成层的标配(本项目暂未做,进阶清单)

流式输出(打字机体验)、引用标注(citation:答案句 → 来源块)、
结构化输出(JSON schema 约束)、多轮对话(messages 历史拼接)。

## 7. 接口契约

```
build_prompt(question, hits) → str        # prompt 拼装
chat(prompt, temperature=0.3) → str       # LLM 调用
rag_answer(question, ..., k=3, recall_k=20)
    → {"answer": str, "hits": list}       # 完整链路,一步到答案
```

## 8. Q&A 自测(先凭理解答,再对照上文;可交 AI 批改)

> 在"你的回答"下直接写,写完自查或让 AI 批改。

### Q1 · prompt 里"只依据资料作答"和"资料不足就直说"分别防什么?哪条是反幻觉的第一道闸?
**难度: 基础** · 考点: prompt 设计原则

> **你的回答**:



---

### Q2 · temperature=0.3 为什么取低温?什么任务该取高温?这个参数在 RAG 和创作场景的分界逻辑?
**难度: 机制** · 考点: 采样参数

> **你的回答**:



---

### Q3 · "薄抽象"到底薄在哪?换 LLM 供应商要改几行?为什么能这么薄——哪个行业事实在支撑?
**难度: 决策** · 考点: 抽象设计

> **你的回答**:



---

### Q4 · RAG 系统的幻觉有哪些来源?分别对应链路的哪个环节、用什么手段治?(至少三个环节)
**难度: 面试** · 考点: 系统级反幻觉(高频面试题)

> **你的回答**:



---

### Q5 · 答案要标注"这句话出自哪个资料块",你需要动哪里?(提示:资料头里已有什么、hits 返回了什么)
**难度: 面试** · 考点: citation 的实现路径

> **你的回答**:



---

### Q6 · "先云端跑通再本地部署"这个顺序的学习价值在哪?反过来(先本地)会损失什么?
**难度: 面试** · 考点: 学习方法论与决策意识

> **你的回答**:


