# agent_project — 从零手写的 RAG 问答系统

> 学习目标:不依赖重型框架,裸写 RAG 全链路,理解每个环节的机制;
> 在此基础上逐步演化为真正的 Agent 系统,最终成为可用的产品。

## 一、项目概览

本项目实现了一个完整的 **RAG(检索增强生成)问答系统**:把文档变成可检索的向量库,
用户提问后系统粗排召回 + 精排重排找到相关内容,交给 LLM 生成忠实于原文的答案。

**技术选型**(刻意选轻量、可理解的组件):

| 环节 | 本项目用的 | 为什么 |
|---|---|---|
| 文档解析 | pdfplumber + RapidOCR | 有文本层直接提取,无文本层自动降级 OCR |
| 文本分块 | langchain-text-splitters | 唯一引入的 langchain 组件,递归字符分割 |
| 向量嵌入 | BGE bge-small-zh-v1.5(本地) | 中文效果好、512 维、模型小加载快 |
| 向量检索 | FAISS IndexFlatIP | 只存向量、精确暴力搜索,机制透明 |
| 重排序 | BGE bge-reranker-base(本地) | 交叉编码器精排,业界 RAG 标配 |
| LLM 生成 | 智谱 GLM-4.7-flash(远程 API) | 免费、中文好、OpenAI 兼容接口 |

## 二、模型资产(统一缓存在 `S:\huggingface_cache`)

所有 HuggingFace 模型统一下载到本机 `S:\huggingface_cache`(**不用**系统默认的
`C:\Users\...\.cache`)。缓存结构(HuggingFace 标准布局):

```
S:\huggingface_cache\
├── models--BAAI--bge-small-zh-v1.5\      ← 嵌入模型(~100MB,双塔 bi-encoder)
│   └── snapshots\<hash>\                 ← 模型本体(config/权重/tokenizer)
└── models--BAAI--bge-reranker-base\      ← 重排模型(~1.1GB,交叉编码器 cross-encoder)
    └── snapshots\<hash>\
```

代码加载方式:解析本地快照目录路径直接喂给模型库,**彻底离线**。
新模型下载方法(国内走 hf-mirror 镜像,需禁用镜像不支持的 Xet 协议):

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("模型名", cache_dir=r"S:\huggingface_cache")
```

## 三、架构总览

```
【离线·建库】只需跑一次
  PDF → ① parse_document 解析 → ② smart_chunk_text 分块
      → ③ embed_texts 嵌入 → ④ build_index 落盘(.index + .json)

【在线·问答】每次提问都跑
  问题 → ⑤ embed_query(加BGE前缀) → ⑥ hybrid_search 混合粗排 top-20(向量+BM25+RRF)
      → ⑦ rerank 精排 top-3 → ⑧ build_prompt → ⑨ chat(LLM)→ 答案
```

```
src/agent_project/
├── main.py            # 演示入口(各阶段验收,已有索引则复用)
├── ingest.py          # 入库命令:python -m agent_project.ingest <文件...> [--list]
├── path_manager.py    # 统一路径管理
├── preprocessor/      # 【阶段1】文档解析(pdf/md 自适应)
├── chunker/           # 【阶段1】文本分块
├── embedder/          # 【阶段2】向量嵌入(BGE)
├── retriever/         # 【阶段3】检索(单文档混合 + 多文档 2×D 路 RRF)
├── reranker/          # 【阶段5】精排重排(cross-encoder)
├── generator/         # 【阶段4】LLM 生成(智谱 API)
└── agent/             # 【阶段6】Agent 循环 + 多轮记忆 + 工具注册表
```

## 四、文档索引

| 想了解 | 去哪 |
|---|---|
| 每个模块的机制/代码走读/踩坑/业界对照 | [docs/架构详解/00-总览与导读.md](docs/架构详解/00-总览与导读.md)(入口,含阅读路线) |
| ① 解析:文本层/OCR 降级、RapidOCR | docs/架构详解/01 |
| ② 分块:递归分割/重叠设计 | docs/架构详解/02 |
| ③ 嵌入:双塔/BGE 前缀/本地加载 | docs/架构详解/03 |
| ④ 检索:FAISS 哲学/双产物 | docs/架构详解/04 |
| ⑤ 重排:交叉编码器/两段式/实证 | docs/架构详解/05 |
| ⑥ 生成:prompt 设计/薄抽象 | docs/架构详解/06 |
| ⑦ Agent 循环:while+tool_calls/tool-call loop/多轮记忆/框架选型 | docs/架构详解/07 |
| ⑧ Token 与上下文预算:token 原理/usage/估算与预算 | docs/架构详解/08 |
| 功能行为契约(SDD) | [docs/specs/](docs/specs/README.md) |
| 开发流程与环境铁律 | 仓库根 CLAUDE.md |

## 五、从 RAG 到 Agent 的路线图

```
✅ RAG 全链路(解析→分块→嵌入→混合检索→精排→生成)
✅ Agent 初版(裸写 while + tool_calls,LLM 自主决策)
✅ 多轮对话记忆(轮间压缩 + 窗口截断,SPEC-004)
✅ 多文档知识库 + 工具注册表 + 入库命令(SPEC-005)
✅ Token 记账与上下文预算(SPEC-006)
✅ 交互式会话终端 + 会话持久化(SPEC-007)
    │
    ▼
【下一步】LangGraph 生产级 Agent(状态机+条件分支+持久化)
    │
    ▼
【最终形态】多轮对话 Agent(检索 + 生成 + 记忆 + 工具调用 + 人工介入)
```

Agent 框架选型到 Agent 阶段再评估(LangGraph 状态机式编排是届时主要候选),
现在裸写的目的是先理解机制。

## 六、运行指南

### 环境

- Python:Anaconda `learning` 环境(`conda activate learning`)
- API Key:复制 `.env.example` 为 `.env`,填入智谱 API Key

### 知识库管理(入库/列表)

```bash
python -m agent_project.ingest 你的文档.pdf 另一份.md   # 入库(同名覆盖)
python -m agent_project.ingest --list                  # 列出已入库文档
```

### 交互式会话终端

```bash
python -m agent_project.chat    # 持续多轮对话;每轮显示 token 统计
```

会话内命令:`/new` 新会话 · `/save [名]` 保存 · `/load <名>` 载入 · `/list` 列已存会话 · `/exit` 退出。

### 快速验证(复用已有索引)

```python
from agent_project.generator import rag_answer
from agent_project.path_manager import PathManager

pm = PathManager()
index_path = next(pm.OUTPUT_DIR.glob("*.index"))
meta_path = next(pm.OUTPUT_DIR.glob("*.json"))

result = rag_answer("Wireshark 里怎么解密 HTTPS 流量?", index_path, meta_path, k=3)
print(result["answer"])
```

### 多轮对话(Agent + 记忆 + token 记账)

```python
from agent_project.agent import run

ans1, hist, st1 = run("Wireshark 怎么解密 HTTPS 流量?")
ans2, hist, st2 = run("你说的第二步在哪个菜单打开?", history=hist)  # 追问依赖第 1 轮
print(ans2, st2)   # st2 含 prompt/completion token 统计(SPEC-006)
```

### 完整流程(重新建库)

```python
from agent_project.retriever import build_index
index_path, meta_path, n_chunks = build_index("你的文档.pdf")
```

### 运行 main.py(全阶段演示)

```bash
conda activate learning
python -m agent_project.main
```

## 七、踩坑速查(细节见各详解的"踩坑记录"节)

| 坑 | 一句话解法 | 详见 |
|---|---|---|
| FAISS 中文文件名报错 | 产物名转 ASCII+哈希 | 详解04 |
| 模型启动联网检查崩溃 | 本地快照路径直载 | 详解03 |
| HF 离线开关不生效 | 双开关(HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE),或路径直载 | 详解03 |
| 下载模型 Xet 401 | `HF_HUB_DISABLE_XET=1` | 详解05 |
| BGE 检索效果差 | query 必须加前缀(embed_query 已内置) | 详解03 |
| numpy 喂 FAISS 报错 | C-contiguous float32 | 详解04 |
| FAISS id=-1 占位 | 过滤 `0 <= i < len(chunks)` | 详解04 |
| 免费 API 429 限流 | max_retries=5 + 重试 | 详解06 |
| conda 装"小"包后行为大变 | 可能连带升级全家桶,装完回归验证 | 详解03 |
| 脚本启动慢(~25s) | langchain_text_splitters 首次 import 拖全家,非 bug | 详解04 |
| 多文档检索小文档抢榜首 | RRF 小文档名次压缩,靠精排纠序(search 工具已两段式) | 详解04§9 |

---

*文档分工:specs 管"该怎样"(契约),架构详解管"怎么实现"(机制),本 README 是入口。
当前版本对应:RAG 六段 + Agent 循环 + 多轮对话记忆(SPEC-004)完成。*
