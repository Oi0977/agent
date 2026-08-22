# AI Agent · RAG 全栈学习项目

从零构建一个**端到端的 RAG（检索增强生成）Agent**，不套框架、逐层手写，覆盖从文档解析到多轮对话的完整链路。

## 架构总览

```
用户提问
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Agent 循环（LLM 自主决策 · 多轮 · 工具调用 · 记忆）         │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ LLM 生成 ◄─── Prompt 组装 ◄─── 上下文压缩 ◄─── 检索结果 ││
│  │    │                                                   ││
│  │    ▼                                                   ││
│  │ 工具调用判断 ──yes──► search 工具 ──► 向量检索 ──┐      ││
│  │    │                                        │      ││
│  │    no                                       ▼      ││
│  │    │              混合检索（向量 + BM25 + RRF）│      ││
│  │    ▼                                        ▼      ││
│  │  最终回答 ◄─── 精排重排（Cross-Encoder）     │      ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## 模块详解

| 阶段 | 模块 | 职责 | 核心技术 |
|------|------|------|----------|
| 1 | 文档解析 | PDF/Markdown → 纯文本 | pdfplumber + RapidOCR（自适应） |
| 2 | 文本分块 | 长文本 → 语义完整的块 | 递归字符分割 + 10% 重叠 |
| 3 | 向量嵌入 | 文本 → 768 维语义向量 | BGE-small-zh-v1.5（ONNX） |
| 4 | 向量检索 | 问题 → Top-K 相关块 | FAISS + BM25 + RRF 混合检索 |
| 5 | 精排重排 | 粗排结果 → 精准排序 | Cross-Encoder 逐对打分 |
| 6 | LLM 生成 | 上下文 + 问题 → 回答 | 智谱 GLM-4 / DeepSeek |
| 7 | Agent 循环 | 多轮对话 + 工具调用 + 记忆 | 手写 while 循环 |
| 8 | LangGraph 重写 | 等价重写，对比学习 | StateGraph + Checkpointer |

## 技术栈

| 类别 | 选型 | 说明 |
|------|------|------|
| 嵌入模型 | BGE-small-zh-v1.5 | 中文 768 维，ONNX 推理，离线免费 |
| 向量库 | FAISS (IndexFlatIP) | 精确暴力检索，313 块毫秒级 |
| 精排模型 | bge-reranker-base | 交叉编码器，从 20 候选选 5 |
| LLM | 智谱 GLM-4-Flash / DeepSeek | 免费 API，支持工具调用 |
| 文档解析 | pdfplumber + RapidOCR | 文本层直读 + OCR 降级 |
| 分块 | RecursiveCharacterTextSplitter | langchain-text-splitters（零重依赖） |
| Agent 框架 | 手写 + LangGraph 1.0.7 | 两种实现并存对比 |

## 快速开始

### 环境要求

- Python 3.11+（推荐 Anaconda）
- Windows / Linux / macOS

### 安装

```bash
git clone https://github.com/你的用户名/ai-agent.git
cd ai-agent/learning/agent_project
pip install -r ../../requirements.txt
```

### 入库 → 问答

```bash
# 1. 将 PDF 放入 docs/sources/ 目录

# 2. 入库（解析 → 分块 → 嵌入 → 建索引）
python -m agent_project.ingest

# 3. 启动问答终端
python -m agent_project.chat
```

### 运行测试

```bash
cd learning/agent_project
python -m pytest tests/ -v
```

## 项目结构

```
ai_agent/
├── learning/agent_project/
│   ├── src/agent_project/
│   │   ├── main.py                 # 演示入口
│   │   ├── ingest.py               # 入库命令
│   │   ├── chat.py                 # 交互式问答终端
│   │   ├── preprocessor/           # 文档解析
│   │   ├── chunker/                # 文本分块
│   │   ├── embedder/               # 向量嵌入
│   │   ├── retriever/              # 混合检索（向量 + BM25 + RRF）
│   │   ├── reranker/               # 精排重排
│   │   ├── generator/              # LLM 生成
│   │   ├── agent/                  # Agent 循环（手写版）
│   │   └── agent_langgraph/        # Agent 循环（LangGraph 版）
│   ├── docs/specs/                 # 设计文档
│   ├── tests/                      # 测试套件
│   └── data/                       # 运行时产物（已 gitignore）
├── requirements.txt
└── README.md
```

## 设计原则

1. **裸写优先**：每个模块手写实现，理解底层机制后再用框架
2. **测试驱动**：每个阶段有独立验收脚本，可重复验证
3. **渐进增强**：从简单定长分块到混合检索，逐步升级
4. **等价对比**：裸写版 vs LangGraph 版并存，用测试证明编排层之外完全一致

## License

MIT
