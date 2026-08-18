# SPEC-001 检索两段式契约(粗排召回 + 精排重排)

- **编号**: SPEC-001
- **类型**: 回溯契约(功能已存在,反向锚定) · 影响面: 完整
- **状态**: ✔ 已验收
- **创建**: 2026-08-18
- **关联**: 架构详解 03(嵌入)、04(检索)、05(重排序)

## 背景

RAG 检索链路已实现两段式(业界标配):双塔向量粗排(快而广)召回 top-recall_k,
交叉编码器精排(慢而准)精选 top-k,再交生成。本文档锚定这套行为的契约,
防止未来会话二次实现或无意破坏关键约定(尤其 BGE 前缀)。

## 行为契约

```
问题 → embed_query(加前缀) → FAISS 粗排 top-recall_k
     → rerank 精排 top_k → build_prompt → chat → 答案
```

## 验收标准(Given-When-Then)

- [x] AC1 — 粗排检索
  - Given 已有某 PDF 的 `.index` + `.json` 产物(如 Wireshark)
  - When `search(问题, index_path, meta_path, k=20)`
  - Then 返回 20 条,按余弦相似度**降序**,每条含 `chunk`/`meta`/`score`;
        id 越界(如 k > 库大小时 FAISS 的 -1 占位)被过滤

- [x] AC2 — 精排重排
  - Given `search` 返回的 top-20 候选
  - When `rerank(问题, hits, top_k=3)`
  - Then 返回 3 条,按 `rerank_score`(sigmoid 后 0~1)**降序**;
        每条保留原 `score`(双塔余弦分)供对照

- [x] AC3 — 精排可改变排序(效果性验证)
  - Given 问题"Wireshark 里怎么解密 HTTPS 流量?"
  - When 对比粗排 top-3 与精排 top-3
  - Then 允许不同;实测精排把块 92(Preferences 配置,向量分 0.626/粗排第 8+)
        提入 top-3,因其与解密路径(Edit→Preferences→SSL)语义相关

- [x] AC4 — 嵌入前缀约定(BGE 非对称约定,不可破坏)
  - When 建库 `embed_texts()` 与查询 `embed_query()` 分别编码
  - Then passage **不加前缀**;query **自动加**
        `"为这个句子生成表示以用于检索相关文章:"` 前缀;
        两边均 L2 归一化(内积=余弦)

- [x] AC5 — 完整链路
  - When `rag_answer(question, index_path, meta_path, k=3, recall_k=20)`
  - Then 返回 `{"answer": str, "hits": list}`;hits 为精排后 top-3

- [x] AC6 — 模型离线加载
  - Given 本地缓存存在(`S:\huggingface_cache`,见 README「模型资产」)
  - When 加载嵌入/重排模型
  - Then 直接解析本地快照目录加载,**不发起任何网络请求**
        (huggingface.co 网络不通时也不受影响)

## 涉及模块

- `retriever/searcher.py`(粗排) / `reranker/reranker.py`(精排)
- `embedder/encoder.py`(前缀约定) / `generator/answer.py`(链路编排)

## 风险与兜底

- FAISS 返回 id=-1 占位(k 超库大)→ search 内过滤
- Windows 中文文件名导致 `faiss.write_index` 报错 → 产物名 ASCII+哈希(建库契约)
- 免费模型 429 限流 → llm_client 已设 `max_retries=5`

## 实现备注

- 验收记录:temp/verify_rerank.py(2026-08-18 实跑,AC3 见其实测输出)
- 粗排/精排分数语义:`score`=双塔余弦(粗排依据),`rerank_score`=交叉编码器 sigmoid(精排依据)
