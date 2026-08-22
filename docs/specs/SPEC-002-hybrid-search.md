# SPEC-002 混合检索(向量 + BM25 + RRF 融合)

- **编号**: SPEC-002
- **类型**: 功能 · 影响面: 完整
- **状态**: ✔ 已验收(2026-08-18)
- **创建**: 2026-08-18
- **关联**: SPEC-001(两段式检索契约,本 spec 扩展其粗排环节)、架构详解 04/05

## 背景

纯向量检索(现状)对**精确关键词/专有名词**类查询弱:向量只认"语义相近",
搜协议名、函数名、配置项等字面术语时,含该术语的块可能因"整体语义"不贴近而排不到
top-k。业界(Dify/RAGFlow/FastGPT)通解:向量 + BM25 全文检索两路并行,
RRF(Reciprocal Rank Fusion,倒数排名融合)合并——语义泛化能力和字面精确匹配能力互补。

```
现状:  query → 向量粗排 top-20 → 精排 top-3
目标:  query → ┬ 向量检索 top-N ─┐
               │                 ├→ RRF 融合 top-20 → 精排 top-3
               └ BM25 全文 top-N ─┘
```

## 目标 / 非目标(MoSCoW)

**Must(必须)**:
- 新增 `hybrid_search(query, index_path, meta_path, ...)`:向量 + BM25 两路,RRF 融合,
  返回格式与 `search()` 兼容(下游 rerank/prompt 不改语义)
- 中文分词用 jieba(查询与文档同分词器,保证可比性)
- BM25 索引按 meta_path 缓存复用(同一库重复检索不重建)
- `rag_answer` 的粗排入口切换为 `hybrid_search`

**Should(应该)**:
- 每个 hit 带 `vector_score` / `bm25_score` 来源分(可观测:命中来自哪一路)
- RRF 融合逻辑写成可独立运行的最小测试(不依赖 pytest 也能跑),tests/ 目录起头

**Won't(不做)**:
- 不引入 Elasticsearch/全文检索引擎(数百块量级,进程内 rank_bm25 足够)
- 不持久化 BM25 索引(查询时从 .json 构建,数百块毫秒级;万级语料再升级)
- 不做融合权重调优(RRF 固定 k=60 起步;调参留待有评测集后)
- 不改现有 `search()`(向量单路保留,作为对照与回退)

## 验收标准(Given-When-Then)

- [x] AC1 — 两路融合,格式兼容
  - Given 已有 Wireshark 向量库产物(.index + .json)
  - When `hybrid_search(问题, index_path, meta_path, k=20)`
  - Then 返回 20 条,按融合分降序;每条含 `chunk`/`meta`/`score`(=融合分,
        供下游兼容)/`rrf_score`/`vector_score`(未入向量 top 者为 None)/`bm25_score`(同理)

- [x] AC2 — RRF 数学正确(最小用例)
  - Given 构造两路排名:路A = [x, y, z],路B = [z, x](k_rrf=60)
  - When 融合
  - Then x = 1/61(路A第1) + 1/62(路B第2) ≈ 0.03252;
        z = 1/61(路B第1) + 1/63(路A第3) ≈ 0.03227;
        y = 1/62(仅路A第2) ≈ 0.01613;
        排序 **x > z > y**(两路都靠前的 x 第一;单路的 y 殿后)
  - 修订: 2026-08-18 原稿手算值有误(x 误算为 1/61+1/61,y 误算为 1/61;
    实现时写测试核对发现,spec 修正 —— 见 tests/test_rrf.py)

- [x] AC3 — 精确关键词命中(核心价值)
  - Given 从 Wireshark 语料选一个专有名词型 query(实现时实测选定,记入实现备注)
  - When 对比 `search()`(纯向量)与 `hybrid_search()` 的 top-20
  - Then 混合检索的字面命中能力 ≥ 纯向量:至少一个"字面包含关键词但向量分不高"
        的块,混合后进入 top-20 而纯向量下不在(或显著靠后)

- [x] AC4 — 语义能力不回退
  - Given 语义型查询(如"怎么把加密流量还原成明文",无字面重合)
  - When `hybrid_search()` 对比 `search()`
  - Then 原向量 top 命中仍以高融合分保留在 top-20(引入 BM25 不丢语义相关块)

- [x] AC5 — 完整链路
  - When `rag_answer(问题, ..., k=3, recall_k=20)`
  - Then 粗排走 `hybrid_search`,精排/生成不变;返回 `{"answer", "hits"}` 结构不变,
        hits 含 `rerank_score`

- [x] AC6 — BM25 索引复用
  - Given 同一 meta_path 连续两次 `hybrid_search`
  - Then 第二次不重建 BM25 索引(缓存命中)

## 涉及模块

- `retriever/hybrid.py`(新):两路检索 + RRF 融合
- `retriever/__init__.py`:导出 `hybrid_search`
- `generator/answer.py`:粗排入口 `search` → `hybrid_search`
- 依赖新增(conda learning 环境):`rank_bm25`(纯算法,KB 级)、`jieba`(中文分词,MB 级)

## 风险与兜底

- **专有名词切分碎**(jieba 未登录词):查询与文档同用 jieba,一致性保证可比;
  必要时对 token 加 lowercase 归一
- **查询时构建 BM25 的延迟**:313 块毫秒级;jieba 首次加载词典 ~1s(进程内只一次);
  按 meta_path 缓存索引。万级语料时升级为持久化(见 Won't)
- **格式破坏下游**:hits 必须保留 `score` 字段(rerank 与 prompt 依赖),
  融合分写进 `score`,来源分另立字段

## 实现备注(实现后回填)

- **AC3 实测数据(2026-08-18)**:query=`pcapng`,纯向量 top-20 完全漏掉 10 个
  字面含 "pcapng" 的块,混合检索全部捞回(10/10 字面命中)——核心价值的直接实证。
  AC5 的 LLM 答案引用的 pcapng 格式设定资料正是来自 BM25 路捞回的块。
- **AC4 实测**:语义型 query(无字面重合)下,原向量 top-5 语义命中 5/5 保留在
  混合 top-20,其中 2 块被 RRF 抬升到第 1、2 名(两路共识加分,符合预期)
- **AC6 实测**:首查 1.68s(jieba 词典+全库分词+建 BM25),缓存后 0.02s
- **tests/ 起头**:`tests/test_rrf.py`(AC2,纯函数,直接 python 运行,不依赖 pytest)
- **依赖安装**:conda(learning,清华镜像+libmamba)装 jieba-0.42.1 + rank-bm25-0.2.2,
  仅新增未动现有包(rev 18)
- **环境观察**:`langchain_text_splitters` 首次 import ~25s(拖 langchain-core 全家,
  本机一直如此),测试脚本慢是 import 不是测试本身
- **AC2 spec 修正**:草稿手算值有误(x 误为 1/61+1/61),写测试核对时发现并修正——
  "测试暴露 spec 错误"的首个案例
