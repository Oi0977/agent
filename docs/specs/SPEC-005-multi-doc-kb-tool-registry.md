# SPEC-005 多文档知识库 + 工具注册表

- **编号**: SPEC-005
- **类型**: 功能 · 影响面: 完整(检索层 + agent 工具层 + 新入口)
- **状态**: ✔ 已验收(2026-08-19)
- **创建**: 2026-08-19
- **关联**: SPEC-001(两段式检索)、SPEC-002(RRF 融合,直接复用 `_rrf_fuse`)、SPEC-003(工具来源)

## 背景

三个捆在一起的债:
1. **知识库只能装一本书**:`_find_index()` 拿 output 目录里第一个 `.index`,第二个文档
   入库即静默丢失;没有入库命令,建库靠 main.py 全量重跑
2. **工具定义是手写 JSON 拼在列表里**:加一个工具要改 TOOLS + `_exec_tool` 两处,
   没有注册机制(业界标准是注册式:函数+schema 一处声明)
3. **main.py 每次运行重建索引**:秒级变分钟级,演示体验差

## 目标 / 非目标(MoSCoW)

**Must**:
- `python -m agent_project.ingest <文件...>` 入库命令:每文档建索引,打印摘要;
  `--list` 列出已入库文档;重复入库覆盖旧产物
- 多文档混合检索 `hybrid_search_all()`:扫描 output 目录全部 (index, json) 对,
  每文档向量一路 + BM25 一路 → **2×D 路排名 RRF 融合**(候选 key=(doc_id, chunk_idx),
  直接复用 `_rrf_fuse`);命中带 `meta.source` 区分来源文档。
  **已知边界(真机踩坑)**:跨文档 RRF 存在"小文档名次压缩"——3 块的小库里任何块
  都天然是双路 top-3,弱匹配也拿满融合分,可能压过正确结果 ⇒ 由下游精排纠序
- **Agent search 工具 = 两段式**(与 rag_answer 对齐):hybrid_search_all 召回 top-10 →
  `rerank` 交叉编码器精排 top-5(精排失败退回 RRF 序,不炸)
- **工具注册表** `agent/tools.py`:`@tool(name, description, parameters)` 装饰器注册,
  `get_tool_schemas()` 自动生成 TOOLS,`execute_tool()` 统一分发
- 新工具 2 个:**calculator**(ast 白名单安全求值)、**list_documents**(列知识库文档)
- `build_index()` 支持自定义 `output_dir`(测试隔离)
- main.py:已有索引则复用,不重建
- 债务清扫:`rag_answer()` 修复 chat() 重构遗留的旧调用方式
  (`chat(prompt)` → messages 列表 + 取 `.content`,恢复返回纯文本答案)

**Should**:
- verbose 打印本次检索覆盖了几个文档
- agent 系统提示随注册表工具数自动提及计算类问题用 calculator

**Won't**:
- 增量更新/删除文档(重入库=全量覆盖)
- 向量数据库(继续 FAISS 文件级)
- 工具并行调用

## 验收标准(Given-When-Then)

- [x] AC1 — 注册表 schema 生成(离线)
  - Given 用 @tool 注册一个带 name/description/parameters 的函数
  - When get_tool_schemas()
  - Then 生成 OpenAI function calling 格式 schema;未注册名 execute_tool 返回错误串而非抛异常

- [x] AC2 — calculator 安全求值(离线)
  - Given 表达式 "2+3*4"、"(1+2)**3"
  - When 执行
  - Then 返回 14、27;对 "import os"、`__import__('os')`、"abc" 返回错误提示串,不执行、不抛异常

- [x] AC3 — 多文档检索(真机/脚本)
  - Given 已入库 Wireshark PDF 和另一份小 md 文档
  - When 对两文档各自领域的问题执行 agent search 工具(召回 + 精排)
  - Then 精排后 top1 命中来自对应领域文档,meta.source 正确区分来源
    (纯 RRF 序允许小文档名次压缩导致的错序 —— 精排负责纠正,见 Must 已知边界)

- [x] AC4 — 无库时明确报错(离线)
  - Given output_dir 为空目录
  - When hybrid_search_all(query)
  - Then 抛 FileNotFoundError 并提示先 ingest

- [x] AC5 — 入库命令(真机)
  - When `python -m agent_project.ingest <md文件>`
  - Then 打印"文件 → 块数 → 产物路径";`--list` 能列出它

- [x] AC6 — Agent 回归(真机)
  - Given SPEC-004 验收同款两轮对话
  - Then 行为不回归:HTTPS 问题照常回答,追问照常指代上文;
    问"123 乘 456 等于多少"时走 calculator 不调 search

## 涉及模块

- `retriever/hybrid.py`:新增 `discover_docs()` / `hybrid_search_all()`
- `retriever/builder.py`:`build_index(..., output_dir=None)`
- `agent/tools.py`(新):注册表 + 4 个工具(search/direct_answer/calculator/list_documents)
- `agent/agent.py`:TOOLS 与分发改走注册表;search 工具切到 hybrid_search_all
- `ingest.py`(新,包根):入库命令行入口
- `main.py`:索引复用逻辑
- `tests/test_tools.py`(新):AC1/AC2 离线;AC4 用临时目录

## 风险与兜底

- **多文档 BM25 全库分词内存**:每文档一份库级缓存(已有 `_lib_cache`),文档多时内存线性涨——学习项目可接受,记录为已知边界
- **跨文档 RRF 公平性**:每文档每路取前 n_per_route,大文档小文档机会均等(按排名不按分数,天然归一)
- **ingest 重复入库**:同名覆盖(ASCII 文件名含文档哈希,天然防撞)

## 实现备注(实现后回填)

- **AC1/AC2/AC4 离线**:`tests/test_tools.py` 全过(含 discover_docs 只认成对产物)
- **AC3 实测**:问"Nmap SYN 半开扫描"→ top1=kb_notes.md;问"Wireshark 解密 HTTPS"
  → top1=Wireshark PDF(精排后;纯 RRF 序下后者曾错被小文档抢走,见下)
- **★ 真机踩坑:跨文档 RRF 小文档名次压缩**(本 spec 最有价值的发现):
  纯 RRF 下"Wireshark 解密 HTTPS"的 top1 被 3 块的 kb_notes 抢走 ——
  小库里任何块都天然是双路 top-3,弱匹配拿满融合分,RRF"排名分布可比"的
  假设被小库破坏。修法:agent search 工具改两段式(召回 top-10 → rerank 精排
  top-5,失败退回 RRF 序)。实现中途修订了本 spec 的 Must/AC3(SDD 规则 6:
  发现方案问题先改 spec),机制详解见 04 详解 §9
- **AC5 实测**:kb_notes.md 入库 3 块;--list 显示 2 份文档(465 块 + 3 块)
- **AC6 实测**:"123 乘 456" → calculator → 56088;SPEC-004 两轮追问回归通过。
  观察:回归轮 LLM 连调了 2 次 search(违反"最多一次"提示)但第 4 轮自然收敛,
  未触发兜底 —— 提示约束是软的,兜底仍是必需品
- **债务清扫**:`rag_answer()` 的 `chat(prompt)` 是 chat() 重构(agent 化)前的
  旧调用,已修为 messages 列表 + 取 .content,恢复返回纯文本;真机验证通过。
  rag_answer 保持单文档接口(hybrid_search),多文档走 agent 工具 —— 两条链路继续对照
- **ingest 幂等**:同名覆盖(ASCII 名含文档哈希),重复入库安全

## 修订记录

- 2026-08-19 修订(当日二次,评测驱动):`hybrid_search_all()` 的 BM25 路改为
  **全局语料**(合并所有文档块统一建 BM25,统一 IDF),不再是每库一路。
  诊断:rank_bm25 的 IDF 是库内的 —— 3 块小库 0.09 分的噪音匹配也拿满
  "第1名"排名,与 465 块大库 13.2 分的第1名在 RRF 里同权,弱匹配被双路
  双计后反压真结果(实测 q01 pcapng 被挤出 top-3)。
  向量分数同一嵌入模型天然跨库可比,维持各库分路;BM25 分数不跨库可比,
  必须全局语料。同次修订:向量路加 0.35 相似度地板(弱语义匹配不参与排名,卫生项)。
  单文档 `hybrid_search`(SPEC-002 契约)不动。触发证据见 SPEC-008 实现备注。
