# SPEC-007 交互式会话终端 + 会话持久化

- **编号**: SPEC-007
- **类型**: 功能 · 影响面: 中(新入口 + 持久化;不改核心循环)
- **状态**: ✔ 已验收(2026-08-19)
- **创建**: 2026-08-19
- **关联**: SPEC-004(多轮记忆)、SPEC-006(会话统计展示)、SPEC-005(入库/检索)

## 背景

Agent 目前只能靠演示脚本驱动,没有可用入口——"麻雀"缺身体。多轮记忆和 token
统计都已就绪,套一层 REPL 即成为可用的最小产品;会话落盘补上 SPEC-004 明确
Won't 的"磁盘持久化"(该 Won't 指记忆机制本身,本 spec 只做会话文件存取)。

## 目标 / 非目标(MoSCoW)

**Must**:
- `python -m agent_project.chat` 终端 REPL:持续多轮对话(历史跨轮传递),
  每轮显示 token 统计行(来自 SPEC-006 stats)
- 命令:`/exit` 退出;`/new` 清空历史开新会话;`/save [名字]` 把当前历史+统计
  存为 JSON;`/load [名字]` 载入;`/list` 列已存会话;`/help` 帮助
- 会话文件落 `data/sessions/<名字>.json`:`{"saved_at", "history", "stats"}`
- **REPL 可注入读写**(chat_loop(read=input, write=print))→ 离线脚本化测试,
  不依赖真终端

**Should**:
- 启动时打印知识库内文档数(提示 ingest)
- 空输入友好忽略;EOF(Ctrl+Z/Ctrl+D)等同 /exit

**Won't**:
- Streamlit Web UI(**单独立项**:装包有连带升级学习环境的风险——本项目曾因装包
  把 transformers 全家桶连带升级导致模型加载崩溃;CLI 已满足 MVP 演示)
- 流式输出;多会话并开;会话内编辑历史

## 验收标准(Given-When-Then)

- [x] AC1 — 会话序列化纯函数(离线)
  - Given 一段 history + stats
  - When save_session()/load_session()(路径可注入)
  - Then JSON 落盘并可无损读回(history 深相等,stats 数值相等)

- [x] AC2 — REPL 离线驱动(离线)
  - Given 注入脚本化输入:["你好","/exit"] 且 run 被 mock 为固定应答
  - When chat_loop()
  - Then 依序产生两轮交互后正常退出,无异常

- [x] AC3 — 命令语义(离线)
  - Given 注入输入序列含 /new、/save t1、/new、/load t1、/exit
  - When chat_loop()
  - Then /new 后历史清空;/save 落盘;/load 后历史恢复(mock run 收到的 history
    与保存前一致);/list 输出含 t1

- [x] AC4 — token 行可见(真机)
  - Given 真实 API
  - When 终端跑一轮问答
  - Then 每轮后显示 [token] prompt/completion/累计 统计行

- [x] AC5 — 真机端到端(真机)
  - When 管道喂入"问一句 → /save demo → /exit",重启进程后 /list、/load demo
  - Then 会话恢复,/load 后追问能接上上文(记忆跨进程存活)

## 涉及模块

- `chat.py`(新,包根):REPL + 命令解析 + save/load 纯函数
- `path_manager.py`:如需 SESSIONS_DIR(data/sessions)
- `agent/__init__.py`:如需导出
- `tests/test_chat.py`(新):AC1/AC2/AC3(mock run,注入 read/write)

## 风险与兜底

- **REPL 测试脆**:读写全部可注入,测试不碰真终端
- **429 限流**:交互场景用户自然有间隔;chat() 已有指数退避
- **损坏会话文件**:load 时 JSON 解析失败 → 报错并保持当前会话不动,不崩溃

## 实现备注(实现后回填)

- **AC1/AC2/AC3 离线**:`tests/test_chat.py` 全过(read/write/run 全注入,
  mock run 精确捕获收到的 history,/load 恢复的就是保存前的列表)
- **AC4/AC5 实测**(2026-08-19,管道驱动两个独立进程):
  - 进程1:问答 → /save demo → /exit;进程2:/list → /load demo → 追问 → /exit
  - 载入后追问沿上下文展开(搜"私钥""Preferences"),token 行全程可见:
    本轮 prompt 18796 | 会话累计 20840/2030 —— 历史重发+多轮检索的成本肉眼可见
- **观察:带历史时"最多一次 search"约束更弱** —— 本轮连搜 5 次触发
  max_iterations,**兜底按设计介入**:基于已检索资料作答且如实声明
  "资料未含私钥设置路径"(不编造)。提示是软约束的又一实证;兜底是必需品
- **设计要点**:chat_loop(read, write, run) 三注入点让 REPL 完全离线可测 ——
  这是"可测试性靠依赖注入而不是靠终端模拟"的最小范例;知识库探测只在
  真实模式(run 未注入)执行,测试不拖重导入
