# AI Agent 学习项目

## 运行环境（务必遵守）

- **Python 环境**: Anaconda 的 `learning` 环境，通过 `conda activate learning` 激活
- **包管理**: 使用 `conda` 或在 learning 环境内使用 `pip` 安装依赖
- **⚠️ 严禁混用其它环境**: 本项目与 `S:\uv-envs\scrape` **无关**。`scrape` 是另一个项目（`s:\Code\baigong`，山东省大学生就业服务平台爬虫）的 uv 环境。给本项目安装任何依赖时，**切勿**装到 `scrape`，也不要对本项目使用 `uv` 命令。

## 项目结构

```
ai_agent/
└── learning/                  # learning 目录下的项目共用 conda learning 环境
    └── agent_project/         # 当前主要项目
        ├── src/agent_project/
        │   ├── main.py        # 演示入口（各阶段验收）
        │   ├── preprocessor/  # 【阶段1】文档解析（pdf/md 自适应）
        │   ├── chunker/       # 【阶段1】文本分块
        │   ├── embedder/      # 【阶段2】向量嵌入（BGE）
        │   ├── retriever/     # 【阶段3】粗排检索（FAISS）
        │   ├── reranker/      # 【阶段5】精排重排（cross-encoder）
        │   ├── generator/     # 【阶段4】LLM 生成（智谱 API）
        │   └── agent/         # 【阶段6】Agent 循环（LLM 自主决策 + 多轮记忆）
        ├── docs/specs/        # SDD 契约库（见「开发流程」）
        ├── tests/             # 可重复运行的验收测试（AC 的长期形态）
        ├── data/              # data/output/ 存向量库产物（不入库）
        └── logs/
```

项目学习文档:`learning/agent_project/docs/架构详解/00-总览与导读.md`（各模块机制/代码走读/踩坑/业界对照,按模块分篇）;项目入口与运行见 `learning/agent_project/README.md`。

## 开发流程（SDD，务必遵守）

规范全文见 `learning/agent_project/docs/specs/README.md`。**闭环七步，每步都有明确产物：**

```
0. grep 防重     → grep docs/specs/ 查相关契约
1. 写 spec       → docs/specs/SPEC-NNN-短名.md（📝草稿）
2. 人确认方案     → spec 改 ✅已确认（门禁通过，可动代码）
3. 实现 + 写测试  → src/ 下写代码 + tests/ 下写验收脚本
4. 逐条验收       → 验收脚本跑通 + AC 核对
5. 同步收尾       → 以下六件事同一次提交完成：
   a. spec 状态改 ✔已验收，逐条勾 AC，回填实现备注
   b. docs/specs/README.md 回溯清单更新（如适用）
   c. 新模块 → docs/架构详解/ 新增对应编号文档（01-07 按需）
   d. docs/架构详解/00-总览与导读.md 模块表 + 阅读路线同步
   e. README.md 文档索引 + 路线图同步
   f. CLAUDE.md 项目结构树同步（如新增/移动目录）
6. git commit    → Conventional Commits 格式提交
```

**硬约束（铁律）：**
- spec 状态 ≠ ✅已确认 时，**严禁**改 `src/agent_project/` 核心代码
- 步骤5 的六件事（a-f）**缺一不可**，每次都有遗漏的教训
- 验收脚本跑不通 / AC 未核对 / 有跳过的步骤，**不得宣称完成，必须如实报告**

## 文档解析模块（自适应架构）

`src/preprocessor/document_parser/pdf_parser.py` 实现自适应 PDF 解析：

```
输入 PDF → pdfplumber 提取
              ├── 有文本层 → 直接返回文本（快）
              └── 无文本层 → RapidOCR 识别（准）
```

### OCR 引擎：RapidOCR

使用 `rapidocr-onnxruntime`（PaddleOCR 训练的模型 + ONNX Runtime 推理），原因：

- **无需 paddlepaddle**：PaddlePaddle 3.x 在 Windows 上有 oneDNN 兼容性 bug（`ConvertPirAttribute2RuntimeAttribute` 错误），2.x 依赖链过长
- **依赖轻量**：只需 `onnxruntime` + `opencv-python`，无需 paddlepaddle 整个生态
- **识别效果相同**：使用 PaddleOCR 训练的 PP-OCRv3 模型，中文识别准确率与 PaddleOCR 一致

### ⚠️ 环境约束

- **不要安装 paddlepaddle**：3.x 有 Windows 兼容性问题，2.x 依赖链过长（imgaug/scikit-image/scipy 等），都装不全
- **不要安装 paddleocr**：同上原因
- **已安装的核心包**：`pdfplumber`、`rapidocr-onnxruntime`、`onnxruntime`、`opencv-python`、`numpy`

### PDF 结构说明

`pdfplumber` 只能提取 PDF 的**文本层**（字符对象）。对于由 HTML 转换而来、内容被光栅化为图片（页面无文本对象，只有嵌入图片 + 矢量路径）的 PDF，`extract_text()` 会返回空。此时自动降级到 RapidOCR 识别。

判断逻辑：`page.chars == 0` 或提取文本长度 < 10 字符 → 视为无文本层，触发 OCR。
