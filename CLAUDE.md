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
        ├── src/
        │   ├── main.py        # 文档解析入口（按扩展名分派到对应解析器）
        │   └── preprocessor/
        │       └── document_parser/   # 文档解析模块（pdf / md 等）
        ├── data/
        └── logs/
```

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
