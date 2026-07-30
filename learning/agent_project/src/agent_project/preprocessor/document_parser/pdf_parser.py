# -*- coding: utf-8 -*-
"""
PDF 自适应解析器
- 有文本层 → pdfplumber 直接提取（快）
- 无文本层 → RapidOCR 识别（PaddleOCR模型 + ONNX推理，无需 paddlepaddle）
"""
import os
import pdfplumber

# ---------- RapidOCR 单例 ----------
_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_instance = RapidOCR()
    return _ocr_instance


# ---------- 核心判断：页面是否有有效文本层 ----------
MIN_TEXT_LENGTH = 10


def _has_text_content(page) -> bool:
    """判断页面是否含有可直接提取的有效文本"""
    if len(page.chars) == 0:
        return False
    text = page.extract_text()
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    return True


# ---------- OCR 识别单页 ----------
def _ocr_page(page) -> str:
    """将页面渲染为图片，用 RapidOCR 识别文字"""
    page_img = page.to_image(resolution=150)  # 150 dpi，兼顾速度与质量
    pil_img = page_img.original

    ocr = _get_ocr()
    result, _ = ocr(pil_img) #返回结果列表，时间

    texts = []
    if result:
        for item in result:
            # RapidOCR 返回: [[box, text, score], ...]
            texts.append(item[1])
    return "\n".join(texts)


# ---------- 主入口 ----------
def parse_pdf(file_path: str) -> str:
    """解析 PDF：文本层可用则直接提取，否则自动 OCR"""
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        print(f"总页数: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            if _has_text_content(page):
                text = page.extract_text()
                print(f"  第{i+1}页: 文本提取, 长度={len(text)}")
                full_text.append(text)
            else:
                print(f"  第{i+1}页: 无文本层, OCR识别中...")
                text = _ocr_page(page)
                print(f"  第{i+1}页: OCR完成, 长度={len(text)}")
                full_text.append(text)
    return "\n".join(full_text)
