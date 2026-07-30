import re


def clean_text(text: str) -> str:
    """
    清洗文本：去除空行、乱码、特殊符号
    """
    #统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 1. 去除零宽字符等不可见字符
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\uFEFF]', '', text)

    # 2. 将多个连续换行合并为单个换行（去除多余空行）
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # 3. 去除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 4. 去除多余连续空格（保留单个空格）
    r"""
    [^\S\n] 这个技巧：\S 是非空白，[^\S] 取反就是"所有空白"，
    再排除 \n 就是"空白但保留换行"。它一次覆盖 NBSP、全角空格、U+2000–U+200A 的各种排版空格、制表符等,
    不用再逐个列
    """
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 5. 可选：去除特殊控制字符（保留常见标点）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 6. 再次去除因清洗产生的空行
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()