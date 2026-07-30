from pathlib import Path

def parse_md(file_path: str) -> str:
    """直接读取Markdown文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()