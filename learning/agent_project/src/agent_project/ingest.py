# -*- coding: utf-8 -*-
"""
入库命令行(SPEC-005)。

    python -m agent_project.ingest <文件...>   # 入库(pdf/md),同名覆盖
    python -m agent_project.ingest --list      # 列出已入库文档

把文档解析/分块/嵌入后落盘成 (.index, .json) 产物对,
供 search / hybrid_search_all / Agent 的 search 工具检索。
"""
import json
import sys
from pathlib import Path

from agent_project.retriever import build_index


def _list_docs():
    from agent_project.retriever.hybrid import discover_docs
    pairs = discover_docs()
    if not pairs:
        print("知识库为空。入库用法: python -m agent_project.ingest <文件...>")
        return
    print(f"知识库共 {len(pairs)} 份文档:")
    for i, (_, mpath) in enumerate(pairs, 1):
        with open(mpath, encoding="utf-8") as f:
            data = json.load(f)
        src = data["metas"][0]["source"] if data["metas"] else Path(mpath).stem
        print(f"  【{i}】{src}({len(data['chunks'])} 块)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv == ["--list"]:
        _list_docs()
        return

    for path in argv:
        if path == "--list":
            _list_docs()
            continue
        p = Path(path)
        if not p.exists():
            print(f"✗ 文件不存在: {p}")
            continue
        try:
            index_path, meta_path, n_chunks = build_index(str(p))
        except Exception as e:
            print(f"✗ {p.name} 入库失败: {e}")
            continue
        print(f"✓ {p.name} → {n_chunks} 块")
        print(f"    {index_path}")
        print(f"    {meta_path}")


if __name__ == "__main__":
    main()
