# -*- coding: utf-8 -*-
"""
SPEC-005 AC1/AC2/AC4 —— 工具注册表与 calculator 的离线验证。

纯函数测试,不依赖 pytest,直接运行:
    python tests/test_tools.py

不调 LLM、不加载模型(tools.py 顶层只有标准库,检索 import 全部延迟)。
多文档检索(AC3)/入库命令(AC5)/Agent 回归(AC6)由真机脚本覆盖。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_project.agent.tools import execute_tool, get_tool_schemas


def test_ac1_registry_schema():
    """AC1:@tool 注册 → get_tool_schemas 生成 OpenAI 格式;未知工具返回错误串。"""
    schemas = get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"search", "direct_answer", "calculator", "list_documents"} <= names, names
    calc = next(s for s in schemas if s["function"]["name"] == "calculator")
    assert calc["type"] == "function"
    assert calc["function"]["parameters"]["required"] == ["expression"]
    # 未注册名:错误串回注,不抛异常(Agent 循环不因打错工具名而崩)
    r = execute_tool("不存在", {})
    assert isinstance(r, str) and "未知工具" in r
    # 参数不匹配:同样转错误串
    r = execute_tool("calculator", {})
    assert isinstance(r, str) and ("参数不匹配" in r or "无法计算" in r)
    print("AC1 ✓ 注册表 schema 生成正确;未知工具/参数错误均安全返回字符串")


def test_ac2_calculator():
    """AC2:正常算术求值正确;危险/非法输入全部安全拒绝。"""
    assert "14" in execute_tool("calculator", {"expression": "2+3*4"})
    assert "27" in execute_tool("calculator", {"expression": "(1+2)**3"})
    assert "56088" in execute_tool("calculator", {"expression": "123*456"})
    assert "-7" in execute_tool("calculator", {"expression": "-3-4"})
    for bad in ["import os", "__import__('os')", "abc", "", "1+'a'"]:
        r = execute_tool("calculator", {"expression": bad})
        assert isinstance(r, str), bad
        assert ("无法计算" in r or "参数不匹配" in r), (bad, r)
        assert " = " not in r, f"{bad} 不应产出成功格式: {r}"
    print("AC2 ✓ calculator 白名单求值正确;import/调用/名字类输入全部拒绝")


def test_discover_docs():
    """discover_docs:只认成对的 (.index, .json);按名排序。"""
    from agent_project.retriever.hybrid import discover_docs
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.index").write_text("x")
        (Path(td) / "a.json").write_text("{}")
        (Path(td) / "b.index").write_text("x")     # 缺 .json 伴 → 残骸,不参与
        pairs = discover_docs(output_dir=td)
        assert len(pairs) == 1 and Path(pairs[0][0]).name == "a.index", pairs
    print("✓ discover_docs 只认成对产物")


def test_ac4_empty_kb():
    """AC4:空知识库 → FileNotFoundError 且提示 ingest。"""
    from agent_project.retriever.hybrid import hybrid_search_all
    with tempfile.TemporaryDirectory() as td:
        try:
            hybrid_search_all("任意问题", output_dir=td)
            raise AssertionError("应抛 FileNotFoundError")
        except FileNotFoundError as e:
            assert "ingest" in str(e), e
    print("AC4 ✓ 空知识库明确报错并提示先 ingest")


test_ac1_registry_schema()
test_ac2_calculator()
test_discover_docs()
test_ac4_empty_kb()
print("\n全部通过:SPEC-005 注册表/calculator/发现逻辑 ✓")
