# -*- coding: utf-8 -*-
"""
工具注册表 —— 注册式工具管理(SPEC-005)。

加一个工具 = 写一个函数 + 一处 @tool 声明(名字/描述/参数 schema);
LLM 看的 TOOLS 列表和代码侧的执行分发都从注册表自动生成,不再手拼两处。

对照业界:LangChain 的 @tool 装饰器、OpenAI 的 function schema,本质相同 ——
把"LLM 看的 schema"和"代码执行的函数"绑在**同一处**声明,消灭双处维护的漂移。

设计约束:本模块顶层只依赖标准库(检索相关 import 全部延迟到函数内),
保证离线单测(tests/test_tools.py)不加载 jieba/torch 等重依赖。
"""
import ast
import json
import operator
from pathlib import Path

# ========== 注册表 ==========

_REGISTRY = {}  # name -> {"fn": callable, "schema": OpenAI function calling 格式}


def tool(name: str, description: str, parameters: dict):
    """装饰器:注册工具函数,并生成喂给 LLM 的 schema。"""
    def deco(fn):
        _REGISTRY[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return fn
    return deco


def get_tool_schemas() -> list[dict]:
    """全部已注册工具的 schema(LLM 的 tools 参数)。"""
    return [v["schema"] for v in _REGISTRY.values()]


def execute_tool(name: str, arguments: dict) -> str:
    """
    统一分发。任何失败都转成字符串结果回注 LLM(让它自己换策略),
    不让异常炸掉 Agent 循环 —— 工具的鲁棒性边界在这里统一画。
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"未知工具: {name}(可用: {', '.join(_REGISTRY)})"
    try:
        return str(entry["fn"](**(arguments or {})))
    except TypeError as e:
        return f"参数不匹配: {e}"
    except Exception as e:
        return f"工具执行失败({name}): {e}"


# ========== 工具实现 ==========

@tool(
    name="search",
    description=(
        "从知识库检索与问题相关的文档片段(多文档,向量+BM25 混合检索)。"
        "当用户的问题涉及特定技术细节、概念解释、操作步骤时,"
        "调用此工具获取参考资料再组织回答。返回最相关的 10 个文档块(含来源)。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词或问题"}
        },
        "required": ["query"]
    },
)
def tool_search(query: str) -> str:
    """两段式多文档检索:RRF 召回 top-10 → 交叉编码器精排 top-5,带来源返回。

    精排不可省(真机踩坑):跨文档 RRF 有"小文档名次压缩"——3 块的小库里
    任何块都天然是双路 top-3,弱匹配也拿满融合分;交叉编码器逐对打分,
    不受名次分布影响,负责把真相关的块提上来。
    """
    from agent_project.retriever.hybrid import hybrid_search_all
    try:
        hits = hybrid_search_all(query, k=10)
    except FileNotFoundError as e:
        return str(e)
    try:
        from agent_project.reranker import rerank
        hits = rerank(query, hits, top_k=5)
    except Exception:
        pass  # 精排失败(如模型缺失)退回 RRF 序,检索不中断
    lines = []
    for i, h in enumerate(hits, 1):
        preview = h["chunk"][:200].replace("\n", " ")
        src = h["meta"].get("source", "?")
        score = h.get("rerank_score", h.get("rrf_score", 0))
        lines.append(f"【{i}】(来源:{src} | 分={score:.4f}) {preview}...")
    n_docs = len({h["meta"].get("source") for h in hits})
    return (f"检索到 {len(hits)} 个相关文档块(来自 {n_docs} 份文档):\n\n"
            + "\n\n".join(lines))


@tool(
    name="direct_answer",
    description=(
        "直接回答用户问题,不需要从知识库检索。"
        "当问题与知识库内容无关(如问候、简单常识、闲聊)时,调用此工具直接给出回答。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "直接回答的内容"}
        },
        "required": ["answer"]
    },
)
def tool_direct_answer(answer: str) -> str:
    """无需检索的直接回答(给 LLM 一个"不检索"的显式选项,SPEC-003 AC2)。"""
    return answer


# ---- calculator:ast 白名单安全求值 ----
# 只放行数字常量与算术运算节点;Name/Call/Attribute 等一律拒绝。
# 这样 "import os"(语法错)、__import__('os')(Call)、"abc"(Name) 都无法执行。

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不允许的表达式元素: {type(node).__name__}(仅支持四则/乘方/取模/括号)")


@tool(
    name="calculator",
    description="计算数学表达式,支持 + - * / ** % // 和括号。任何算术问题都用此工具,不要心算。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式,如 (1+2)*3"}
        },
        "required": ["expression"]
    },
)
def tool_calculator(expression: str) -> str:
    """算术求值(ast 白名单,非 eval,无注入面)。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)  # 6.0 → 6,观感
        return f"{expression} = {result}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return f"无法计算 '{expression}': {e}"


@tool(
    name="list_documents",
    description="列出当前知识库中已入库、可被 search 检索的全部文档及各自的块数。",
    parameters={"type": "object", "properties": {}, "required": []},
)
def tool_list_documents() -> str:
    """列知识库清单(让 LLM/用户知道库里有什么,再决定怎么搜)。"""
    from agent_project.retriever.hybrid import discover_docs
    pairs = discover_docs()
    if not pairs:
        return "知识库为空。入库方法: python -m agent_project.ingest <文件>"
    lines = []
    for i, (_, mpath) in enumerate(pairs, 1):
        with open(mpath, encoding="utf-8") as f:
            data = json.load(f)
        src = data["metas"][0]["source"] if data["metas"] else Path(mpath).stem
        lines.append(f"【{i}】{src}({len(data['chunks'])} 块)")
    return f"知识库共 {len(pairs)} 份文档:\n" + "\n".join(lines)
