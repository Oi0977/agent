# -*- coding: utf-8 -*-
"""
交互式会话终端(SPEC-007)。

    python -m agent_project.chat

REPL 持续多轮对话:历史跨轮传递(SPEC-004 记忆)、每轮显示 token 统计
(SPEC-006 记账)、会话可存取(/save /load,JSON 落 data/sessions/)。

读写与 run 全部可注入(chat_loop(read, write, run))→ 离线脚本化测试,
不碰真终端、不调真 LLM(tests/test_chat.py)。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from agent_project.path_manager import PathManager

SESSIONS_DIR = PathManager().DATA_ROOT / "sessions"

HELP = ("命令:/new 新会话 | /save [名字] 保存当前会话 | /load <名字> 载入 "
        "| /list 列已存会话 | /help 帮助 | /exit 退出")


# ========== 会话持久化(纯函数,路径可注入) ==========

def save_session(name, history, stats, sessions_dir=None) -> Path:
    """history + stats → <name>.json;返回落盘路径。"""
    d = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "history": history,
        "stats": stats,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_session(name, sessions_dir=None) -> dict:
    """读回会话;不存在/损坏分别抛 FileNotFoundError/ValueError,由调用方提示。"""
    d = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
    path = d / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"会话不存在: {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"会话文件损坏({name}): {e}") from e


def list_sessions(sessions_dir=None) -> list[str]:
    d = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def _empty_totals() -> dict:
    return {"llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "turns": 0}


def _totals_from(payload_stats: dict) -> dict:
    t = _empty_totals()
    for k in t:
        t[k] = int(payload_stats.get(k, 0) or 0)
    return t


# ========== REPL 主循环 ==========

def chat_loop(read=input, write=print, run=None, sessions_dir=None):
    """
    :param read: 取一行输入的函数(默认 input;测试注入脚本序列)
    :param write: 输出函数(默认 print;测试注入收集器)
    :param run: 问答函数,签名 run(question, history=...) → (answer, history, stats);
                默认 agent.run。测试可 mock —— 同时跳过知识库探测的重导入
    """
    if run is None:
        from agent_project.agent import run as default_run
        run = default_run
        from agent_project.retriever.hybrid import discover_docs
        n_docs = len(discover_docs())
        if n_docs:
            write(f"(知识库:{n_docs} 份文档;输入问题开始,/help 查看命令)")
        else:
            write("(知识库为空!先运行: python -m agent_project.ingest <文件>)")

    history = None
    totals = _empty_totals()

    while True:
        try:
            line = read("你> ").strip()
        except EOFError:                     # Ctrl+Z / Ctrl+D / 管道关闭
            write("(输入结束,退出)")
            return
        if not line:
            continue

        if line == "/exit":
            write("再见。")
            return
        if line == "/help":
            write(HELP)
            continue
        if line == "/new":
            history = None
            totals = _empty_totals()
            write("(已开新会话)")
            continue
        if line.startswith("/save"):
            parts = line.split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else datetime.now().strftime("s%m%d_%H%M")
            save_session(name, history, totals, sessions_dir)
            write(f"(已保存会话 '{name}' → data/sessions/{name}.json)")
            continue
        if line.startswith("/load"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                write("(用法:/load <名字>;先 /list 看已存会话)")
                continue
            try:
                payload = load_session(parts[1].strip(), sessions_dir)
            except (FileNotFoundError, ValueError) as e:
                write(f"(载入失败:{e};当前会话未受影响)")
                continue
            history = payload.get("history") or None
            totals = _totals_from(payload.get("stats") or {})
            n = sum(1 for m in (history or []) if m.get("role") == "user")
            write(f"(已载入 '{parts[1].strip()}',恢复 {n} 轮对话)")
            continue
        if line == "/list":
            names = list_sessions(sessions_dir)
            write("已存会话:" + (", ".join(names) if names else "(无)"))
            continue
        if line.startswith("/"):
            write(f"(未知命令 {line};/help 查看可用命令)")
            continue

        # 普通对话:run 内部 verbose 关掉(输出由 write 统一走,避免双份横幅)
        try:
            answer, history, st = run(line, history=history)
        except Exception as e:
            write(f"(出错了:{e};会话已保留,可继续输入)")
            continue
        for k in ("llm_calls", "tool_calls", "prompt_tokens", "completion_tokens"):
            totals[k] += st.get(k, 0) or 0
        totals["turns"] += 1
        write(f"[token] 本轮 prompt {st.get('prompt_tokens', 0)} / "
              f"completion {st.get('completion_tokens', 0)} | "
              f"会话累计 {totals['prompt_tokens']}/{totals['completion_tokens']}")
        write(f"AI: {answer}")


if __name__ == "__main__":
    chat_loop()
