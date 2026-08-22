# -*- coding: utf-8 -*-
"""
SPEC-007 AC1/AC2/AC3 —— 会话终端离线验证(注入 read/write,mock run)。

不依赖 pytest、不碰真终端、不调 LLM,直接运行:
    python tests/test_chat.py

真机 AC4/AC5(管道驱动真实 API)由 temp 脚本覆盖。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_project.chat import chat_loop, list_sessions, load_session, save_session


def test_ac1_roundtrip():
    """AC1:save_session/load_session 无损往返(history 深相等,stats 数值相等)。"""
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "答1"},
    ]
    stats = {"llm_calls": 2, "tool_calls": 1, "prompt_tokens": 100,
             "completion_tokens": 50, "turns": 1}
    with tempfile.TemporaryDirectory() as td:
        path = save_session("t1", history, stats, sessions_dir=td)
        assert path.name == "t1.json"
        payload = load_session("t1", sessions_dir=td)
        assert payload["history"] == history, "history 应无损"
        for k, v in stats.items():
            assert payload["stats"][k] == v, k
        assert "saved_at" in payload
        # 不存在的会话 → FileNotFoundError
        try:
            load_session("nope", sessions_dir=td)
            raise AssertionError("应抛 FileNotFoundError")
        except FileNotFoundError:
            pass
    print("AC1 ✓ 会话序列化往返无损;缺失会话明确报错")


def _mock_run_factory(calls):
    """mock run:记录 (question, history),返回可预测的 (answer, 新history, stats)。"""
    def run(question, history=None):
        calls.append((question, history))
        new_hist = list(history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"echo:{question}"},
        ]
        return (f"echo:{question}", new_hist,
                {"llm_calls": 1, "tool_calls": 0, "prompt_tokens": 10,
                 "completion_tokens": 5, "history_turns": 1})
    return run


def _drive(inputs, sessions_dir):
    """跑一遍 REPL,返回 (输出行, mock 的调用记录)。"""
    outs, calls = [], []
    it = iter(inputs)
    chat_loop(read=lambda prompt="": next(it),
              write=outs.append,
              run=_mock_run_factory(calls),
              sessions_dir=sessions_dir)
    return outs, calls


def test_ac2_basic_flow():
    """AC2:脚本化输入 ['你好','/exit'] → 两轮交互(1问1答)后正常退出。"""
    with tempfile.TemporaryDirectory() as td:
        outs, calls = _drive(["你好", "/exit"], td)
    assert len(calls) == 1 and calls[0][0] == "你好"
    assert calls[0][1] is None                      # 新会话首问无历史
    assert any("echo:你好" in o for o in outs)      # 答案经 write 输出
    assert outs[-1] == "再见。"
    print("AC2 ✓ REPL 脚本化驱动:一问一答后 /exit 正常退出")


def test_ac2_eof():
    """/exit 之外:输入流关闭(EOFError)也干净退出。"""
    def read_eof(prompt=""):
        raise EOFError
    outs = []
    chat_loop(read=read_eof, write=outs.append,
              run=_mock_run_factory([]), sessions_dir=None)
    assert any("输入结束" in o for o in outs)
    print("✓ EOF 干净退出")


def test_ac3_commands():
    """AC3:/new 清空、/save 落盘、/load 恢复(mock run 收到的 history 与保存前一致)、/list 列出。"""
    with tempfile.TemporaryDirectory() as td:
        outs, calls = _drive([
            "第一问",        # 0: history=None → 产生 hist_A
            "/save t1",     # 落盘
            "/new",         # 清空
            "第二问",        # 1: history 应为 None
            "/load t1",     # 恢复
            "第三问",        # 2: history 应等于 hist_A
            "/list",
            "/exit",
        ], td)
        assert calls[1][1] is None, "/new 后应无历史"
        # 第三问收到的 history 应等于第一问后保存的历史(user=第一问 在最前)
        assert calls[2][1] is not None
        assert len(calls[2][1]) == 2 and calls[2][1][0]["content"] == "第一问"
        assert any("t1" in o for o in outs if "已存会话" in o), "/list 应含 t1"
        # 载入的统计也恢复了(turns=1 来自存档)
        assert any("恢复 1 轮" in o for o in outs)
        # 落盘文件确实存在
        assert list_sessions(td) == ["t1"]
    print("AC3 ✓ /new /save /load /list 语义全部正确")


test_ac1_roundtrip()
test_ac2_basic_flow()
test_ac2_eof()
test_ac3_commands()
print("\n全部通过:SPEC-007 会话终端离线验证 ✓")
