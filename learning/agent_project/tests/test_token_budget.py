# -*- coding: utf-8 -*-
"""
SPEC-006 AC1/AC2/AC5 —— token 估算与预算截断纯函数的离线验证。

不依赖 pytest、不调 LLM,直接运行:
    python tests/test_token_budget.py

真机部分(AC3 记账累计 / AC4 调用方)由 temp 脚本 + main.py 阶段六覆盖。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_project.agent.agent import SYSTEM_PROMPT, _build_history, _estimate_tokens, _messages_tokens


def test_ac1_estimate():
    """AC1:中文≈1字1token;英文≈4字符/token;混合按比例叠加;单调不减。"""
    assert _estimate_tokens("一" * 100) == 100
    assert _estimate_tokens("a" * 400) == 100          # (400+3)//4
    assert _estimate_tokens("一" * 50 + "a" * 200) == 50 + 50
    assert _estimate_tokens("") == 0
    for t in ["一二三", "hello world", "混合mixed文本text"]:
        assert _estimate_tokens(t) <= _estimate_tokens(t + "更多内容"), t
    print("AC1 ✓ 估算:中文 1:1、英文 4:1、混合叠加、单调不减")


def _mk_prev(n_turns):
    """构造 n_turns 轮历史,每轮 user+assistant 各 101 估算 token(100汉字+数字)。"""
    prev = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(1, n_turns + 1):
        prev.append({"role": "user", "content": "问" * 100 + str(i)})      # 100 + 1 = 101
        prev.append({"role": "assistant", "content": "答" * 100 + str(i)})  # 101
    return prev


def test_ac2_budget_truncation():
    """AC2:超预算从最旧轮整对丢弃;system 保留;无残缺对;结果入预算。"""
    prev = _mk_prev(12)                       # 12 轮 × 202 tok = 2424
    hist = _build_history(prev, "问" * 100 + "13", "答" * 100 + "13",
                          max_turns=10, max_history_tokens=1000)
    # max_turns 先切到 10 轮(2020 tok)→ 预算 1000 再切:4 轮 808 ≤ 1000,5 轮 1010 > 1000
    assert len(hist) == 1 + 4 * 2, f"应剩 system+4轮,实际 {len(hist)} 条"
    assert hist[0]["role"] == "system"
    assert hist[1]["content"].endswith("10"), "最早保留的应是第10轮(第4~9轮被丢)"
    assert hist[-1]["content"].endswith("13")
    assert _messages_tokens(hist[1:]) <= 1000, "截断后应入预算"
    roles = [m["role"] for m in hist[1:]]
    assert roles[::2] == ["user"] * 4 and roles[1::2] == ["assistant"] * 4
    print("AC2 ✓ 预算截断:12轮+本轮 → 双闸后剩4轮,整对丢弃、system 保留、入预算")


def test_ac5_tiny_budget():
    """AC5:极小预算(50)下仍至少保留最近 1 轮 —— 预算闸真实生效而非摆设。"""
    prev = _mk_prev(5)
    hist = _build_history(prev, "问" * 100 + "6", "答" * 100 + "6",
                          max_turns=10, max_history_tokens=50)
    assert len(hist) == 3, f"应只剩 system+最近1轮,实际 {len(hist)} 条"
    assert hist[1]["content"].endswith("6") and hist[2]["content"].endswith("6")
    print("AC5 ✓ 极小预算:退守到 system+最近1轮,不空、不残缺")


test_ac1_estimate()
test_ac2_budget_truncation()
test_ac5_tiny_budget()
print("\n全部通过:SPEC-006 估算/预算截断纯函数 ✓")
