# -*- coding: utf-8 -*-
"""
SPEC-004 AC2/AC5 —— 轮间历史构造纯函数(_build_history)的离线验证。

纯函数测试,不依赖 pytest、不调 LLM、不加载任何模型,直接运行:
    python tests/test_history.py

真机部分(AC1 跨轮记忆 / AC3 协议完整 / AC4 向后等价)由 main.py 阶段六演示覆盖。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_project.agent.agent import SYSTEM_PROMPT, _build_history


def test_ac2_first_turn_compression():
    """AC2:一轮含 search 的对话完成后,返回历史只留 [system, user, assistant答案]。

    对照 —— 轮内工作列表长这样(中间消息全都不该泄漏进历史):
      [system, user, assistant(tool_calls=search), tool(2500字符结果), assistant(答案)]
    """
    working_list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Wireshark 怎么解密 HTTPS 流量?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search", "arguments": "{\"query\": \"https 解密\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "检索到 10 个相关文档块:\n\n【1】..."},
        {"role": "assistant", "content": "五步:Edit→Preferences→SSL→..."},
    ]
    answer = working_list[-1]["content"]
    # 工作列表用完即弃:历史由 (prev=None, 原始question, 最终answer) 显式重建
    hist = _build_history(None, working_list[1]["content"], answer)

    assert hist[0]["role"] == "system"
    assert sum(1 for m in hist if m["role"] == "system") == 1, "system 必须在头部且仅一条"
    assert hist[-2] == {"role": "user", "content": "Wireshark 怎么解密 HTTPS 流量?"}
    assert hist[-1] == {"role": "assistant", "content": answer}
    assert not any(m["role"] == "tool" for m in hist), "tool 消息不得进入历史"
    assert not any(m.get("tool_calls") for m in hist), "带 tool_calls 的 assistant 不得进入历史"
    print("AC2 ✓ 轮间压缩:tool 中间消息被丢弃,历史 = [system, user, assistant]")

    # 续轮:第 2 轮的历史含第 1 轮问答(AC1 的离线前提 —— 指代所需的上下文确实在)
    hist2 = _build_history(hist, "你说的第二步里的 SSL 协议设置,具体在哪个菜单打开?", "在 Edit 菜单...")
    assert len(hist2) == 5, hist2
    assert sum(1 for m in hist2 if m["role"] == "system") == 1, "续轮不得重复添加 system"
    assert hist2[1]["content"].startswith("Wireshark"), "第 1 轮问题保留在历史"
    assert hist2[-1]["role"] == "assistant"
    print("AC2 ✓ 续轮拼接:第 2 轮历史 = [system, 轮1问答对, 轮2问答对],system 无重复")


def test_ac5_window_truncation():
    """AC5:system + 12 轮历史 + 本轮(共 13 轮),max_turns=10 → 留 system + 最近 10 轮,最早 3 轮整对丢弃。"""
    prev = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(1, 13):  # 已有 12 轮
        prev.append({"role": "user", "content": f"问题{i}"})
        prev.append({"role": "assistant", "content": f"答案{i}"})

    hist = _build_history(prev, "问题13", "答案13", max_turns=10)

    assert len(hist) == 1 + 10 * 2, f"应为 system + 10轮×2条,实际 {len(hist)} 条"
    assert hist[0]["role"] == "system", "system 永远保留"
    assert hist[1]["content"] == "问题4", "最早 3 轮(问题1/2/3)应被整对丢弃"
    assert hist[-2]["content"] == "问题13"
    assert hist[-1] == {"role": "assistant", "content": "答案13"}
    # 无残缺对:去掉 system 后严格 user/assistant 交替
    roles = [m["role"] for m in hist[1:]]
    assert roles[::2] == ["user"] * 10 and roles[1::2] == ["assistant"] * 10
    print("AC5 ✓ 窗口截断:保留最近 10 轮,最早 3 轮整对丢弃,无残缺对")


test_ac2_first_turn_compression()
test_ac5_window_truncation()
print("\n全部通过:SPEC-004 压缩/截断纯函数 ✓")
