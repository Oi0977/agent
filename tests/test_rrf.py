# -*- coding: utf-8 -*-
"""
SPEC-002 AC2 —— RRF 融合数学的最小验证。

纯函数测试,不依赖 pytest,直接运行:
    python tests/test_rrf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_project.retriever.hybrid import _rrf_fuse

# 构造两路排名(spec AC2 用例):
#   路A = [x, y, z]  → x 第1, y 第2, z 第3
#   路B = [z, x]     → z 第1, x 第2
# 手算期望(k_rrf=60,每名得分 1/(60+名次)):
#   x = 1/61(路A第1) + 1/62(路B第2)
#   y = 1/62(仅路A第2,路B未上榜)
#   z = 1/63(路A第3) + 1/61(路B第1)
fused = _rrf_fuse([["x", "y", "z"], ["z", "x"]], k_rrf=60)

assert abs(fused["x"] - (1 / 61 + 1 / 62)) < 1e-12, fused["x"]
assert abs(fused["y"] - 1 / 62) < 1e-12, fused["y"]
assert abs(fused["z"] - (1 / 63 + 1 / 61)) < 1e-12, fused["z"]

# 排序:x(两路都靠前) > z(两路上榜,一路第1) > y(单路第2)
order = sorted(fused, key=fused.get, reverse=True)
assert order == ["x", "z", "y"], order

print("AC2 ✓ RRF 数学正确")
print("  x = 1/61+1/62 =", round(fused["x"], 6))
print("  z = 1/61+1/63 =", round(fused["z"], 6))
print("  y = 1/62      =", round(fused["y"], 6))
print("  排序:", order)
