# -*- coding: utf-8 -*-
"""
阶段4·生成 —— LLM 调用薄层(可换供应商)。

对外只暴露 chat(prompt) -> str。
当前接智谱 GLM(OpenAI 兼容接口);将来换本地 Ollama,
只改顶部 BASE_URL 和 MODEL 两行,其余代码不动 —— 这就是"薄抽象"的意义。
"""
import os

from openai import OpenAI

from agent_project.path_manager import PathManager

# ---- 供应商配置:换 LLM 只改这里 ----
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # 智谱;本地 Ollama 改为 http://localhost:11434/v1
MODEL = "glm-4.7-flash"  # 智谱免费款;本地改为 qwen2.5:7b


# ---------- .env 加载(裸实现,不引 python-dotenv) ----------
def _load_dotenv():
    """把项目根 .env 里的 KEY=VALUE 读进环境变量(已存在的不覆盖)。

    原理三步:读文件 → 逐行拆 KEY/VALUE → setdefault。
    好处:不为加载一个小文件多装一个依赖。
    """
    env_path = PathManager().PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# ---------- 客户端单例 ----------
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _load_dotenv()
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 ZHIPU_API_KEY:请在项目根的 .env 里配置(参考 .env.example)"
            )
        # max_retries:SDK 内置退避重试,扛免费模型高峰期的 429 限流
        _client = OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=5)
    return _client


def chat(prompt: str, temperature: float = 0.3) -> str:
    """把拼好的 prompt 发给 LLM,返回生成的答案文本。

    :param prompt: 完整 prompt(含检索到的上下文 + 问题)
    :param temperature: 采样温度;RAG 问答要"忠于资料",取低值减少自由发挥
    """
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
