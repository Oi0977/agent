# -*- coding: utf-8 -*-
"""
LLM 调用层(可换供应商)。

对外暴露:
  chat(messages, ...)            —— Agent 模式:多轮消息,返回完整 Choice
  rag_answer 直接调的也是它       —— 旧模式:单轮 prompt,返回纯文本

当前接智谱 GLM(OpenAI 兼容接口);将来换本地 Ollama,只改顶部两行。
"""
import os

from openai import OpenAI

from agent_project.path_manager import PathManager

# ---- 供应商配置:换 LLM 只改这里 ----
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4.7-flash"


# ---------- .env 加载 ----------
def _load_dotenv():
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
            raise RuntimeError("未找到 ZHIPU_API_KEY:请在项目根的 .env 里配置")
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


# ---------- 公开接口 ----------

def chat(messages, tools=None, temperature: float = 0.3, max_retries: int = 5):
    """
    LLM 调用(通用接口,Agent 和 RAG 共用)。

    :param messages: [{"role": "user/assistant/tool", "content": "...", ...}]
    :param tools: OpenAI function calling 格式的工具列表(可选)
    :param temperature: 采样温度
    :param max_retries: 429 限流重试次数(每次翻倍等待)
    :return: ChatCompletion 对象(含 .choices[0].message,可判 .tool_calls)
    """
    import time
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools

    for attempt in range(max_retries):
        try:
            return _get_client().chat.completions.create(**kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"    ⚠ 429限流,等待{wait}秒后重试({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
