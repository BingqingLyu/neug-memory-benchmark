"""DashScope OpenAI 兼容端点客户端：答题模型 + judge 模型。

协议依据 benchmark-plan.md 3.1：answer=qwen-plus，judge=qwen-max，temperature=0，
版本 pin 死。API key 从环境变量 DASHSCOPE_API_KEY 读取，禁止硬编码。
"""
import os
import time

from openai import OpenAI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ANSWER_MODEL = "qwen-plus"
DEFAULT_JUDGE_MODEL = "qwen-max"


class LLMClient:
    def __init__(self, base_url=None, api_key=None, timeout=120, max_retries=4):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY not set. export DASHSCOPE_API_KEY=... "
                "(never hardcode keys in code or config)"
            )
        self.client = OpenAI(
            base_url=base_url or os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            api_key=self.api_key,
            timeout=timeout,
        )
        self.max_retries = max_retries

    def chat(self, model, prompt, system=None, temperature=0.0):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001 - retry all transient errors
                last_err = e
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def answer(self, prompt, system=None, model=None):
        return self.chat(model or DEFAULT_ANSWER_MODEL, prompt, system=system)

    def judge(self, prompt, system=None, model=None):
        return self.chat(model or DEFAULT_JUDGE_MODEL, prompt, system=system)
