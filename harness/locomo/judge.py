"""LoCoMo 判分：gpt 风格二分类 judge，对齐 Zep 修正协议（cat1-4，CORRECT/WRONG）。

prompt 保持各家 harness 原样、不为 Qwen 调优（3.1 协议 4 Prompt 纪律）；
如 Qwen 在个别 prompt 行为异常，统一修改并在 manifest 声明。
"""
import json
import re

JUDGE_SYSTEM = (
    "You are an impartial evaluator. Judge whether the model's answer is "
    "semantically correct given the reference answer. Respond with a JSON object: "
    '{"verdict": "CORRECT" or "WRONG", "reason": "<one sentence>"}.'
)

JUDGE_TEMPLATE = """Question: {question}

Reference answer: {reference}

Model answer: {answer}

Judge whether the model answer is correct given the reference. Minor phrasing differences are acceptable; factual contradictions or missing key facts are WRONG."""


def build_judge_prompt(question: str, reference, answer: str) -> str:
    ref = reference if isinstance(reference, str) else json.dumps(reference, ensure_ascii=False)
    return JUDGE_TEMPLATE.format(question=question, reference=ref, answer=answer)


def parse_verdict(raw: str) -> str:
    """从 judge 输出解析 CORRECT/WRONG，容错 JSON/纯文本。"""
    m = re.search(r'"verdict"\s*:\s*"(CORRECT|WRONG)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.search(r"\bCORRECT\b", raw, re.IGNORECASE) and not re.search(r"\bWRONG\b", raw, re.IGNORECASE):
        return "CORRECT"
    return "WRONG"


ANSWER_TEMPLATE = """You are a helpful assistant with access to retrieved memories. Answer the question concisely using ONLY the provided memory context. If the context does not contain the answer, say you don't know.

Memory context:
{context}

Question: {question}

Answer:"""


def build_answer_prompt(context: str, question: str) -> str:
    return ANSWER_TEMPLATE.format(context=context, question=question)
