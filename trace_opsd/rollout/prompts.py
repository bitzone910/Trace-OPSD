# ruff: noqa: E501
"""Prompts and Qwen2.5 chat messages for Trace-OPSD rollout synthesis."""

from __future__ import annotations

from typing import Final

DIRECT_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant that can solve the given question step by step. "
    "Given a question, you need to first think about the reasoning process in the "
    "mind and then provide the answer. During thinking, the reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags "
    "respectively. For example, <think> This is the reasoning process. </think> "
    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, "
    "the final exact answer is enclosed within \\boxed{} with latex format."
)

TOOL_PROMPT_TEMPLATE: Final[str] = r"""You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. This process MUST be enclosed within <think> </think> tags.

After completing your reasoning, choose only one of the following actions (exactly one):

(1) If you find you lack some knowledge, you can call a search engine to get more external information using the following format:
<search> your query </search>.

(2) If you need to perform mathematical calculations, you can call a Python interpreter using the following format:
<python> your code </python>.
Your code should include a final print(...) statement so that the interpreter returns a usable result.

(3) If you have enough information to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. The final exact answer must be enclosed within \boxed{{}} in LaTeX format. For example:
<answer>The final answer is \[ \boxed{{answer here}} \]</answer>."""

INVALID_ACTION_FEEDBACK: Final[str] = r"""Your previous response did not contain a complete tool call or a final <answer>...</answer> block with a boxed answer.

Continue solving the original problem. You may reason freely, but your next response must include exactly one of:

(1) A complete Python tool call:
<python> your Python code </python>

(2) A complete search tool call:
<search> your search query </search>

(3) A complete final answer block containing:
<answer>\boxed{your final answer}</answer>

Do not generate tool results yourself."""

FINAL_ANSWER_ONLY_INSTRUCTION: Final[str] = (
    r"The tool-call budget is now exhausted. Do not call any tool. Complete the "
    r"reasoning and return the final answer now. The final exact answer must be "
    r"enclosed within <answer>\boxed{your final answer}</answer>."
)


def _problem_message(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a nonempty string")
    return problem.strip()


def build_direct_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": _problem_message(problem)},
    ]


def build_tool_prompt(task_description: str) -> str:
    return TOOL_PROMPT_TEMPLATE.format(
        task_description=_problem_message(task_description)
    )


def build_tool_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_tool_prompt(problem)}]


def build_direct_prompt(problem: str) -> str:
    messages = build_direct_messages(problem)
    return f"{messages[0]['content']}\n\n{messages[1]['content']}"


def build_final_answer_instruction() -> str:
    return FINAL_ANSWER_ONLY_INSTRUCTION


__all__ = [
    "DIRECT_SYSTEM_PROMPT",
    "FINAL_ANSWER_ONLY_INSTRUCTION",
    "INVALID_ACTION_FEEDBACK",
    "TOOL_PROMPT_TEMPLATE",
    "build_direct_messages",
    "build_direct_prompt",
    "build_final_answer_instruction",
    "build_tool_messages",
    "build_tool_prompt",
]
