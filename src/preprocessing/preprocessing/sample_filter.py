from __future__ import annotations

from ..core.text_utils import whitespace_tokens


MAX_ANSWER_TOKENS = 20


def count_answer_tokens(answer: str) -> int:
    return len(whitespace_tokens(answer))


def should_keep_sample(answer_token_count: int, max_answer_tokens: int = MAX_ANSWER_TOKENS) -> bool:
    return answer_token_count <= max_answer_tokens
