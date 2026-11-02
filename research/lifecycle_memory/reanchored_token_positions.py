from __future__ import annotations

import hashlib
from typing import Any, Sequence


QUESTION_TOKEN_CONTRACT = (
    "last_token_of_last_exact_tokenizer_subsequence_match_for_the_literal_user_question"
)


def resolve_final_question_token(
    input_ids: Sequence[int], tokenizer: Any, question: str
) -> dict[str, Any]:
    """Resolve the last token of the literal question, excluding chat-template suffixes."""

    ids = [int(value) for value in input_ids]
    literal = str(question)
    variants: list[str] = []
    for text in (literal, literal.strip(), " " + literal.strip(), "\n" + literal.strip()):
        if text and text not in variants:
            variants.append(text)
    for variant in variants:
        try:
            encoded = [
                int(value)
                for value in tokenizer.encode(variant, add_special_tokens=False)
            ]
        except Exception:
            continue
        hits = [
            start
            for start in range(len(ids) - len(encoded) + 1)
            if encoded and ids[start : start + len(encoded)] == encoded
        ]
        if hits:
            start = hits[-1]
            index = start + len(encoded) - 1
            return {
                "status": "pass",
                "index": index,
                "question_token_count": len(encoded),
                "match_count": len(hits),
                "variant_sha256": hashlib.sha256(variant.encode("utf-8")).hexdigest(),
                "input_token_count": len(ids),
                "tokens_after_question": len(ids) - index - 1,
                "contract": QUESTION_TOKEN_CONTRACT,
            }
    return {
        "status": "fail",
        "reason": "exact_question_token_subsequence_not_found",
        "input_token_count": len(ids),
        "contract": QUESTION_TOKEN_CONTRACT,
    }
