"""Compatibility helpers retained verbatim for the public source closure.
Unrelated legacy commands and API backends are not part of this export.
"""
from __future__ import annotations


from collections import Counter, defaultdict


from typing import Any


def score_visor_predictions(
    *,
    predictions: dict[str, Any],
    hidden_index: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_request_id = {
        row["request_id"]: row for row in hidden_index.get("answers", [])
    }
    responses_by_request_id = {
        row.get("request_id"): row for row in predictions.get("responses", [])
        if isinstance(row, dict)
    }
    rows = []
    correct = 0
    type_counts: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    score_request_ids = list(hidden_by_request_id)
    predicted_request_ids = {
        str(request_id)
        for request_id in responses_by_request_id
        if request_id is not None
    }
    if 0 < len(predicted_request_ids) < len(hidden_by_request_id):
        score_request_ids = [
            request_id
            for request_id in score_request_ids
            if request_id in predicted_request_ids
        ]
    for request_id in score_request_ids:
        hidden = hidden_by_request_id[request_id]
        response = responses_by_request_id.get(request_id)
        query_type = hidden["query_type"]
        type_counts[query_type] += 1
        if response is None:
            normalized_answer = None
            is_correct = False
            error = "missing_prediction"
        else:
            normalized_answer = _normalize_answer(response.get("answer"), query_type=query_type)
            is_correct = normalized_answer == hidden["expected_answer"]
            error = None
        if is_correct:
            correct += 1
            type_correct[query_type] += 1
        rows.append(
            {
                "request_id": request_id,
                "query_id": hidden["query_id"],
                "query_type": query_type,
                "expected_answer": hidden["expected_answer"],
                "normalized_answer": normalized_answer,
                "correct": is_correct,
                "error": error,
            }
        )
    total = len(rows)
    by_type = {
        query_type: {
            "correct": type_correct[query_type],
            "total": type_counts[query_type],
            "accuracy": type_correct[query_type] / type_counts[query_type],
        }
        for query_type in sorted(type_counts)
    }
    return {
        "kind": "epic_visor_dynamic_smoke_score",
        "fixture_or_model": predictions.get("fixture") or predictions.get("model"),
        "backend": predictions.get("backend"),
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "by_query_type": by_type,
        "claim_boundary": (
            "Scores use the hidden VISOR smoke index after predictions exist. "
            "They are smoke diagnostics, not population-level VISOR benchmark claims."
        ),
        "rows": rows,
    }


def _normalize_answer(answer: Any, *, query_type: str) -> Any:
    if query_type == "contact_history_frames":
        if not isinstance(answer, list):
            return None
        normalized = []
        for item in answer:
            if isinstance(item, bool):
                return None
            if isinstance(item, int):
                normalized.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                normalized.append(int(item.strip()))
            else:
                return None
        return sorted(set(normalized))
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"true", "yes", "touching", "touched"}:
            return True
        if lowered in {"false", "no", "not touching", "not_touched"}:
            return False
    return None


