"""Compatibility helpers retained verbatim for the public source closure.
Unrelated legacy commands and API backends are not part of this export.
"""
from __future__ import annotations


import json


import re


from pathlib import Path


from typing import Any


from research.lifecycle_memory.run_epic_visor_dynamic_vlm_smoke import (
    _parse_json_object,
    _read_json,
    _resolve_path,
    _show_frame_labels,
)


PROMPT_ECHO_MIN_CHARS = 48


def _safe_request_id(request_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", request_id)[:100]


def _blank_image_same_size(source_path: Path, out_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - minimal env fallback
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(source_path.read_bytes())
        return out_path
    with Image.open(source_path) as image:
        blank = Image.new(image.mode, image.size, "white")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blank.save(out_path, format="JPEG", quality=95)
    return out_path


def _parse_visor_answer(raw_text: str) -> tuple[dict[str, Any], str | None]:
    text = str(raw_text or "").strip()
    if not text:
        return {"answer": None, "abstain": True, "abstention_reason": "empty_output", "confidence": 0.0}, "empty_output"
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    answer_text = text_without_think or text
    try:
        parsed = _parse_json_object(answer_text)
        answer = parsed.get("answer")
        if isinstance(answer, bool):
            return {
                "answer": answer,
                "abstain": bool(parsed.get("abstain", False)),
                "abstention_reason": parsed.get("abstention_reason"),
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            }, None
        if isinstance(answer, str):
            lowered = answer.strip().lower()
            if lowered in {"true", "yes"}:
                return {"answer": True, "abstain": False, "abstention_reason": None, "confidence": 0.0}, None
            if lowered in {"false", "no"}:
                return {"answer": False, "abstain": False, "abstention_reason": None, "confidence": 0.0}, None
    except (json.JSONDecodeError, ValueError):
        pass
    lowered = answer_text.strip().lower()
    if re.fullmatch(r"yes\.?", lowered):
        return {"answer": True, "abstain": False, "abstention_reason": None, "confidence": 0.0}, None
    if re.fullmatch(r"no\.?", lowered):
        return {"answer": False, "abstain": False, "abstention_reason": None, "confidence": 0.0}, None
    if lowered in {"true", "false"}:
        return {
            "answer": lowered == "true",
            "abstain": False,
            "abstention_reason": None,
            "confidence": 0.0,
        }, None
    matches = re.findall(r"\b(yes|no|true|false)\b", lowered)
    if matches:
        last = matches[-1]
        return {
            "answer": last in {"yes", "true"},
            "abstain": False,
            "abstention_reason": None,
            "confidence": 0.0,
        }, None
    return {"answer": None, "abstain": True, "abstention_reason": "parse_failure", "confidence": 0.0}, "parse_failure"


def _llamacpp_yes_no_prompt(
    request: dict[str, Any],
    *,
    pack_presentation: dict[str, Any] | None = None,
    blank_image: bool = False,
) -> str:
    query = request.get("query") if isinstance(request.get("query"), dict) else {}
    query_type = str(query.get("query_type", ""))
    presentation = request.get("presentation") if isinstance(request.get("presentation"), dict) else {}
    pack_presentation = pack_presentation or {}
    order_stripped = (
        presentation.get("mode") == "order_stripped"
        or pack_presentation.get("mode") == "order_stripped"
        or presentation.get("frame_labels") == "none"
        or pack_presentation.get("frame_labels") == "none"
    )
    if query_type == "contact_order_before":
        if order_stripped:
            answer_rule = (
                "YES means the first visible hand-object touch of the target object "
                "happens earlier in the shown image sequence than the first visible "
                "hand-object touch of the comparison object. NO means otherwise."
            )
        else:
            answer_rule = (
                "YES means the first hand-object touch of the target object happens "
                "at a lower frame_index than the first hand-object touch of the "
                "comparison object. NO means otherwise."
            )
    else:
        if order_stripped:
            answer_rule = (
                "YES means a hand is touching the target object in the final shown image. "
                "NO means the target is visible but no hand is touching it."
            )
        else:
            answer_rule = (
                "YES means a hand is touching the target object in the final shown "
                "frame_index. NO means the target is visible but no hand is touching it."
            )
    if order_stripped:
        frame_line = (
            "The contact sheet is the only image input. Images are shown in sequence by "
            "layout order, left-to-right then top-to-bottom. No frame numbers, timestamps, "
            "or original filenames are provided."
        )
    else:
        frame_line = "The contact sheet is the only image input. It is ordered by public frame_index labels."
    lines = [
        "Answer one EPIC-KITCHENS VISOR dynamic contact question from the contact sheet.",
        frame_line,
        "Use only visible RGB evidence and the public question.",
        f"Question: {query.get('text')}",
        f"Query type: {query_type}",
        f"Target object: {query.get('target_label')}",
    ]
    comparison = query.get("comparison_target_label")
    if comparison:
        lines.append(f"Comparison object: {comparison}")
    if blank_image:
        lines.append(
            "The image may be intentionally blank for a control run. Even if the image is "
            "blank or uninformative, make your best forced-choice guess from the public "
            "question text. Do not abstain, refuse, or say the image is blank."
        )
    lines.extend(
        [
            answer_rule,
            "Do not think out loud. Do not describe the task. Do not write analysis.",
            "Your first and only output token must be YES or NO. Return exactly one token. Do not explain.",
        ]
    )
    return "\n".join(lines)


def _prompt_echo_detected(*, prompt: str, raw_text: str) -> bool:
    normalized_prompt = " ".join(prompt.split())
    normalized_output = " ".join(raw_text.split())
    if len(normalized_prompt) < PROMPT_ECHO_MIN_CHARS:
        return normalized_prompt and normalized_prompt in normalized_output
    for chunk_size in (240, 160, PROMPT_ECHO_MIN_CHARS):
        snippet = normalized_prompt[:chunk_size]
        if len(snippet) >= PROMPT_ECHO_MIN_CHARS and snippet in normalized_output:
            return True
    return False


def _answers_by_request_id(predictions: dict[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for row in predictions.get("responses", []):
        if isinstance(row, dict) and row.get("request_id") is not None:
            answers[str(row["request_id"])] = row.get("answer")
    return answers


def build_gate_report(
    *,
    responses: list[dict[str, Any]],
    stdout_cap: int,
    allow_one_label: bool,
    compare_predictions_path: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    parse_failures = [row for row in responses if row.get("error") or row.get("answer") is None]
    checks.append(
        {
            "name": "all_responses_parsed",
            "pass": not parse_failures,
            "failure_count": len(parse_failures),
            "failed_request_ids": [row.get("request_id") for row in parse_failures],
        }
    )

    stdout_violations = [
        row
        for row in responses
        if len(str(row.get("raw_text", ""))) > stdout_cap
    ]
    checks.append(
        {
            "name": "stdout_below_cap",
            "pass": not stdout_violations,
            "stdout_cap": stdout_cap,
            "failure_count": len(stdout_violations),
            "failed_request_ids": [row.get("request_id") for row in stdout_violations],
        }
    )

    echo_rows = [row for row in responses if row.get("prompt_echo")]
    checks.append(
        {
            "name": "no_prompt_echo",
            "pass": not echo_rows,
            "failure_count": len(echo_rows),
            "failed_request_ids": [row.get("request_id") for row in echo_rows],
        }
    )

    parsed_answers = [row.get("answer") for row in responses if isinstance(row.get("answer"), bool)]
    one_label = len(parsed_answers) >= 2 and len(set(parsed_answers)) == 1
    checks.append(
        {
            "name": "answer_distribution_not_all_one_label",
            "pass": allow_one_label or not one_label,
            "response_count": len(parsed_answers),
            "unique_answers": sorted(set(parsed_answers)),
            "allow_one_label": allow_one_label,
        }
    )

    compare_check: dict[str, Any] | None = None
    if compare_predictions_path is not None:
        compare_payload = _read_json(compare_predictions_path)
        current = _answers_by_request_id({"responses": responses})
        compare = _answers_by_request_id(compare_payload)
        shared_ids = sorted(set(current) & set(compare))
        differing = [request_id for request_id in shared_ids if current[request_id] != compare[request_id]]
        compare_check = {
            "name": "compare_predictions_differs",
            "pass": bool(differing),
            "compare_predictions": str(compare_predictions_path),
            "shared_request_count": len(shared_ids),
            "differing_request_ids": differing,
        }
        checks.append(compare_check)

    status = "pass" if all(check["pass"] for check in checks) else "fail"
    return {
        "kind": "epic_visor_dynamic_llamacpp_metal_gate_report",
        "status": status,
        "checks": checks,
    }


