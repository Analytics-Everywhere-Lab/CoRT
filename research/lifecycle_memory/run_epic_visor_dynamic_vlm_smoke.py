"""Compatibility helpers retained verbatim for the public source closure.
Unrelated legacy commands and API backends are not part of this export.
"""
from __future__ import annotations


import json


from pathlib import Path


import re


from typing import Any


def _show_frame_labels(*, request: dict[str, Any], pack_presentation: dict[str, Any]) -> bool:
    presentation = request.get("presentation") if isinstance(request.get("presentation"), dict) else {}
    return not (
        presentation.get("frame_labels") == "none"
        or pack_presentation.get("frame_labels") == "none"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _resolve_path(path: Any, *, project_root: Path, pack_root: Path | None = None) -> Path:
    resolved = Path(str(path))
    if not resolved.is_absolute():
        project_candidate = project_root / resolved
        pack_candidate = (pack_root / resolved) if pack_root is not None else None
        if project_candidate.exists() or pack_candidate is None:
            resolved = project_candidate
        else:
            resolved = pack_candidate
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


