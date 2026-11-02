"""Compatibility helpers retained verbatim for the public source closure.
Unrelated legacy commands and API backends are not part of this export.
"""
from __future__ import annotations


import math


from pathlib import Path


from typing import Any, Callable


def build_contact_sheet_image(
    *,
    request: dict[str, Any],
    frame_paths: list[Path],
    out_path: Path,
    thumb_width: int = 320,
    thumb_height: int = 240,
    columns: int = 2,
    show_labels: bool = True,
) -> Path:
    from PIL import Image, ImageDraw

    frames_meta = [
        frame
        for frame in request.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    if len(frames_meta) != len(frame_paths):
        frames_meta = [{} for _ in frame_paths]
    thumb_w = max(96, int(thumb_width))
    thumb_h = max(72, int(thumb_height))
    if not frame_paths:
        blank = Image.new("RGB", (thumb_w, thumb_h), "white")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blank.save(out_path, format="JPEG", quality=95)
        return out_path
    cols = min(max(1, int(columns)), len(frame_paths))
    rows = math.ceil(len(frame_paths) / cols)
    label_h = 24 if show_labels else 0
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet) if show_labels else None
    for index, (frame_meta, frame_path) in enumerate(zip(frames_meta, frame_paths)):
        col = index % cols
        row = index // cols
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = col * thumb_w + (thumb_w - image.width) // 2
            y = row * (thumb_h + label_h) + label_h
            sheet.paste(image, (x, y))
        if not show_labels:
            continue
        frame_index = frame_meta.get("frame_index", index)
        time_index = frame_meta.get("time", index)
        label = f"frame_index={frame_index} time={time_index}"
        header_top = row * (thumb_h + label_h)
        assert draw is not None
        draw.rectangle(
            [col * thumb_w, header_top, (col + 1) * thumb_w, header_top + label_h],
            fill=(245, 245, 245),
        )
        draw.text((col * thumb_w + 6, header_top + 6), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=95)
    return out_path


