#!/usr/bin/env python3
"""EgoHOS marker renderer with intended-pixel tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


@dataclass(frozen=True)
class MarkerStyle:
    """Frozen marker style for deterministic rendering."""

    name: str
    target_rgb: tuple[int, int, int]
    hand_rgb: tuple[int, int, int]
    fill_alpha: int
    mask_outline_width: int
    bbox_width: int
    target_label: str = "target"
    hand_label: str = "hand"


DISCOVERY_STYLE: Final[MarkerStyle] = MarkerStyle(
    name="discovery",
    target_rgb=(20, 150, 60),
    hand_rgb=(35, 105, 210),
    fill_alpha=58,
    mask_outline_width=1,
    bbox_width=4,
    target_label="target",
    hand_label="hand",
)

ALTERNATE_STYLE: Final[MarkerStyle] = MarkerStyle(
    name="alternate_descriptive",
    target_rgb=(206, 58, 171),
    hand_rgb=(233, 168, 42),
    fill_alpha=0,
    mask_outline_width=2,
    bbox_width=2,
    target_label="target",
    hand_label="hand",
)


def _validate_style(style: MarkerStyle) -> None:
    for field_name, value in (
        ("target_rgb", style.target_rgb),
        ("hand_rgb", style.hand_rgb),
    ):
        if len(value) != 3:
            raise ValueError(f"{field_name} must have exactly 3 channels")
        if any((not isinstance(channel, int)) or channel < 0 or channel > 255 for channel in value):
            raise ValueError(f"{field_name} channels must be 0..255 integers")
    if style.fill_alpha < 0 or style.fill_alpha > 255:
        raise ValueError("fill_alpha must be in [0, 255]")
    if style.mask_outline_width < 1:
        raise ValueError("mask_outline_width must be >= 1")
    if style.bbox_width < 1:
        raise ValueError("bbox_width must be >= 1")
    if style.target_label != "target" or style.hand_label != "hand":
        raise ValueError("labels must be the generic literals 'target' and 'hand'")


def _validate_image(image_rgb: np.ndarray) -> tuple[int, int]:
    if image_rgb.ndim != 3:
        raise ValueError(f"image_rgb must be HxWx3, got ndim={image_rgb.ndim}")
    if image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must have 3 channels, got shape={image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")
    height, width, _ = image_rgb.shape
    if height < 1 or width < 1:
        raise ValueError("image_rgb must have non-zero dimensions")
    return int(height), int(width)


def _validate_mask(mask: np.ndarray, *, name: str, height: int, width: int) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if mask.shape != (height, width):
        raise ValueError(
            f"{name} shape mismatch: expected {(height, width)}, got {tuple(mask.shape)}"
        )
    as_bool = np.asarray(mask).astype(bool, copy=False)
    if not np.any(as_bool):
        raise ValueError(f"{name} must be nonempty")
    return as_bool


def _outline_mask(mask: np.ndarray, width: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    # Match a centered drawing stroke: the requested width is split across the
    # inside and outside of the segmentation boundary.
    outside = (width + 1) // 2
    inside = width // 2
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=outside)
    eroded = (
        ndimage.binary_erosion(mask, structure=structure, iterations=inside, border_value=0)
        if inside
        else mask
    )
    return np.asarray(dilated & ~eroded, dtype=bool)


def _bbox_outline_mask(mask: np.ndarray, width: int) -> np.ndarray:
    ys, xs = np.where(mask)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    bbox = np.zeros_like(mask, dtype=bool)
    for offset in range(width):
        top = max(0, y0 - offset)
        bottom = min(mask.shape[0] - 1, y1 + offset)
        left = max(0, x0 - offset)
        right = min(mask.shape[1] - 1, x1 + offset)
        bbox[top, left : right + 1] = True
        bbox[bottom, left : right + 1] = True
        bbox[top : bottom + 1, left] = True
        bbox[top : bottom + 1, right] = True
    return bbox


def _paint_region(
    overlay_rgba: np.ndarray,
    region_mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> None:
    if alpha <= 0:
        return
    if not np.any(region_mask):
        return
    overlay_rgba[region_mask, 0] = color[0]
    overlay_rgba[region_mask, 1] = color[1]
    overlay_rgba[region_mask, 2] = color[2]
    overlay_rgba[region_mask, 3] = np.maximum(overlay_rgba[region_mask, 3], alpha)


def _bbox_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    y0 = int(np.min(ys))
    x0 = int(np.min(xs))
    return (x0 + 4, y0 + 4)


def render_marked_frame(
    *,
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    hand_mask: np.ndarray,
    style: MarkerStyle = DISCOVERY_STYLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Render deterministic target/hand markers and return intended-pixel mask."""

    _validate_style(style)
    height, width = _validate_image(image_rgb)
    target_mask_bool = _validate_mask(target_mask, name="target_mask", height=height, width=width)
    hand_mask_bool = _validate_mask(hand_mask, name="hand_mask", height=height, width=width)

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for role_mask, role_color in (
        (target_mask_bool, style.target_rgb),
        (hand_mask_bool, style.hand_rgb),
    ):
        if style.fill_alpha > 0:
            _paint_region(overlay, role_mask, role_color, style.fill_alpha)
        _paint_region(
            overlay,
            _outline_mask(role_mask, style.mask_outline_width),
            role_color,
            255,
        )
        _paint_region(
            overlay,
            _bbox_outline_mask(role_mask, style.bbox_width),
            role_color,
            255,
        )

    overlay_image = Image.fromarray(overlay, mode="RGBA")
    draw = ImageDraw.Draw(overlay_image)
    draw.text(_bbox_anchor(target_mask_bool), style.target_label, fill=(*style.target_rgb, 255))
    draw.text(_bbox_anchor(hand_mask_bool), style.hand_label, fill=(*style.hand_rgb, 255))
    overlay_arr = np.asarray(overlay_image, dtype=np.uint8)
    intended_pixel_mask = overlay_arr[:, :, 3] > 0

    base = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    rendered = Image.alpha_composite(base, overlay_image).convert("RGB")
    rendered_rgb = np.asarray(rendered, dtype=np.uint8)

    if rendered_rgb.shape != image_rgb.shape:
        raise AssertionError("rendered image shape changed unexpectedly")
    if intended_pixel_mask.shape != image_rgb.shape[:2]:
        raise AssertionError("intended mask shape mismatch after rendering")
    return rendered_rgb, intended_pixel_mask
