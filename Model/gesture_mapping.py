"""
Utilities for mapping thumb/index geometry to an angle, letter, and smoothing level.
"""

from __future__ import annotations

import math
from typing import Tuple

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SECTOR_DEGREES = 360.0 / len(LETTERS)


def angle_from_points(thumb: Tuple[int, int], index: Tuple[int, int]) -> float:
    """
    Compute angle in degrees from thumb tip to index tip.
    0 degrees points right; angles increase counterclockwise.
    """
    dx = index[0] - thumb[0]
    dy = index[1] - thumb[1]
    angle_rad = math.atan2(-dy, dx)
    return (math.degrees(angle_rad) + 360.0) % 360.0


def distance_from_points(thumb: Tuple[int, int], index: Tuple[int, int]) -> float:
    """Euclidean distance between thumb and index in pixels."""
    return math.hypot(index[0] - thumb[0], index[1] - thumb[1])


def letter_from_angle(angle_deg: float) -> str:
    """Map a 0–360° angle into one of 26 angular sectors labeled A–Z."""
    idx = int(angle_deg // SECTOR_DEGREES)
    return LETTERS[min(idx, len(LETTERS) - 1)]


def smooth_level_from_distance(dist: float, frame_width: int) -> float:
    """
    Map thumb-index distance to smoothing factor in [0, 1].
    Uses roughly half the frame width as the maximum distance.
    """
    if frame_width <= 0:
        return 0.0
    max_dist = 0.5 * float(frame_width)
    if max_dist <= 0.0:
        return 0.0
    level = dist / max_dist
    if level < 0.0:
        return 0.0
    if level > 1.0:
        return 1.0
    return level
