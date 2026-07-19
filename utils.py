"""
utils.py
--------
Small, dependency-light helper utilities shared across modules:
- FPS measurement
- Vector / angle math used for scale- and rotation-invariant gesture logic
- Exponential moving average point smoother
- Generic clamp helper
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class FPSCounter:
    """Rolling-average FPS counter based on a sliding time window."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vector(a: Tuple[float, float], b: Tuple[float, float]) -> np.ndarray:
    """Vector from point a to point b."""
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors. Returns 180.0 if degenerate."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


class PointSmoother:
    """Exponential moving average smoother for a 2D point stream.

    Resets cleanly when a new stroke starts so old strokes never bleed
    into new ones.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = clamp(alpha, 0.01, 1.0)
        self._prev: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, point: Tuple[float, float]) -> Tuple[float, float]:
        if self._prev is None:
            self._prev = point
            return point
        sx = self.alpha * point[0] + (1 - self.alpha) * self._prev[0]
        sy = self.alpha * point[1] + (1 - self.alpha) * self._prev[1]
        self._prev = (sx, sy)
        return self._prev
