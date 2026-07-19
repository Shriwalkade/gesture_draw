"""
gesture_detector.py
--------------------
Turns raw hand landmarks into a stable, high-level gesture:

    DRAW  -> index extended, middle/ring/pinky folded
    ERASE -> all five fingers extended (open palm)
    IDLE  -> anything else (including no hand / low confidence)

Design goals (see project brief):
- Robust across left/right hand, hand size, and rotation: every finger
  "extended/folded" decision uses angles and distances normalized by the
  hand's own palm size, never absolute pixel thresholds.
- Temporally stable: a raw per-frame classification is first smoothed by a
  short majority-vote history, then must persist for GESTURE_STABLE_MS
  before the application actually switches modes. This kills flicker from
  single noisy frames without adding much perceptible latency.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

import config
from hand_tracker import (
    HandResult,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    THUMB_CMC, THUMB_MCP, THUMB_TIP,
    WRIST,
)
from utils import angle_between, clamp, euclidean, vector


class Gesture(str, Enum):
    DRAW = "DRAW"
    ERASE = "ERASE"
    IDLE = "IDLE"
    NONE = "NONE"  # no hand detected at all


@dataclass
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


def _finger_extended(
    norm: np.ndarray, mcp: int, pip: int, tip: int, palm_size: float
) -> bool:
    """Angle- and distance-based extended/folded test for one finger.

    A finger is "extended" when:
      1. The MCP->PIP and PIP->TIP bone vectors are nearly colinear
         (small bend angle), AND
      2. The tip sits meaningfully farther from the wrist than the PIP
         joint does, scaled by palm size (so it works for any hand size
         or distance from the camera).
    Both checks are rotation-invariant because they only use vectors
    between the hand's own landmarks, never absolute image axes.
    """
    v1 = vector(norm[mcp], norm[pip])
    v2 = vector(norm[pip], norm[tip])
    bend_angle = angle_between(v1, v2)

    wrist = norm[WRIST]
    tip_dist = euclidean(wrist, norm[tip])
    pip_dist = euclidean(wrist, norm[pip])
    extension_ratio = (tip_dist - pip_dist) / palm_size

    return bend_angle < config.FINGER_STRAIGHT_ANGLE_DEG and extension_ratio > config.FINGER_EXTENSION_MARGIN


def _thumb_extended(norm: np.ndarray, palm_size: float) -> bool:
    """Thumb needs its own test: it bends mainly at the CMC/MCP joint and
    moves largely sideways rather than "up", so the same vertical-distance
    logic used for other fingers doesn't apply. Instead we measure the
    angle at the MCP joint and how far the tip sits from the palm center
    relative to palm size.
    """
    v1 = vector(norm[THUMB_CMC], norm[THUMB_MCP])
    v2 = vector(norm[THUMB_MCP], norm[THUMB_TIP])
    bend_angle = angle_between(v1, v2)

    palm_center = (norm[WRIST] + norm[MIDDLE_MCP]) / 2.0
    tip_dist = euclidean(palm_center, norm[THUMB_TIP])
    mcp_dist = euclidean(palm_center, norm[THUMB_MCP])
    extension_ratio = (tip_dist - mcp_dist) / palm_size

    return bend_angle < config.THUMB_STRAIGHT_ANGLE_DEG and extension_ratio > config.FINGER_EXTENSION_MARGIN


def compute_finger_state(hand: HandResult) -> FingerState:
    norm = hand.landmarks_norm
    palm = hand.palm_size
    return FingerState(
        thumb=_thumb_extended(norm, palm),
        index=_finger_extended(norm, INDEX_MCP, INDEX_PIP, INDEX_TIP, palm),
        middle=_finger_extended(norm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, palm),
        ring=_finger_extended(norm, RING_MCP, RING_PIP, RING_TIP, palm),
        pinky=_finger_extended(norm, PINKY_MCP, PINKY_PIP, PINKY_TIP, palm),
    )


def classify_raw_gesture(fingers: FingerState) -> Gesture:
    """Single-frame classification from finger states. No temporal logic here."""
    if fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky:
        return Gesture.DRAW
    if fingers.index and fingers.middle and fingers.ring and fingers.pinky:
        return Gesture.ERASE
    return Gesture.IDLE


class GestureStabilizer:
    """Two-stage temporal filter: majority-vote smoothing + minimum hold time.

    1. Each raw per-frame gesture is pushed into a short rolling history;
       the majority vote of that history is the "smoothed" gesture. This
       absorbs single-frame misclassifications from noisy landmarks.
    2. The smoothed gesture must then remain unchanged for at least
       GESTURE_STABLE_MS before it becomes the "active" gesture the rest
       of the app reacts to. This is what actually prevents accidental
       mode switching (e.g. a half-second transition while curling a
       finger doesn't spuriously trigger erase-then-draw).
    """

    def __init__(self) -> None:
        self._history: Deque[Gesture] = deque(maxlen=config.GESTURE_HISTORY_LEN)
        self._pending_gesture: Gesture = Gesture.NONE
        self._pending_since: float = time.perf_counter()
        self._active_gesture: Gesture = Gesture.NONE

    def reset(self) -> None:
        self._history.clear()
        self._pending_gesture = Gesture.NONE
        self._active_gesture = Gesture.NONE
        self._pending_since = time.perf_counter()

    def update(self, raw_gesture: Gesture) -> Gesture:
        now = time.perf_counter()
        self._history.append(raw_gesture)

        counts = Counter(self._history)
        smoothed_gesture = counts.most_common(1)[0][0]

        if smoothed_gesture != self._pending_gesture:
            self._pending_gesture = smoothed_gesture
            self._pending_since = now

        held_ms = (now - self._pending_since) * 1000.0
        if held_ms >= config.GESTURE_STABLE_MS:
            self._active_gesture = self._pending_gesture

        return self._active_gesture

    @property
    def active_gesture(self) -> Gesture:
        return self._active_gesture


class GestureDetector:
    """Top-level entry point: HandResult -> stabilized Gesture + fingertip point."""

    def __init__(self) -> None:
        self._stabilizer = GestureStabilizer()

    def reset(self) -> None:
        self._stabilizer.reset()

    def process(self, hand: Optional[HandResult]) -> Tuple[Gesture, Optional[FingerState], Optional[Tuple[float, float]]]:
        """Returns (active_gesture, finger_state_or_None, index_tip_px_or_None)."""
        if hand is None:
            active = self._stabilizer.update(Gesture.NONE)
            return active, None, None

        fingers = compute_finger_state(hand)
        raw = classify_raw_gesture(fingers)
        active = self._stabilizer.update(raw)
        index_tip_px = (
            float(hand.landmarks_px[INDEX_TIP][0]),
            float(hand.landmarks_px[INDEX_TIP][1]),
        )
        return active, fingers, index_tip_px
