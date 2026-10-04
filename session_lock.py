"""
session_lock.py
---------------
Single-owner gesture session lock for PLM use.

THE PROBLEM
    MediaPipe knows hands, not people. In the original app the "active" hand
    was simply the highest-CONFIDENCE hand -- that is detector certainty, not
    ownership, so a bystander can hijack the session.

WHAT THIS DOES (and does not do)
    Ownership is established by a deliberate engagement gesture and then kept
    by CONTINUITY TRACKING (wrist position + palm size + handedness). Anything
    that is not the owner's hand is ignored, and the session auto-locks when
    the owner disappears or stops interacting.

    This is NOT authentication. It cannot tell *who* the owner is, and a
    bystander who physically overlaps the owner's hand can still swap the
    track. For real identity, unlock with a badge/PIN/SSO out-of-band and use
    this module only to prevent accidental triggers.

STATE MACHINE
    LOCKED --(3-finger pose held, in zone, close enough)--> ARMING
    ARMING --(held ENGAGE_HOLD_SEC, hand continuous)------> ENGAGED
    ENGAGED --(owner lost > OWNER_LOST_SEC | idle timeout | pose held again)--> LOCKED

Duck-typed on purpose: a "hand" only needs .landmarks_norm (21,2),
.palm_size and .handedness; "fingers" only .index/.middle/.ring/.pinky.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

import config

WRIST = 0


class LockState(str, Enum):
    LOCKED = "LOCKED"
    ARMING = "ARMING"
    ENGAGED = "ENGAGED"


@dataclass
class LockDecision:
    state: LockState
    owner: Optional[Any]   # the owner's hand object; None unless actionable this frame
    progress: float        # 0..1 hold progress (engage or disengage)
    message: str


def is_engage_pose(fingers: Any) -> bool:
    """Index+middle+ring extended, pinky folded. Thumb ignored (noisy)."""
    return bool(fingers.index and fingers.middle and fingers.ring and not fingers.pinky)


Candidate = Tuple[Any, Any]  # (hand, fingers)


def _bbox(hand: Any, margin_palms: float) -> Tuple[float, float, float, float]:
    xs, ys = hand.landmarks_norm[:, 0], hand.landmarks_norm[:, 1]
    m = margin_palms * hand.palm_size
    return float(xs.min() - m), float(ys.min() - m), float(xs.max() + m), float(ys.max() + m)


def hands_overlap(a: Any, b: Any) -> bool:
    """True if the (margin-expanded) bounding boxes of two hands intersect."""
    ax0, ay0, ax1, ay1 = _bbox(a, config.LOCK_OVERLAP_MARGIN)
    bx0, by0, bx1, by1 = _bbox(b, config.LOCK_OVERLAP_MARGIN)
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


class OwnerLock:
    def __init__(self) -> None:
        self.state = LockState.LOCKED
        self._hold_since: Optional[float] = None
        self._released_since_engage = False
        self._need_release = False   # after a lock, the pose must be dropped once
        self._last_activity = 0.0
        self._clear_owner()

    # ------------------------------------------------------------ public API
    def update(self, candidates: Sequence[Candidate], now: Optional[float] = None) -> LockDecision:
        now = time.perf_counter() if now is None else now
        if self.state is LockState.ENGAGED:
            return self._update_engaged(candidates, now)
        return self._update_locked(candidates, now)

    def note_activity(self, now: Optional[float] = None) -> None:
        """Call whenever the owner does something real (resets idle timeout)."""
        self._last_activity = time.perf_counter() if now is None else now

    def force_lock(self, reason: str = "Locked manually") -> LockDecision:
        return self._lock(reason)

    # ------------------------------------------------------- LOCKED / ARMING
    def _update_locked(self, cands: Sequence[Candidate], now: float) -> LockDecision:
        posing = [
            (h, f) for h, f in cands
            if is_engage_pose(f) and self._in_zone(h) and h.palm_size >= config.LOCK_MIN_ENGAGE_PALM
        ]
        # Hands that overlap another hand are ambiguous: never arm on them.
        posing = [(h, f) for h, f in posing if not any(o is not h and hands_overlap(h, o) for o, _ in cands)]
        if not posing:
            self._hold_since = None
            self._need_release = False
            self.state = LockState.LOCKED
            self._clear_owner()
            return LockDecision(LockState.LOCKED, None, 0.0, "LOCKED - hold 3 fingers up to unlock")
        if self._need_release:
            return LockDecision(LockState.LOCKED, None, 0.0, "LOCKED - lower your hand, then raise 3 fingers to unlock")

        # Largest palm = closest to the camera = most plausibly the operator.
        hand, _ = max(posing, key=lambda c: c[0].palm_size)

        if self.state is LockState.ARMING and self._pos is not None:
            matched = self._match([c for c in posing], now)
            if matched is None:            # a different hand took over mid-arming: restart
                self._hold_since = None
                self._clear_owner()
            else:
                hand = matched[0]

        if self._hold_since is None:
            self._hold_since = now
            self._set_signature(hand, now)
            self.state = LockState.ARMING
        else:
            self._update_signature(hand, now)

        progress = min(1.0, (now - self._hold_since) / config.LOCK_ENGAGE_HOLD_SEC)
        if progress >= 1.0:
            self.state = LockState.ENGAGED
            self._hold_since = None
            self._released_since_engage = False
            self._last_activity = now
            return LockDecision(LockState.ENGAGED, hand, 0.0, "UNLOCKED")
        return LockDecision(LockState.ARMING, None, progress, "Keep holding...")

    # --------------------------------------------------------------- ENGAGED
    def _update_engaged(self, cands: Sequence[Candidate], now: float) -> LockDecision:
        matched = self._match(cands, now)
        if matched is None:
            if now - self._last_seen > config.LOCK_OWNER_LOST_SEC:
                return self._lock("Owner lost")
            return LockDecision(LockState.ENGAGED, None, 0.0, "Tracking owner... (input paused)")

        hand, fingers = matched
        if now - self._last_activity > config.LOCK_IDLE_TIMEOUT_SEC:
            return self._lock("Idle timeout")

        # Another hand is touching/near the owner's: we can no longer be sure
        # which track is whose. Freeze input and DON'T update the signature,
        # so the owner track cannot drift onto the other hand. Resumes by
        # continuity once the hands separate; idle timeout still applies.
        if any(o is not hand and hands_overlap(hand, o) for o, _ in cands):
            self._hold_since = None
            return LockDecision(LockState.ENGAGED, None, 0.0, "Paused: hands overlapping")
        self._update_signature(hand, now)

        # Re-lock gesture. Must see the pose released once after unlocking,
        # otherwise the unlock hold would immediately start the lock hold.
        progress = 0.0
        if is_engage_pose(fingers):
            if self._released_since_engage:
                if self._hold_since is None:
                    self._hold_since = now
                progress = min(1.0, (now - self._hold_since) / config.LOCK_DISENGAGE_HOLD_SEC)
                if progress >= 1.0:
                    return self._lock("Locked by gesture")
        else:
            self._released_since_engage = True
            self._hold_since = None

        return LockDecision(LockState.ENGAGED, hand, progress, "")

    # -------------------------------------------------------------- matching
    def _match(self, cands: Sequence[Candidate], now: float) -> Optional[Candidate]:
        """Pick the candidate that is plausibly the SAME hand as the owner."""
        if self._pos is None:
            return None
        dt = max(0.0, now - self._last_seen)
        radius = min(config.LOCK_MATCH_MAX_RADIUS,
                     config.LOCK_MATCH_BASE_RADIUS + config.LOCK_MATCH_SPEED * dt)
        lo, hi = config.LOCK_PALM_RATIO_RANGE
        best: Optional[Candidate] = None
        best_score = float("inf")
        for hand, fingers in cands:
            wx, wy = hand.landmarks_norm[WRIST]
            dist = math.hypot(wx - self._pos[0], wy - self._pos[1])
            ratio = hand.palm_size / self._palm
            if dist > radius or not (lo <= ratio <= hi):
                continue
            score = dist + (config.LOCK_HANDEDNESS_PENALTY if hand.handedness != self._label else 0.0)
            if score < best_score:
                best, best_score = (hand, fingers), score
        return best

    # ------------------------------------------------------------ signature
    def _set_signature(self, hand: Any, now: float) -> None:
        self._pos = (float(hand.landmarks_norm[WRIST][0]), float(hand.landmarks_norm[WRIST][1]))
        self._palm = float(hand.palm_size)
        self._label = hand.handedness
        self._last_seen = now

    def _update_signature(self, hand: Any, now: float) -> None:
        a = 0.5
        wx, wy = float(hand.landmarks_norm[WRIST][0]), float(hand.landmarks_norm[WRIST][1])
        self._pos = (a * wx + (1 - a) * self._pos[0], a * wy + (1 - a) * self._pos[1])
        self._palm = a * float(hand.palm_size) + (1 - a) * self._palm
        self._label = hand.handedness
        self._last_seen = now

    def _clear_owner(self) -> None:
        self._pos: Optional[Tuple[float, float]] = None
        self._palm: float = 1.0
        self._label: str = ""
        self._last_seen: float = 0.0

    def _lock(self, reason: str) -> LockDecision:
        self.state = LockState.LOCKED
        self._hold_since = None
        self._need_release = True
        self._clear_owner()
        return LockDecision(LockState.LOCKED, None, 0.0, reason)

    @staticmethod
    def _in_zone(hand: Any) -> bool:
        x0, y0, x1, y1 = config.LOCK_ENGAGE_ZONE
        wx, wy = hand.landmarks_norm[WRIST]
        return x0 <= wx <= x1 and y0 <= wy <= y1
