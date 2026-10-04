"""Headless tests for session_lock + shopfloor using synthetic hands (no camera)."""
import os, sys
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from session_lock import OwnerLock, LockState
from shopfloor import ShopfloorView

POSE = NS(index=True, middle=True, ring=True, pinky=False)   # engage pose
OPEN = NS(index=True, middle=True, ring=True, pinky=True)
FIST = NS(index=False, middle=False, ring=False, pinky=False)

def hand(x, y, palm=0.12, label="Right", w=1280, h=720):
    n = np.tile([x, y], (21, 1)).astype(float)
    n[9] = [x, y - palm]                 # middle MCP above wrist
    px = n * [w, h]
    return NS(landmarks_norm=n, landmarks_px=px, palm_size=palm, handedness=label)

def run(lock, cands, t0, dur, step=0.033):
    t, d = t0, None
    while t < t0 + dur:
        d = lock.update(cands, t); t += step
    return d, t

def test_unlock_needs_hold():
    l = OwnerLock(); me = hand(.5, .6)
    d, t = run(l, [(me, POSE)], 0, 1.0); assert d.state == LockState.ARMING
    d, t = run(l, [(me, POSE)], t, 0.7);  assert d.state == LockState.ENGAGED and d.owner is me

def test_brief_pose_does_not_unlock():
    l = OwnerLock(); me = hand(.5, .6)
    run(l, [(me, POSE)], 0, 0.8); d, _ = run(l, [(me, FIST)], 0.8, 0.5)
    assert d.state == LockState.LOCKED

def test_far_bystander_cannot_engage():
    l = OwnerLock(); far = hand(.8, .5, palm=0.04)
    d, _ = run(l, [(far, POSE)], 0, 3.0); assert d.state == LockState.LOCKED

def test_bystander_cannot_hijack_while_engaged():
    l = OwnerLock(); me = hand(.3, .6); _, t = run(l, [(me, POSE)], 0, 1.6)
    by = hand(.75, .5, palm=0.12, label="Left")           # same size, different place
    for _ in range(60):
        me = hand(.3, .6)
        d = l.update([(by, OPEN), (me, OPEN)], t); t += 0.033
        assert d.owner is me, "bystander hijacked the session"

def test_bystander_beside_owner_when_owner_lost():
    l = OwnerLock(); me = hand(.3, .6); _, t = run(l, [(me, POSE)], 0, 1.6)
    by = hand(.75, .5)
    d, t = run(l, [(by, OPEN)], t, 0.5); assert d.owner is None          # paused, not hijacked
    d, t = run(l, [(by, OPEN)], t, 1.5); assert d.state == LockState.LOCKED  # then locks

def test_idle_timeout():
    l = OwnerLock(); me = hand(.5, .6); _, t = run(l, [(me, POSE)], 0, 1.6)
    d, _ = run(l, [(me, OPEN)], t, config.LOCK_IDLE_TIMEOUT_SEC + 1); assert d.state == LockState.LOCKED

def test_activity_prevents_idle_timeout():
    l = OwnerLock(); me = hand(.5, .6); _, t = run(l, [(me, POSE)], 0, 1.6)
    end = t + config.LOCK_IDLE_TIMEOUT_SEC + 5
    while t < end:
        l.note_activity(t); d = l.update([(me, OPEN)], t); t += 0.1
    assert d.state == LockState.ENGAGED

def test_relock_gesture_requires_release_first():
    l = OwnerLock(); me = hand(.5, .6); _, t = run(l, [(me, POSE)], 0, 1.6)
    d, t = run(l, [(me, POSE)], t, 3.0)                  # still holding the unlock pose
    assert d.state == LockState.ENGAGED, "unlock hold must not instantly relock"
    _, t = run(l, [(me, OPEN)], t, 0.3)
    d, t = run(l, [(me, POSE)], t, 2.0); assert d.state == LockState.LOCKED

def test_shopfloor_dwell_select_and_pinch_drag():
    v = ShopfloorView(1280, 720); s = v.stations[0]
    cx, cy = v.to_px(s.x, s.y)
    hnd = hand(.5, .5); hnd.landmarks_px[8] = [cx, cy]; hnd.landmarks_px[4] = [cx + 200, cy]  # not pinching
    hnd.landmarks_px[0] = [cx, cy + 200]; hnd.landmarks_px[9] = [cx, cy + 100]
    t = 0.0
    while t < 1.0: v.update(hnd, t); t += 0.033
    assert v.selected is s
    hnd.landmarks_px[4] = [cx + 5, cy]                    # pinch on station
    v.update(hnd, t); t += 0.033
    hnd.landmarks_px[8] = [cx + 150, cy]; hnd.landmarks_px[4] = [cx + 155, cy]
    v.update(hnd, t)
    assert s.x > s.home[0] + 1 and v.whatif_active
    v.update(None, t + 0.1); assert not v._pinching and v.cursor is None  # lock cancels everything
    v.reset_whatif(); assert not v.whatif_active

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try: fn(); print("PASS", name)
            except AssertionError as e: fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
