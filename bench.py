"""
bench.py -- per-frame cost of the NEW code paths. Run: python bench.py

LIMITS (read before quoting numbers):
  * MediaPipe is timed on a synthetic frame with NO hand in it. That exercises
    only the palm detector, so it is a LOWER BOUND; a real hand adds the
    landmark model. The number that matters is end-to-end FPS on YOUR machine
    with a real camera (shown live in the main.py HUD).
  * Numbers depend on this CPU. Do not compare across machines.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "tests")
from synth import make_hand
from gesture_detector import compute_finger_state
from session_lock import OwnerLock
from shopfloor import ShopfloorView

def timeit(fn, n):
    fn(); t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000.0

if __name__ == "__main__":
    owner = make_hand(.4, .7, ("index", "middle", "ring")); by = make_hand(.8, .7, ("index",), label="Left")
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    lock = OwnerLock(); t = [0.0]
    def lock_step():
        t[0] += 0.033; lock.update([(owner, compute_finger_state(owner)), (by, compute_finger_state(by))], t[0])
    v = ShopfloorView(1280, 720)
    rows = [("finger state, 1 hand", timeit(lambda: compute_finger_state(owner), 2000)),
            ("lock update + 2x finger state", timeit(lock_step, 2000)),
            ("shopfloor update", timeit(lambda: v.update(owner, 0.0), 2000)),
            ("shopfloor draw 1280x720", timeit(lambda: v.draw(frame, locked=False), 100))]
    try:
        from hand_tracker import HandTracker
        if os.path.isfile("models/hand_landmarker.task"):
            tr = HandTracker(); rows.append(("MediaPipe, empty frame (LOWER BOUND)", timeit(lambda: tr.process(frame), 60))); tr.close()
    except Exception as exc:  # noqa: BLE001
        rows.append((f"MediaPipe skipped: {exc}"[:60], float("nan")))
    for name, ms in rows: print(f"{name:<42s}{ms:8.3f} ms")
    new = sum(ms for n, ms in rows if n.startswith(("lock", "shopfloor")))
    print(f"\nNew code per frame (lock + update + draw): {new:.2f} ms  ->  caps at ~{1000/new:.0f} FPS if it were the only cost")
