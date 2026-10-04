"""
plm_main.py
-----------
PLM prototype entry point:  python plm_main.py

Camera -> HandTracker -> OwnerLock (who may act?) -> ShopfloorView (what happens?)

Gesture-only: hover the panel buttons (LAYER / RESET / REFRESH / LOCK) for 0.7 s.
Keyboard shortcuts also work:  L cycle layer | R reset what-if | X lock now | Q quit

Options:  python plm_main.py [--csv stations.csv]   (default: MOCK data)
Unlock: hold index+middle+ring up (pinky folded) for ~1.5 s inside the frame.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

import config
from audit import AuditLog
from gesture_detector import compute_finger_state
from hand_tracker import HandTracker, ModelNotFoundError
from security import CameraSession, CameraUnavailableError, setup_logging
from plm_adapter import AdapterError, CsvAdapter, MockAdapter
from session_lock import LockState, OwnerLock
from shopfloor import ShopfloorView

WIN = "PLM Gesture Shopfloor (prototype)"
OWNER_COLOR, OTHER_COLOR = (0, 220, 0), (140, 140, 140)


def draw_hand(frame, hand, is_owner: bool) -> None:
    if is_owner:
        HandTracker.draw_skeleton(frame, hand.landmarks_px)
    else:  # visibly "not you": grey dots only, no skeleton
        for x, y in hand.landmarks_px:
            cv2.circle(frame, (int(x), int(y)), 3, OTHER_COLOR, -1, cv2.LINE_AA)


def draw_lock_hud(frame, decision, hand_count: int) -> None:
    h, w = frame.shape[:2]
    color = {LockState.LOCKED: (0, 0, 220), LockState.ARMING: (0, 180, 255), LockState.ENGAGED: OWNER_COLOR}[decision.state]
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    msg = decision.message or ("Owner active - hold 3 fingers to lock" if decision.state is LockState.ENGAGED else "")
    cv2.putText(frame, f"{decision.state.value}  |  hands:{hand_count}  |  {msg}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    if decision.progress > 0:  # hold-progress bar
        cv2.rectangle(frame, (0, 36), (int(w * decision.progress), 42), color, -1)


def run(csv_path: str = "") -> int:
    logger = setup_logging()
    lock = OwnerLock()
    view = None
    audit = AuditLog(config.AUDIT_DIR)
    audit.record("session_start", source=csv_path or "MOCK")
    logger.info("Audit log: %s", audit.path)
    adapter = CsvAdapter(csv_path) if csv_path else MockAdapter()
    try:
        with HandTracker(logger) as tracker, CameraSession(
            config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, logger
        ) as camera:
            last_state = LockState.LOCKED
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    logger.error("Camera unavailable; stopping.")
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if view is None:
                    view = ShopfloorView(w, h, adapter)

                now = time.perf_counter()
                hands = tracker.process(frame)
                cands = [(hd, compute_finger_state(hd)) for hd in hands]
                decision = lock.update(cands, now)
                owner = decision.owner

                if decision.state is not last_state:  # state changes only; no coordinates
                    logger.info("Lock state: %s -> %s (%s)", last_state.value, decision.state.value, decision.message)
                    audit.record("lock_state", frm=last_state.value, to=decision.state.value, reason=decision.message)
                    last_state = decision.state
                    if decision.state is LockState.LOCKED:
                        view.update(None, now)

                view.update(owner, now)
                if owner is not None and view.interacting:
                    lock.note_activity(now)
                for name, data in view.pop_events():
                    if name == "lock_request":
                        lock.force_lock("Locked by button")
                    else:
                        audit.record(name, **data)

                out = view.draw(frame, locked=decision.state is not LockState.ENGAGED)
                for hd in hands:
                    draw_hand(out, hd, is_owner=hd is owner)
                draw_lock_hud(out, decision, len(hands))
                cv2.imshow(WIN, out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("l"):
                    view.cycle_layer()
                    audit.record("layer", layer=view.layer, via="key")
                elif key == ord("r"):
                    view.reset_whatif()
                    audit.record("reset", via="key")
                elif key == ord("x"):
                    lock.force_lock("Locked manually")
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except (ModelNotFoundError, CameraUnavailableError, AdapterError) as exc:
        logger.error("Fatal: %s", exc)
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        audit.record("session_end")
        audit.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PLM gesture shopfloor prototype")
    ap.add_argument("--csv", default="", help="station CSV (see sample_stations.csv); default MOCK data")
    sys.exit(run(ap.parse_args().csv))
