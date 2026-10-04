"""Tests for: real-classifier poses, overlap freeze, dwell buttons, CSV adapter, audit log,
and an end-to-end run of main.py with a scripted camera."""
import json, os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from synth import make_hand, set_index_tip_px, wiggle_index_tip
from gesture_detector import compute_finger_state, classify_raw_gesture, Gesture
from session_lock import OwnerLock, LockState, is_engage_pose, hands_overlap
from shopfloor import ShopfloorView
from plm_adapter import CsvAdapter, AdapterError, MockAdapter
from audit import AuditLog

ENG = ("index", "middle", "ring")


def run(lock, cands, t0, dur, step=0.033):
    t, d = t0, None
    while t < t0 + dur:
        d = lock.update(cands, t); t += step
    return d, t


# ---- real classifier ---------------------------------------------------------
def test_real_classifier_poses():
    cases = {("index",): Gesture.DRAW, ("index", "middle"): Gesture.ERASE,
             ("index", "middle", "ring", "pinky"): Gesture.IDLE, (): Gesture.IDLE, ENG: Gesture.IDLE}
    for up, want in cases.items():
        f = compute_finger_state(make_hand(.5, .7, up))
        assert classify_raw_gesture(f) == want, (up, f, classify_raw_gesture(f))
    assert is_engage_pose(compute_finger_state(make_hand(.5, .7, ENG)))
    assert not is_engage_pose(compute_finger_state(make_hand(.5, .7, ("index", "middle", "ring", "pinky"))))

def test_rotated_scaled_hand_still_classified():
    for s in (0.6, 1.0, 1.5):
        f = compute_finger_state(make_hand(.5, .7, ENG, scale=s))
        assert is_engage_pose(f), s

def test_open_palm_erase_option():
    config.ERASE_GESTURE = "OPEN_PALM"
    try:
        f = compute_finger_state(make_hand(.5, .7, ("index", "middle", "ring", "pinky")))
        assert classify_raw_gesture(f) == Gesture.ERASE
        f = compute_finger_state(make_hand(.5, .7, ("index", "middle")))
        assert classify_raw_gesture(f) == Gesture.IDLE
    finally:
        config.ERASE_GESTURE = "TWO_FINGERS"


# ---- lock with REAL finger states + overlap -------------------------------------
def cand(h): return (h, compute_finger_state(h))

def test_lock_with_real_classifier():
    l = OwnerLock(); me = make_hand(.5, .7, ENG)
    d, t = run(l, [cand(me)], 0, 1.8)
    assert d.state == LockState.ENGAGED and d.owner is me

def test_overlap_detection():
    a = make_hand(.40, .70); far = make_hand(.75, .70); near = make_hand(.47, .70, label="Left")
    assert not hands_overlap(a, far) and hands_overlap(a, near)

def test_overlap_freezes_then_resumes_without_hijack():
    l = OwnerLock(); me = make_hand(.40, .70, ENG); _, t = run(l, [cand(me)], 0, 1.8)
    open_me = make_hand(.40, .70, ("index", "middle", "ring", "pinky"))
    d, t = run(l, [cand(open_me)], t, 0.3); assert d.owner is open_me
    by = make_hand(.47, .70, ("index", "middle", "ring", "pinky"), label="Left")      # crosses owner's hand
    d, t = run(l, [cand(open_me), cand(by)], t, 0.5)
    assert d.state == LockState.ENGAGED and d.owner is None and "overlapping" in d.message
    by_far = make_hand(.80, .70, ("index", "middle", "ring", "pinky"), label="Left")  # separates
    d, t = run(l, [cand(open_me), cand(by_far)], t, 0.3)
    assert d.owner is open_me, "owner should resume after separation, not the bystander"

def test_cannot_arm_when_hands_overlap():
    l = OwnerLock(); a = make_hand(.40, .70, ENG); b = make_hand(.46, .70, ENG, label="Left")
    d, _ = run(l, [cand(a), cand(b)], 0, 3.0)
    assert d.state == LockState.LOCKED


# ---- shopfloor buttons / refresh / events ----------------------------------------
def _pointing(v, x, y):
    h = make_hand(.5, .9, ("index",)); set_index_tip_px(h, x, y); return h

def test_button_dwell_cycles_layer_once_and_emits():
    v = ShopfloorView(1280, 720); key, x0, y0, x1, y1 = v.buttons[0]; assert key == "LAYER"
    h = _pointing(v, (x0 + x1) / 2, (y0 + y1) / 2); t = 0.0
    while t < 3.0: v.update(h, t); t += 0.033          # hover 3 s: must fire exactly once
    assert v.layer == "REVISION", v.layer
    h2 = _pointing(v, 100, 400); v.update(h2, t); t += 0.033   # leave (empty floor), come back
    h = _pointing(v, (x0 + x1) / 2, (y0 + y1) / 2); t0 = t
    while t < t0 + 1.0: v.update(h, t); t += 0.033
    assert v.layer == "WIP"
    assert [e[0] for e in v.pop_events()].count("layer") == 2

def test_lock_button_emits_request():
    v = ShopfloorView(1280, 720); key, x0, y0, x1, y1 = v.buttons[3]; assert key == "LOCK"
    h = _pointing(v, (x0 + x1) / 2, (y0 + y1) / 2); t = 0.0
    while t < 1.0: v.update(h, t); t += 0.033
    assert "lock_request" in [e[0] for e in v.pop_events()]

def test_pinch_does_not_trigger_button():
    v = ShopfloorView(1280, 720); key, x0, y0, x1, y1 = v.buttons[0]
    h = _pointing(v, (x0 + x1) / 2, (y0 + y1) / 2); h.landmarks_px[4] = h.landmarks_px[8] + [3, 0]; t = 0.0
    while t < 2.0: v.update(h, t); t += 0.033
    assert v.layer == "STATUS"

def test_whatif_move_event_on_release():
    v = ShopfloorView(1280, 720); s = v.stations[0]; cx, cy = v.to_px(s.x, s.y)
    h = _pointing(v, cx, cy); h.landmarks_px[4] = [cx + 4, cy]; v.update(h, 0)
    h.landmarks_px[8] = [cx + 160, cy]; h.landmarks_px[4] = [cx + 164, cy]; v.update(h, .1)
    v.update(None, .2)     # lock cancels mid-grab -> move still audited
    ev = [e for e in v.pop_events() if e[0] == "whatif_move"]
    assert ev and ev[0][1]["station"] == s.sid

def test_csv_adapter_valid_and_errors():
    good = CsvAdapter(os.path.join(os.path.dirname(__file__), "..", "sample_stations.csv")).load_stations()
    assert len(good) == 6 and good[3].rev_mismatch
    d = tempfile.mkdtemp()
    def write(name, text):
        p = os.path.join(d, name); open(p, "w").write(text); return p
    hdr = ",".join(["station_id","name","x_m","y_m","w_m","h_m","status","work_order","part_no","rev_running","rev_released","bom_items","wip"])
    for text, frag in [("station_id,name\nA,b\n", "missing columns"),
                       (hdr + "\nA,n,1,1,1,1,BROKEN,w,p,A,A,1,1\n", "status"),
                       (hdr + "\nA,n,99,1,1,1,IDLE,w,p,A,A,1,1\n", "outside floor"),
                       (hdr + "\nA,n,1,1,1,1,IDLE,w,p,A,A,x,1\n", "row 2"),
                       (hdr + "\nA,n,1,1,1,1,IDLE,w,p,A,A,1,1\nA,n,2,2,1,1,IDLE,w,p,A,A,1,1\n", "duplicate"),
                       (hdr + "\n", "no station rows")]:
        try: CsvAdapter(write("x.csv", text)).load_stations(); assert False, frag
        except AdapterError as e: assert frag in str(e), (frag, str(e))
    try: CsvAdapter(os.path.join(d, "nope.csv")).load_stations(); assert False
    except AdapterError: pass

def test_refresh_failure_keeps_old_data_and_success_clears_whatif():
    d = tempfile.mkdtemp(); p = os.path.join(d, "s.csv")
    open(p, "w").write(open(os.path.join(os.path.dirname(__file__), "..", "sample_stations.csv")).read())
    v = ShopfloorView(1280, 720, CsvAdapter(p)); v.selected = v.stations[1]; v.stations[0].x += 3
    assert v.whatif_active and v.refresh() and not v.whatif_active and v.selected.sid == v.stations[1].sid
    open(p, "w").write("garbage"); n = len(v.stations)
    assert not v.refresh() and len(v.stations) == n and "Refresh failed" in v.status_msg


# ---- audit ---------------------------------------------------------------------
def test_audit_chain_and_tamper_detection():
    d = tempfile.mkdtemp(); a = AuditLog(d); a.record("lock_state", frm="LOCKED", to="ENGAGED")
    a.record("select", station="CNC-01"); a.record("reset"); a.close()
    ok, msg = AuditLog.verify(str(a.path)); assert ok, msg
    lines = open(a.path).read().splitlines()
    rec = json.loads(lines[1]); rec["data"]["station"] = "WLD-01"; lines[1] = json.dumps(rec, separators=(",", ":"))
    open(a.path, "w").write("\n".join(lines) + "\n"); assert not AuditLog.verify(str(a.path))[0]
    open(a.path, "w").write(lines[0] + "\n" + lines[2] + "\n"); assert not AuditLog.verify(str(a.path))[0]


def test_bystander_fixture_really_is_draw_pose():
    for dx, dy in ((40, 15), (-40, 15), (40, -15), (-40, -15)):   # extremes of the motion used in the e2e tests
        by = make_hand(0.80, 0.75, ("index",), label="Left"); wiggle_index_tip(by, dx, dy)
        assert classify_raw_gesture(compute_finger_state(by)) == Gesture.DRAW, (dx, dy)


# ---- end-to-end: main.py (drawing app) gated by the lock ----------------------------
def _run_main(script):
    """script(frame_idx, t) -> list of HandResult. Returns max drawn alpha pixels."""
    import cv2, main, hand_tracker
    from drawing_canvas import DrawingCanvas
    made = []
    class Canvas(DrawingCanvas):
        def __init__(s, *a, **k): super().__init__(*a, **k); made.append(s)
    class FakeTracker:
        def __init__(s, *a, **k): s.t0 = time.perf_counter()
        def process(s, frame): return script(time.perf_counter() - s.t0)
        draw_skeleton = staticmethod(hand_tracker.HandTracker.draw_skeleton)
        def __enter__(s): return s
        def __exit__(s, *a): return False
    class FakeCam:
        def __init__(s, *a, **k): s.t0 = None
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s):
            time.sleep(0.02); s.t0 = s.t0 or time.perf_counter()
            return True, np.zeros((720, 1280, 3), np.uint8)
    deadline = {}
    def waitKey(_):
        deadline.setdefault("t", time.perf_counter())
        return ord("q") if time.perf_counter() - deadline["t"] > 4.5 else 255
    saved = (main.HandTracker, main.CameraSession, main.DrawingCanvas, cv2.imshow, cv2.waitKey, cv2.getWindowProperty, cv2.setWindowTitle, cv2.destroyAllWindows, cv2.destroyWindow, cv2.namedWindow)
    try:
        main.HandTracker, main.CameraSession, main.DrawingCanvas = FakeTracker, FakeCam, Canvas
        cv2.imshow = lambda *a, **k: None; cv2.waitKey = waitKey
        cv2.getWindowProperty = lambda *a, **k: 1; cv2.setWindowTitle = lambda *a, **k: None
        cv2.destroyAllWindows = lambda: None; cv2.destroyWindow = lambda *a, **k: None; cv2.namedWindow = lambda *a, **k: None
        assert main.run() == 0
    finally:
        (main.HandTracker, main.CameraSession, main.DrawingCanvas, cv2.imshow, cv2.waitKey,
         cv2.getWindowProperty, cv2.setWindowTitle, cv2.destroyAllWindows, cv2.destroyWindow, cv2.namedWindow) = saved
    return int(np.count_nonzero(made[0].layer[:, :, 3]))

def _draw_hand(t, x0=0.40):
    h = make_hand(x0, 0.75, ("index",)); wiggle_index_tip(h, 40 * np.sin(t * 3), 15 * np.cos(t * 3)); return h

def test_main_locked_ignores_drawing():
    config.REQUIRE_LOCK_FOR_DRAWING = True
    px = _run_main(lambda t: [_draw_hand(t)])             # draws from the start, never unlocks
    assert px == 0, f"drew {px} px while locked"

def test_main_unlock_then_draw_and_bystander_ignored():
    config.REQUIRE_LOCK_FOR_DRAWING = True
    def script(t):
        owner = make_hand(0.40, 0.75, ENG) if t < 2.0 else _draw_hand(t)
        by = make_hand(0.80, 0.75, ("index",), label="Left"); wiggle_index_tip(by, 40 * np.sin(t * 4), 15 * np.cos(t * 4))
        return [owner, by]
    px = _run_main(script)
    assert px > 200, f"owner drew only {px} px after unlock"

def test_main_bystander_alone_cannot_draw_even_after_owner_unlocked_then_left():
    config.REQUIRE_LOCK_FOR_DRAWING = True
    def script(t):
        if t < 2.0: return [make_hand(0.40, 0.75, ENG)]
        by = make_hand(0.80, 0.75, ("index",), label="Left"); wiggle_index_tip(by, 40 * np.sin(t * 4), 15 * np.cos(t * 4))
        return [by]                                         # owner gone, someone else draws
    px = _run_main(script)
    assert px == 0, f"bystander drew {px} px"

def test_main_without_lock_still_works():
    config.REQUIRE_LOCK_FOR_DRAWING = False
    try:
        px = _run_main(lambda t: [_draw_hand(t)])
        assert px > 200
    finally:
        config.REQUIRE_LOCK_FOR_DRAWING = True


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try: fn(); print("PASS", name)
            except Exception as e:
                import traceback; fails += 1; print("FAIL", name, repr(e)); traceback.print_exc(limit=3)
    sys.exit(1 if fails else 0)
