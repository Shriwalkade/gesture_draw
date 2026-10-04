# PLM extension: owner-locked gesture UI + virtual shopfloor (prototype)

Run the shopfloor demo: `python plm_main.py` (mock data) or `python plm_main.py --csv sample_stations.csv`
Drawing app (now lock-gated): `python main.py`
Tests (no camera): `python tests/test_plm.py && python tests/test_plm2.py`   Benchmark: `python bench.py`

## Operator flow
1. Hold index+middle+ring up (pinky folded) ~1.5 s -> UNLOCKED. Only that hand is "owner"; others are grey dots, ignored.
2. Hover a station 0.7 s -> select (info card). Pinch+drag -> what-if relocate (never saved).
3. Hover the panel buttons 0.7 s: LAYER (STATUS/REVISION/WIP), RESET, REFRESH (re-read data source), LOCK. No keyboard needed.
4. Lock: hold the 3-finger pose again, hover LOCK, or `X`. Auto-locks: owner lost 1.5 s, idle 30 s.
5. Another hand overlapping the owner's -> input freezes ("Paused: hands overlapping"), resumes when they separate.

## Files
| File | Role |
|---|---|
| `session_lock.py` | Owner lock: engage pose, continuity tracking, overlap freeze, auto-lock |
| `shopfloor.py` | Faux-3D floor overlay, dwell select/buttons, pinch what-if, layers, event queue |
| `plm_adapter.py` | Read-only data seam: `MockAdapter`, `CsvAdapter` (strict validation) |
| `audit.py` | Hash-chained JSONL audit log with `verify()` |
| `plm_main.py` | Shopfloor entry point (camera -> lock -> view -> audit) |
| `main.py` | Original drawing app, now gated by the lock |
| `bench.py` | Per-frame cost of new code |
| `tests/` | 28 headless tests incl. end-to-end run of `main.py` with a scripted camera |

## Changes to ORIGINAL behaviour (review these)
- Erase gesture is now index+middle (was open palm). Revert: `ERASE_GESTURE = "OPEN_PALM"`.
- `main.py` ignores all hands until unlocked. Disable: `REQUIRE_LOCK_FOR_DRAWING = False`.
- `requirements.txt`: `mediapipe>=0.10.30` (was `>=0.10.14`). Only 0.10.33 was actually run; the 0.10.30 floor comes from the original README's claim.

## What is NOT done / NOT true
- **Not authentication.** The lock cannot tell who the user is; anyone can unlock with the pose. The audit log's only actor is a random session id.
- **Not AR.** Webcam overlay with a fake extrusion. No spatial anchoring, no real 3D.
- **No real PLM/MES connection.** `CsvAdapter` reads a schema I invented; a real connector = one `PLMAdapter` subclass.
- **Never run on a live camera.** All tests use synthetic hands (geometrically plausible, fed through the real finger classifier).
- **Thresholds are uncalibrated guesses:** `LOCK_MIN_ENGAGE_PALM`, `LOCK_ENGAGE_ZONE`, `LOCK_OVERLAP_MARGIN`, `LOCK_MATCH_*`.
- **Overlap freeze is conservative:** a legitimate two-hand gesture near the owner's hand will pause input (and idle-lock after 30 s).
- **Gloves untested.** MediaPipe on gloved hands is unverified.
- **Audit log is tamper-evident, not tamper-proof** (whole-file rewrite/deletion is undetectable). Ship off-box for real assurance.
- Drawing app has no audit logging; only `plm_main.py` does.
