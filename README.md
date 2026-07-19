# Hand Gesture Drawing & Erasing

Real-time webcam drawing app controlled entirely by hand gestures, built on
OpenCV + MediaPipe Hands. Fully offline. No network calls, no frame storage,
no biometric persistence.

## Gestures

| Gesture | Fingers | Effect |
|---|---|---|
| **Draw** | Index up, middle/ring/pinky folded | Continuous smoothed stroke follows fingertip |
| **Erase** | All five fingers extended (open palm) | Red translucent circle erases inside it |
| **Idle** | Anything else | Drawing/erasing disabled |

A gesture must hold steady for **300 ms** before it activates, so brief
transitions while your fingers move don't cause accidental switching.

## Requirements — read this before installing

- **Python 3.9 – 3.12 only.** MediaPipe does not currently publish wheels
  for Python 3.13+ (build fails on Bazel/pybind11 ABI changes — this is an
  upstream limitation, not something this app can work around). Check your
  version with `python --version` before you spend time debugging install
  errors.
- A working webcam (built-in or USB).
- ~200 MB free disk for dependencies (mediapipe ships its own native
  runtime + models).

## Installation

```bash
# 1. Create an isolated environment (strongly recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

If `pip install mediapipe` fails outright, it's almost always the Python
version. Run `python --version`; if it's 3.13+, install 3.12 alongside it
and recreate the venv with `python3.12 -m venv venv`.

## Running

```bash
python main.py
```

**First run only:** the app automatically downloads the hand-tracking model
(~8 MB, from Google's official MediaPipe model host) the first time you
start it — you'll see a one-line download message before the camera window
opens. Every run after that finds the file already on disk and makes zero
network calls; that first download is the only network access anywhere in
this codebase.

If the machine running the app has no internet access, pre-fetch the model
on a machine that does by running `python download_model.py` there, then
copy the resulting `models/hand_landmarker.task` file over.

A window opens showing your camera feed with the hand skeleton overlaid and
a status bar (FPS, active mode, confidence, brush/eraser size).

## Controls

| Key | Action |
|---|---|
| `C` | Clear canvas |
| `S` | Save drawing (transparent PNG, written to `saved_drawings/`) |
| `Q` | Quit |
| `+` / `-` | Increase / decrease brush size |
| `E` / `R` | Increase / decrease eraser size |
| `N` | Cycle brush color |
| (window close button) | Quit cleanly |

## Why this should work for most hands/lighting

Finger up/down detection does **not** use fixed pixel thresholds. It uses:

- The angle between each finger's MCP→PIP and PIP→TIP bone vectors (a
  straight finger has a small bend angle regardless of hand rotation).
- The fingertip-to-wrist distance relative to the PIP-to-wrist distance,
  normalized by the hand's own palm size (wrist-to-middle-MCP distance).

Because everything is normalized by the hand's own geometry, the same
thresholds work for small hands, large hands, hands close to or far from
the camera, and either left or right hand — MediaPipe already reports
handedness and 21 landmarks per hand regardless of skin tone or lighting;
this app doesn't add any additional bias on top of that.

**Honest limitation:** MediaPipe's underlying detector is a fixed pretrained
model. If MediaPipe itself fails to detect a hand (extreme low light,
hand mostly out of frame, heavy motion blur), no amount of downstream
logic here recovers detections that never happened — the app just falls
back to idle and waits for a clean frame, which it does not crash on.

## Privacy / security

- The **only** network access anywhere in this codebase is the one-time
  model download on first launch (see Running, above). Every run after
  that is 100% offline — verify yourself with
  `grep -rn "socket\|requests\|urllib" *.py` and you'll only find it in
  `hand_tracker.py`'s `ensure_model_available()` and the optional
  `download_model.py` pre-fetch script.
- Camera frames are **never** written to disk or logged. Only the drawing
  layer is saved, and only when you press `S`.
- Save paths are sandboxed to `saved_drawings/` and can't traverse outside
  it.
- Logs (`logs/app.log`) contain only structural events (camera opened,
  mode changed, file saved, errors) — never coordinates, frames, or
  identity data.

## Project structure

```
gesture_draw/
├── main.py              # entry point, frame loop, HUD, keyboard handling
├── hand_tracker.py       # MediaPipe HandLandmarker wrapper + skeleton drawing
├── gesture_detector.py   # scale/rotation-invariant finger state + stabilizer
├── drawing_canvas.py     # persistent BGRA canvas, brush/eraser, save/clear
├── security.py           # logging, safe camera lifecycle, sandboxed saves
├── utils.py              # FPS counter, vector math, point smoother
├── config.py             # every tunable constant
├── download_model.py     # one-time setup: fetches hand_landmarker.task
├── requirements.txt
├── README.md
├── models/               # created by download_model.py (not committed)
└── assets/
```

**API note:** MediaPipe deprecated and removed the old `mp.solutions.hands`
Python API in current releases. This project uses the current replacement,
`mediapipe.tasks.vision.HandLandmarker`, which is why a separate model
download step exists. If you find older tutorials referencing
`mp.solutions.hands`, that code will not run on mediapipe ≥0.10.30 —
verified against the actually installed package while building this, not
assumed from memory.

## Known trade-offs (not hidden from you)

- Only the **highest-confidence hand** drives drawing/erasing when two
  hands are visible; the second hand is tracked and drawn but doesn't
  control anything. Two-hand collaborative drawing is out of scope as
  written — extend `main.py`'s hand-selection logic if you need it.
- `model_complexity=1` in `config.py` trades some CPU for accuracy. Drop it
  to `0` if you're on a low-power CPU and see FPS below ~20.
- The 300 ms stability window is a fixed constant, not adaptive. If it
  feels laggy on your hardware, lower `GESTURE_STABLE_MS` in `config.py` —
  you'll trade some flicker-resistance for responsiveness.
