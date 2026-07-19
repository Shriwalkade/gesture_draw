"""
config.py
---------
Central configuration for the Hand Gesture Drawing application.
Every tunable constant lives here so behavior can be adjusted without
touching logic code.
"""

from dataclasses import dataclass, field
from typing import Tuple


# --------------------------------------------------------------------------
# Camera / capture
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
TARGET_FPS: int = 60
CAMERA_RECONNECT_ATTEMPTS: int = 5
CAMERA_RECONNECT_DELAY_SEC: float = 1.0

# --------------------------------------------------------------------------
# MediaPipe Hands
# --------------------------------------------------------------------------
# Path to the HandLandmarker model bundle. Not shipped in this repo -- run
# `python download_model.py` once (one-time internet access) to fetch it.
# The running application never downloads anything itself.
MODEL_PATH: str = "models/hand_landmarker.task"

MAX_NUM_HANDS: int = 2
MIN_DETECTION_CONFIDENCE: float = 0.6
MIN_TRACKING_CONFIDENCE: float = 0.6

# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
# A gesture must be held continuously for this many milliseconds before
# it is "activated" (prevents flicker / accidental switching).
GESTURE_STABLE_MS: float = 300.0

# Number of recent classification samples kept for majority-vote smoothing.
GESTURE_HISTORY_LEN: int = 7

# A finger is considered "extended" when the angle (in degrees) between the
# (MCP->PIP) and (PIP->TIP) bone vectors is below this threshold, i.e. the
# finger is roughly straight, AND the tip is farther from the palm center
# than the PIP joint (scale-normalized). This combination is robust to
# hand rotation, hand size, and left/right hand because it never relies on
# raw pixel coordinates or absolute image-space thresholds.
FINGER_STRAIGHT_ANGLE_DEG: float = 55.0

# Extra normalized-distance margin (tip must be at least this fraction of
# palm-size farther from wrist than the PIP joint) to call a finger extended.
FINGER_EXTENSION_MARGIN: float = 0.05

# Thumb uses a separate check (it never folds the same way as other
# fingers) based on the angle at the thumb MCP joint.
THUMB_STRAIGHT_ANGLE_DEG: float = 40.0

# Smoothing factor (0..1) for the exponential moving average applied to the
# drawing point. Higher = smoother but more lag.
POINT_SMOOTHING_ALPHA: float = 0.45

# If the fingertip moves farther than this (normalized by frame diagonal)
# between two consecutive frames, treat it as a tracking jump and do not
# draw a connecting line (prevents wild streaks across the canvas).
MAX_JUMP_FRACTION: float = 0.18

# --------------------------------------------------------------------------
# Drawing / canvas
# --------------------------------------------------------------------------
DEFAULT_BRUSH_SIZE: int = 6
MIN_BRUSH_SIZE: int = 1
MAX_BRUSH_SIZE: int = 60
BRUSH_STEP: int = 2

DEFAULT_ERASER_SIZE: int = 40
MIN_ERASER_SIZE: int = 10
MAX_ERASER_SIZE: int = 200
ERASER_STEP: int = 5

DEFAULT_BRUSH_COLOR_BGR: Tuple[int, int, int] = (0, 0, 255)  # red (BGR)
ERASER_RING_COLOR_BGR: Tuple[int, int, int] = (0, 0, 255)
ERASER_RING_ALPHA: float = 0.35

BRUSH_COLOR_PALETTE = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 255),  # white
]

# --------------------------------------------------------------------------
# Filesystem / persistence (privacy-relevant, see security.py)
# --------------------------------------------------------------------------
SAVE_DIR: str = "saved_drawings"
LOG_DIR: str = "logs"
LOG_FILE: str = "app.log"

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
WINDOW_NAME: str = "Hand Gesture Drawing"
HUD_FONT_SCALE: float = 0.55
HUD_COLOR_BGR: Tuple[int, int, int] = (255, 255, 255)
HUD_BG_ALPHA: float = 0.45

# Landmark connections for skeleton drawing use MediaPipe's own constant,
# imported directly where needed to avoid duplicating that table here.

KEY_QUIT = ord('q')
KEY_CLEAR = ord('c')
KEY_SAVE = ord('s')
KEY_BRUSH_INC = ord('=')   # '+' key on most layouts without shift
KEY_BRUSH_INC_ALT = ord('+')
KEY_BRUSH_DEC = ord('-')
KEY_ERASER_INC = ord('e')
KEY_ERASER_DEC = ord('r')
KEY_COLOR_NEXT = ord('n')

# --------------------------------------------------------------------------
# Professional UI Configuration
# --------------------------------------------------------------------------

# Show FPS on screen
SHOW_FPS = True

# Show gesture name
SHOW_GESTURE = True

# Show help overlay at startup
SHOW_HELP_AT_START = True

# Help overlay auto-hide (seconds)
HELP_TIMEOUT = 10

# Enable dashboard
SHOW_DASHBOARD = True

# Enable notifications
SHOW_NOTIFICATIONS = True

# Enable session timer
SHOW_SESSION_TIMER = True

# Enable gesture history
SHOW_GESTURE_HISTORY = True

# Enable smoothing
ENABLE_CURSOR_SMOOTHING = True

# Dashboard colors (BGR)
DASHBOARD_BG = (35, 35, 35)
DASHBOARD_TEXT = (255, 255, 255)
SUCCESS_COLOR = (0, 220, 0)
WARNING_COLOR = (0, 180, 255)
ERROR_COLOR = (0, 0, 255)

APP_NAME = "AI Gesture Studio"
APP_VERSION = "2.0"