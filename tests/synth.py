"""Geometrically plausible synthetic hands (real HandResult) so tests exercise the REAL
finger classifier. Upright hand, wrist at (wx, wy) in normalized image coords."""
import numpy as np
from hand_tracker import HandResult

MCP = {"index": (-0.030, -0.115), "middle": (0.0, -0.120), "ring": (0.028, -0.113), "pinky": (0.052, -0.100)}
FIRST = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(wx, wy, up=("index",), scale=1.0, label="Right", conf=0.95, w=1280, h=720):
    """up = names of extended fingers; the others are curled. Thumb is always folded."""
    n = np.zeros((21, 2))
    n[0] = (0, 0)
    n[1], n[2], n[3], n[4] = (0.030, -0.020), (0.045, -0.040), (0.040, -0.060), (0.020, -0.075)  # folded across palm
    for name, i in FIRST.items():
        mx, my = MCP[name]
        n[i] = (mx, my)
        n[i + 1] = (mx, my - 0.045)                                   # PIP
        if name in up:
            n[i + 2] = (mx, my - 0.075); n[i + 3] = (mx, my - 0.100)  # straight
        else:
            n[i + 2] = (mx, my - 0.035); n[i + 3] = (mx, my - 0.010)  # curled back toward palm
    n = n * scale + np.array([wx, wy])
    px = n * np.array([w, h])
    palm = float(np.linalg.norm(n[0] - n[9]))
    return HandResult(handedness=label, confidence=conf, landmarks_norm=n, landmarks_px=px, palm_size=palm)


def set_index_tip_px(hand, x, y):
    """Move only the index fingertip (for drawing motion)."""
    hand.landmarks_px[8] = (x, y)
    hand.landmarks_norm[8] = (x / 1280, y / 720)


def wiggle_index_tip(hand, dx_px, dy_px):
    """Offset the index fingertip from its natural position by a few px; keeps the draw pose valid."""
    hand.landmarks_px[8] = hand.landmarks_px[8] + (dx_px, dy_px)
    hand.landmarks_norm[8] = hand.landmarks_px[8] / (1280, 720)
