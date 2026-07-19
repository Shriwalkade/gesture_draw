"""
hand_tracker.py
----------------
Thin, robust wrapper around MediaPipe's HandLandmarker (the current
`mediapipe.tasks.vision` API).

Note on API version: MediaPipe removed the old `mp.solutions.hands` Python
API from recent package releases. The supported way to run hand tracking
now is the Tasks API (`mediapipe.tasks.vision.HandLandmarker`), which is
what this module uses. It requires a separately-downloaded `.task` model
bundle (see download_model.py / README) -- that download is a one-time
setup step, not something the running application ever fetches, so the
"fully offline at runtime" guarantee still holds.

Responsibilities:
- Run hand detection/tracking on a BGR frame (VIDEO running mode, since we
  feed it a synchronous stream of frames with increasing timestamps).
- Convert normalized landmarks to both normalized and pixel coordinates.
- Report per-hand confidence and handedness (Left/Right).
- Draw the skeleton and fingertip markers for the HUD.
- Never persist frames or landmarks -- everything is returned to the
  caller and discarded once the frame loop moves on.
"""

from __future__ import annotations

import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmark indices, named for readability (MediaPipe hand model, 21 points).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class ModelNotFoundError(RuntimeError):
    """Raised when the hand_landmarker.task model bundle can't be obtained."""


def ensure_model_available(model_path: str, logger: logging.Logger) -> None:
    """Guarantee the model file exists on disk, downloading it once if needed.

    This is the ONLY place in the whole application that ever touches the
    network, and it only fires the first time the app runs (or if the file
    was deleted). Every subsequent launch finds the file already on disk
    and makes no network call at all -- the "fully offline" guarantee holds
    for every run after the first.
    """
    dest = Path(model_path)
    if dest.is_file():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Hand-tracking model not found locally; downloading it once from %s", MODEL_URL)
    print(f"First run: downloading hand-tracking model (~8 MB) from:\n  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, dest)
    except (urllib.error.URLError, OSError) as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)  # don't leave a partial/corrupt file behind
        raise ModelNotFoundError(
            f"Could not download the hand-tracking model automatically ({exc}).\n"
            f"If this machine has no internet access, download it manually from:\n"
            f"  {MODEL_URL}\n"
            f"on a machine that does, and place it at '{dest}'."
        ) from exc

    size_kb = dest.stat().st_size / 1024
    logger.info("Model downloaded (%.0f KB) to %s. No further downloads will happen.", size_kb, dest)
    print(f"Model downloaded ({size_kb:.0f} KB). Starting camera...")


@dataclass
class HandResult:
    """All the data extracted for a single detected hand in one frame."""

    handedness: str  # "Left" or "Right" (as reported by the model; see note in README about mirroring)
    confidence: float
    landmarks_norm: np.ndarray  # shape (21, 2), normalized [0,1] image coords
    landmarks_px: np.ndarray  # shape (21, 2), pixel coords in the given frame
    palm_size: float  # normalized scale reference (wrist -> middle_mcp distance)


class HandTracker:
    """Wraps mediapipe.tasks.vision.HandLandmarker with defensive error handling."""

    def __init__(self, logger: Optional[logging.Logger] = None, model_path: Optional[str] = None) -> None:
        self.logger = logger or logging.getLogger("gesture_draw")
        model_path = model_path or config.MODEL_PATH

        ensure_model_available(model_path, self.logger)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        """Run detection on a single BGR frame. Returns a list of HandResult.

        Never mutates or stores the input frame. Any internal failure is
        caught and logged; an empty list is returned so the app can keep
        running (temporary tracking loss must never crash the app).
        """
        try:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires a strictly increasing timestamp per call.
            timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Hand detection failed on this frame: %s", exc)
            return []

        hands_out: List[HandResult] = []
        if not result or not result.hand_landmarks:
            return hands_out

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            norm = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float64)
            px = norm.copy()
            px[:, 0] *= w
            px[:, 1] *= h

            confidence = 0.0
            label = "Unknown"
            if idx < len(result.handedness) and result.handedness[idx]:
                top = result.handedness[idx][0]
                confidence = float(top.score) if top.score is not None else 0.0
                label = top.category_name or "Unknown"

            palm_size = float(np.linalg.norm(norm[WRIST] - norm[MIDDLE_MCP]))
            palm_size = max(palm_size, 1e-4)  # never zero -- guards later division

            hands_out.append(
                HandResult(
                    handedness=label,
                    confidence=confidence,
                    landmarks_norm=norm,
                    landmarks_px=px,
                    palm_size=palm_size,
                )
            )

        return hands_out

    @staticmethod
    def draw_skeleton(frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> None:
        """Draw the hand skeleton (bone connections + joint dots + fingertip
        markers) directly from pixel-space landmarks.
        """
        for connection in HAND_CONNECTIONS:
            x1, y1 = landmarks_px[connection.start]
            x2, y2 = landmarks_px[connection.end]
            cv2.line(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2, cv2.LINE_AA)

        fingertip_indices = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
        for idx, (x, y) in enumerate(landmarks_px):
            if idx in fingertip_indices:
                cv2.circle(frame_bgr, (int(x), int(y)), 7, (0, 140, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (255, 180, 0), -1, cv2.LINE_AA)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
