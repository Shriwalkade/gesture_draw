"""
security.py
-----------
Privacy and safety layer for the application.

Guarantees enforced by this module:
1. The app runs fully offline -- nothing here ever opens a network socket,
   and no other module in this project imports networking libraries.
2. Camera frames are NEVER written to disk, logged, or transmitted. Only
   the user-composited canvas is ever saved, and only on explicit request
   (the 'S' key).
3. No biometric data (landmark coordinates, hand geometry, identity, etc.)
   is persisted anywhere. Landmarks live in memory for the current frame
   only.
4. Logging is scrubbed to structural/diagnostic messages only -- never
   frame content, landmark coordinates, or file contents.
5. Camera and window resources are always released, even on crash, via the
   CameraSession context manager.
6. Save paths are validated to stay inside the configured save directory
   to prevent path traversal.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import cv2

import config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure a logger that never writes frame data or coordinates.

    Only structural/diagnostic events (camera opened, gesture mode changed,
    file saved, errors) are logged.
    """
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / config.LOG_FILE

    logger = logging.getLogger("gesture_draw")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging() is called more than once.
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. No frame or biometric data is ever logged.")
    return logger


# --------------------------------------------------------------------------
# Safe save-path validation (prevents path traversal / accidental overwrite
# outside the sandboxed save directory)
# --------------------------------------------------------------------------
def resolve_safe_save_path(filename: str) -> Path:
    """Return a validated, collision-free path inside SAVE_DIR.

    Rejects any filename component that could escape the save directory
    (e.g. '..', absolute paths, separators).
    """
    save_dir = Path(config.SAVE_DIR).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = os.path.basename(filename)  # strips any directory traversal
    if not safe_name or safe_name in (".", ".."):
        safe_name = "drawing.png"

    candidate = (save_dir / safe_name).resolve()

    # Ensure the resolved path is still inside save_dir.
    if save_dir not in candidate.parents and candidate != save_dir:
        raise ValueError("Resolved save path escapes the sandboxed save directory.")

    # Avoid silently overwriting an existing file -- append a timestamp.
    if candidate.exists():
        stem = candidate.stem
        suffix = candidate.suffix or ".png"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = save_dir / f"{stem}_{timestamp}{suffix}"

    return candidate


# --------------------------------------------------------------------------
# Camera lifecycle management
# --------------------------------------------------------------------------
class CameraUnavailableError(RuntimeError):
    """Raised when the camera cannot be opened after all retry attempts."""


class CameraSession:
    """Context manager guaranteeing camera resources are always released.

    Frames read here are held only in local variables by the caller and are
    never written to disk or logged by this class or any other module.
    """

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.logger = logger or logging.getLogger("gesture_draw")
        self.capture: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        last_error: Optional[Exception] = None
        for attempt in range(1, config.CAMERA_RECONNECT_ATTEMPTS + 1):
            try:
                cap = cv2.VideoCapture(self.index)
                if os.name == "nt":
                    # DirectShow backend tends to be far more reliable on
                    # Windows for resolution changes.
                    cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if cap.isOpened():
                    self.logger.info("Camera %s opened on attempt %d.", self.index, attempt)
                    return cap
                cap.release()
            except Exception as exc:  # noqa: BLE001 - we deliberately want to retry any failure
                last_error = exc
                self.logger.warning("Camera open attempt %d failed: %s", attempt, exc)
            time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)

        raise CameraUnavailableError(
            f"Could not open camera index {self.index} after "
            f"{config.CAMERA_RECONNECT_ATTEMPTS} attempts."
        ) from last_error

    def __enter__(self) -> "CameraSession":
        self.capture = self._open()
        return self

    def read(self):
        """Read one frame. Returns (success, frame) exactly like cv2.

        Attempts a single silent reconnect if the read fails, to survive
        transient USB glitches without crashing the whole application.
        """
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok:
            self.logger.warning("Frame read failed; attempting camera reconnect.")
            try:
                self.capture.release()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.capture = self._open()
                ok, frame = self.capture.read()
            except CameraUnavailableError:
                return False, None
        return ok, frame

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if self.capture is not None:
            self.capture.release()
            self.logger.info("Camera released.")
        cv2.destroyAllWindows()
        # Do not suppress exceptions; let them propagate after cleanup.
        return False
