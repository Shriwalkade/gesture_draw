"""
drawing_canvas.py
-------------------
Owns the persistent drawing surface and all mutation operations on it
(stroke drawing, erasing, clearing, saving, brush/eraser size + color).

The canvas is stored as a BGRA numpy array so strokes can be composited
onto the live camera feed with proper alpha blending, and saved as a
transparent PNG on request.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from security import resolve_safe_save_path
from utils import clamp, euclidean


class DrawingCanvas:
    def __init__(self, width: int, height: int, logger: Optional[logging.Logger] = None) -> None:
        self.width = width
        self.height = height
        self.logger = logger or logging.getLogger("gesture_draw")

        # BGRA: alpha channel lets us composite only the drawn pixels onto
        # the camera frame, and lets us save a transparent PNG.
        self.layer = np.zeros((height, width, 4), dtype=np.uint8)

        self.brush_size = config.DEFAULT_BRUSH_SIZE
        self.eraser_size = config.DEFAULT_ERASER_SIZE
        self._color_index = 0
        self.brush_color = config.BRUSH_COLOR_PALETTE[self._color_index]

        self._last_draw_point: Optional[Tuple[int, int]] = None

    # ---------------------------------------------------------------- draw
    def begin_stroke(self) -> None:
        """Call when a new draw stroke starts (e.g. gesture just became DRAW)."""
        self._last_draw_point = None

    def draw_to(self, point: Tuple[float, float]) -> None:
        """Draw a smooth segment from the last point to `point`.

        Guards against tracking jumps: if the jump is too large relative to
        the frame diagonal, the segment is skipped and treated as the start
        of a new sub-stroke, so a momentary tracking glitch doesn't draw a
        streak across the whole canvas.
        """
        x, y = int(round(point[0])), int(round(point[1]))
        x = int(clamp(x, 0, self.width - 1))
        y = int(clamp(y, 0, self.height - 1))

        color_bgra = (*self.brush_color, 255)

        if self._last_draw_point is None:
            cv2.circle(self.layer, (x, y), self.brush_size // 2 + 1, color_bgra, -1, lineType=cv2.LINE_AA)
            self._last_draw_point = (x, y)
            return

        diag = float(np.hypot(self.width, self.height))
        jump = euclidean(self._last_draw_point, (x, y))
        if jump / diag > config.MAX_JUMP_FRACTION:
            # Treat as a new sub-stroke rather than connecting across the jump.
            cv2.circle(self.layer, (x, y), self.brush_size // 2 + 1, color_bgra, -1, lineType=cv2.LINE_AA)
            self._last_draw_point = (x, y)
            return

        cv2.line(self.layer, self._last_draw_point, (x, y), color_bgra, self.brush_size, lineType=cv2.LINE_AA)
        cv2.circle(self.layer, (x, y), self.brush_size // 2, color_bgra, -1, lineType=cv2.LINE_AA)
        self._last_draw_point = (x, y)

    def end_stroke(self) -> None:
        self._last_draw_point = None

    # --------------------------------------------------------------- erase
    def erase_at(self, point: Tuple[float, float]) -> None:
        x, y = int(round(point[0])), int(round(point[1]))
        x = int(clamp(x, 0, self.width - 1))
        y = int(clamp(y, 0, self.height - 1))
        cv2.circle(self.layer, (x, y), self.eraser_size, (0, 0, 0, 0), -1, lineType=cv2.LINE_AA)

    # ---------------------------------------------------------------- misc
    def clear(self) -> None:
        self.layer[:] = 0
        self._last_draw_point = None
        self.logger.info("Canvas cleared by user.")

    def cycle_color(self) -> None:
        self._color_index = (self._color_index + 1) % len(config.BRUSH_COLOR_PALETTE)
        self.brush_color = config.BRUSH_COLOR_PALETTE[self._color_index]

    def increase_brush(self) -> None:
        self.brush_size = int(clamp(self.brush_size + config.BRUSH_STEP, config.MIN_BRUSH_SIZE, config.MAX_BRUSH_SIZE))

    def decrease_brush(self) -> None:
        self.brush_size = int(clamp(self.brush_size - config.BRUSH_STEP, config.MIN_BRUSH_SIZE, config.MAX_BRUSH_SIZE))

    def increase_eraser(self) -> None:
        self.eraser_size = int(clamp(self.eraser_size + config.ERASER_STEP, config.MIN_ERASER_SIZE, config.MAX_ERASER_SIZE))

    def decrease_eraser(self) -> None:
        self.eraser_size = int(clamp(self.eraser_size - config.ERASER_STEP, config.MIN_ERASER_SIZE, config.MAX_ERASER_SIZE))

    def resize(self, width: int, height: int) -> None:
        """Resize the persistent layer (e.g. camera resolution changed),
        preserving existing strokes by resampling.
        """
        if (width, height) == (self.width, self.height):
            return
        self.layer = cv2.resize(self.layer, (width, height), interpolation=cv2.INTER_LINEAR)
        self.width, self.height = width, height

    # ----------------------------------------------------------- compositing
    def composite_onto(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Alpha-blend the drawing layer on top of a BGR camera frame."""
        alpha = (self.layer[:, :, 3:4].astype(np.float32)) / 255.0
        fg = self.layer[:, :, :3].astype(np.float32)
        bg = frame_bgr.astype(np.float32)
        out = fg * alpha + bg * (1.0 - alpha)
        return out.astype(np.uint8)

    def draw_eraser_ring(self, frame_bgr: np.ndarray, point: Tuple[float, float]) -> np.ndarray:
        """Overlay a semi-transparent red ring showing the eraser's reach."""
        x, y = int(round(point[0])), int(round(point[1]))
        overlay = frame_bgr.copy()
        cv2.circle(overlay, (x, y), self.eraser_size, config.ERASER_RING_COLOR_BGR, 3, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), self.eraser_size, config.ERASER_RING_COLOR_BGR, -1, lineType=cv2.LINE_AA)
        return cv2.addWeighted(overlay, config.ERASER_RING_ALPHA, frame_bgr, 1 - config.ERASER_RING_ALPHA, 0)

    # ----------------------------------------------------------------- save
    def save(self, filename: str = "drawing.png") -> str:
        """Save only the drawing layer (transparent PNG) -- never the camera
        frame -- to the sandboxed save directory. Returns the final path.
        """
        path = resolve_safe_save_path(filename)
        ok = cv2.imwrite(str(path), self.layer)
        if not ok:
            raise IOError(f"Failed to write drawing to {path}")
        self.logger.info("Drawing saved to %s", path)
        return str(path)
