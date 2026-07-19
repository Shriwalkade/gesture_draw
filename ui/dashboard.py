"""
dashboard.py

Professional dashboard overlay for AI Gesture Studio.
"""

from __future__ import annotations

import cv2
import time


class Dashboard:
    def __init__(self):
        self.start_time = time.time()

    def _draw_panel(self, frame, x, y, w, h):
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (40, 40, 40),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0,
            frame,
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (90, 90, 90),
            1,
        )

    def _put(self, frame, text, x, y, color=(255, 255, 255), scale=0.55):

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw(
        self,
        frame,
        fps=0,
        gesture="None",
        mode="Idle",
        brush=8,
        color_name="Blue",
    ):

        h, w = frame.shape[:2]

        self._draw_panel(frame, 10, 10, 340, 185)

        self._put(frame, "AI Gesture Studio", 25, 35, (0, 255, 255), 0.75)

        self._put(frame, f"FPS : {fps}", 25, 65)

        self._put(frame, f"Gesture : {gesture}", 25, 90)

        self._put(frame, f"Mode : {mode}", 25, 115)

        self._put(frame, f"Brush : {brush}px", 25, 140)

        self._put(frame, f"Color : {color_name}", 25, 165)

        elapsed = int(time.time() - self.start_time)

        minutes = elapsed // 60

        seconds = elapsed % 60

        self._draw_panel(frame, 10, h - 60, 260, 50)

        self._put(
            frame,
            f"Session : {minutes:02}:{seconds:02}",
            25,
            h - 28,
            (0, 255, 0),
        )

        return frame