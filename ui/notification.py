"""
notification.py

Popup notification system for AI Gesture Studio
"""

from __future__ import annotations

import cv2
import time


class NotificationManager:

    def __init__(self):

        self.message = ""
        self.color = (0, 255, 0)
        self.start_time = 0
        self.duration = 2.0

    def show(self, message, color=(0, 255, 0), duration=2):

        self.message = message
        self.color = color
        self.duration = duration
        self.start_time = time.time()

    def draw(self, frame):

        if self.message == "":
            return frame

        elapsed = time.time() - self.start_time

        if elapsed > self.duration:
            self.message = ""
            return frame

        h, w = frame.shape[:2]

        overlay = frame.copy()

        box_width = 380
        box_height = 60

        x = int((w - box_width) / 2)
        y = 20

        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            (40, 40, 40),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0,
            frame,
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            self.color,
            2,
        )

        cv2.putText(
            frame,
            self.message,
            (x + 20, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.color,
            2,
            cv2.LINE_AA,
        )

        return frame