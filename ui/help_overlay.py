"""
help_overlay.py

Professional Help Screen
"""

from __future__ import annotations

import cv2


class HelpOverlay:

    def __init__(self):
        self.visible = False

    def toggle(self):
        self.visible = not self.visible

    def draw(self, frame):

        if not self.visible:
            return frame

        h, w = frame.shape[:2]

        overlay = frame.copy()

        x = 30
        y = 30
        width = 420
        height = 460

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (25, 25, 25),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.72,
            frame,
            0.28,
            0,
            frame,
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (100, 100, 100),
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        yy = 60

        cv2.putText(
            frame,
            "AI Gesture Studio",
            (55, yy),
            font,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        yy += 35

        cv2.putText(
            frame,
            "Available Gestures",
            (55, yy),
            font,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        gestures = [
            ("INDEX", "Draw"),
            ("PINCH", "Erase"),
            ("THUMB UP", "Save Drawing"),
            ("OPEN PALM", "Clear Canvas"),
            ("FIST", "Pause"),
            ("VICTORY", "Mouse Mode"),
            ("THUMB + PINKY", "Right Click"),
                    ]
        

        yy += 40

        for gesture, action in gestures:

            cv2.putText(
                frame,
                gesture,
                (55, yy),
                font,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                action,
                (250, yy),
                font,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            yy += 32

        yy += 15

        cv2.putText(
            frame,
            "Keyboard Shortcuts",
            (55, yy),
            font,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        yy += 35

        shortcuts = [

            ("H", "Show / Hide Help"),

            ("C", "Clear Canvas"),

            ("S", "Save Drawing"),

            ("Q", "Quit"),

        ]

        for key, action in shortcuts:

            cv2.putText(
                frame,
                f"[{key}]",
                (55, yy),
                font,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                action,
                (120, yy),
                font,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            yy += 30

        yy += 15

        cv2.putText(
            frame,
            "Press H to Close",
            (55, yy),
            font,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame