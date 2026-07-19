"""
Professional Splash Screen
"""

import cv2
import time


class SplashScreen:

    def __init__(self, duration=2.5):
        self.duration = duration

    def show(self):

        width = 900
        height = 550

        img = 255 * __import__("numpy").ones((height, width, 3), dtype="uint8")

        cv2.rectangle(img, (0, 0), (width, height), (35, 35, 35), -1)

        cv2.putText(
            img,
            "AI Gesture Studio",
            (180, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 255),
            3,
        )

        cv2.putText(
            img,
            "Professional Gesture Recognition System",
            (120, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            img,
            "Loading Camera...",
            (280, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )

        cv2.putText(
            img,
            "Loading AI Model...",
            (280, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )

        cv2.putText(
            img,
            "Initializing Dashboard...",
            (280, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )

        cv2.putText(
            img,
            "Version 2.0",
            (360, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
        )

        cv2.imshow("Loading", img)

        cv2.waitKey(int(self.duration * 1000))

        cv2.destroyWindow("Loading")