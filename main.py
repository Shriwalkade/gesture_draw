"""
main.py
-------
Application entry point for the Hand Gesture Drawing & Erasing app.

Run with:  python main.py

Controls:
    H          -> toggle help overlay
    C          -> clear canvas
    S          -> save drawing (PNG, transparent background)
    Q          -> quit
    + / -      -> increase / decrease brush size
    E / R      -> increase / decrease eraser size
    N          -> cycle brush color
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import cv2

from ui.dashboard import Dashboard
from ui.help_overlay import HelpOverlay
from ui.notification import NotificationManager
from ui.splash_screen import SplashScreen

import config
from drawing_canvas import DrawingCanvas
from gesture_detector import Gesture, GestureDetector
from hand_tracker import HandTracker, ModelNotFoundError
from security import CameraSession, CameraUnavailableError, setup_logging
from utils import FPSCounter, PointSmoother

# Color lookup for dashboard display
COLOR_LOOKUP = {
    (255, 0, 0): "Blue",
    (0, 255, 0): "Green",
    (0, 0, 255): "Red",
    (0, 255, 255): "Yellow",
    (255, 255, 255): "White",
}


def draw_hud(
    frame,
    fps: float,
    gesture: Gesture,
    confidence: float,
    brush_size: int,
    eraser_size: int,
    brush_color_bgr: Tuple[int, int, int],
    hand_count: int,
) -> None:
    """Render a semi-transparent status bar with live diagnostics."""
    h, w = frame.shape[:2]
    bar_height = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, config.HUD_BG_ALPHA, frame, 1 - config.HUD_BG_ALPHA, 0, dst=frame)

    gesture_colors = {
        Gesture.DRAW: (0, 220, 0),
        Gesture.ERASE: (0, 0, 220),
        Gesture.IDLE: (0, 200, 200),
        Gesture.NONE: (150, 150, 150),
    }
    gcolor = gesture_colors.get(gesture, config.HUD_COLOR_BGR)

    text = (
        f"FPS:{fps:4.1f} "
        f"Hands:{hand_count} "
        f"Conf:{confidence:0.2f} "
        f"Mode:{gesture.value:<5} "
        f"Brush:{brush_size:2d} "
        f"Eraser:{eraser_size:3d} "
        f"[H]Help "
        f"[C]Clear "
        f"[S]Save "
        f"[N]Color "
        f"[Q]Quit"
    )
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE, gcolor, 1, cv2.LINE_AA)

    # Brush color swatch.
    cv2.rectangle(frame, (w - 30, 8), (w - 10, 28), brush_color_bgr, -1)
    cv2.rectangle(frame, (w - 30, 8), (w - 10, 28), (255, 255, 255), 1)


def run() -> int:
    logger = setup_logging()
    logger.info("Starting Hand Gesture Drawing application (offline, no network access).")

    canvas: Optional[DrawingCanvas] = None
    smoother = PointSmoother(config.POINT_SMOOTHING_ALPHA)
    detector = GestureDetector()
    fps_counter = FPSCounter()
    dashboard = Dashboard()
    help_overlay = HelpOverlay()
    notification = NotificationManager()

    # Show splash screen
    splash = SplashScreen()
    splash.show()

    # Set window title for better UX
    cv2.setWindowTitle(
        config.WINDOW_NAME,
        "AI Gesture Studio v2.0"
    )

    try:
        with HandTracker(logger) as tracker, CameraSession(
            config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, logger
        ) as camera:

            was_drawing = False
            last_gesture: Gesture = Gesture.NONE

            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    logger.error("Camera unavailable; stopping frame loop.")
                    break

                frame = cv2.flip(frame, 1)  # mirror for natural interaction
                h, w = frame.shape[:2]

                if canvas is None:
                    canvas = DrawingCanvas(w, h, logger)
                elif (canvas.width, canvas.height) != (w, h):
                    canvas.resize(w, h)

                hands = tracker.process(frame)

                active_gesture = Gesture.NONE
                confidence = 0.0
                index_tip: Optional[Tuple[float, float]] = None

                if hands:
                    # Use the highest-confidence hand as the "active" controller
                    # so a second, incidental hand in frame doesn't fight it.
                    primary = max(hands, key=lambda h_: h_.confidence)
                    confidence = primary.confidence
                    active_gesture, fingers, index_tip = detector.process(primary)

                    for h_result in hands:
                        HandTracker.draw_skeleton(frame, h_result.landmarks_px)
                else:
                    active_gesture, _, _ = detector.process(None)

                # ---- state transition handling -------------------------------
                if active_gesture != last_gesture:
                    logger.info("Gesture mode changed: %s -> %s", last_gesture.value, active_gesture.value)
                    if active_gesture != Gesture.DRAW:
                        canvas.end_stroke()
                        smoother.reset()
                    if active_gesture == Gesture.DRAW:
                        canvas.begin_stroke()
                        smoother.reset()
                    last_gesture = active_gesture

                # ---- apply gesture effect -------------------------------------
                if active_gesture == Gesture.DRAW and index_tip is not None:
                    smoothed = smoother.update(index_tip)
                    canvas.draw_to(smoothed)
                    was_drawing = True
                else:
                    if was_drawing:
                        canvas.end_stroke()
                        was_drawing = False
                    smoother.reset()

                composited = canvas.composite_onto(frame)

                if active_gesture == Gesture.ERASE and index_tip is not None:
                    canvas.erase_at(index_tip)
                    composited = canvas.draw_eraser_ring(composited, index_tip)

                fps = fps_counter.tick()
                draw_hud(
                    composited,
                    fps=fps,
                    gesture=active_gesture,
                    confidence=confidence,
                    brush_size=canvas.brush_size,
                    eraser_size=canvas.eraser_size,
                    brush_color_bgr=canvas.brush_color,
                    hand_count=len(hands),
                )

                # Get color name from lookup
                color_name = COLOR_LOOKUP.get(
                    canvas.brush_color,
                    "Custom"
                )

                # Map gesture to more professional display names
                gesture_display = {
                    "draw": "Drawing",
                    "erase": "Erasing",
                    "idle": "Idle",
                    "none": "None"
                }
                
                mode_display = {
                    "draw": "Paint Mode",
                    "erase": "Erase Mode",
                    "idle": "Standby",
                    "none": "Inactive"
                }
                
                gesture_text = gesture_display.get(active_gesture.value.lower(), active_gesture.value)
                mode_text = mode_display.get(active_gesture.value.lower(), active_gesture.value)

                dashboard.draw(
                    composited,
                    fps=int(fps),
                    gesture=gesture_text,
                    mode=mode_text,
                    brush=canvas.brush_size,
                    color_name=color_name,
                )

                help_overlay.draw(composited)

                notification.draw(composited)

                cv2.imshow(config.WINDOW_NAME, composited)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("h"):
                    help_overlay.toggle()

                elif key == config.KEY_QUIT:
                    logger.info("Quit requested by user.")
                    break

                elif key == config.KEY_CLEAR:
                    canvas.clear()

                    notification.show(
                        "Canvas Cleared",
                        (0, 0, 255)  # Red for destructive action
                    )

                elif key == config.KEY_SAVE:

                    try:
                        saved_path = canvas.save()

                        notification.show(
                            "Drawing Saved",
                            (0, 255, 0)  # Green for success
                        )

                        logger.info("Saved drawing to %s", saved_path)

                    except IOError as exc:
                        logger.error("Save failed: %s", exc)

                elif key in (config.KEY_BRUSH_INC, config.KEY_BRUSH_INC_ALT):

                    canvas.increase_brush()

                    notification.show(
                        f"Brush Size: {canvas.brush_size} px",
                        (255, 255, 0)  # Yellow
                    )

                elif key == config.KEY_BRUSH_DEC:

                    canvas.decrease_brush()

                    notification.show(
                        f"Brush Size: {canvas.brush_size} px",
                        (255, 255, 0)  # Yellow
                    )

                elif key == config.KEY_ERASER_INC:

                    canvas.increase_eraser()

                    notification.show(
                        f"Eraser Size: {canvas.eraser_size} px",
                        (255, 120, 0)  # Orange
                    )

                elif key == config.KEY_ERASER_DEC:

                    canvas.decrease_eraser()

                    notification.show(
                        f"Eraser Size: {canvas.eraser_size} px",
                        (255, 120, 0)  # Orange
                    )

                elif key == config.KEY_COLOR_NEXT:

                    canvas.cycle_color()

                    notification.show(
                        "Brush Color Changed",
                        (255, 150, 0)  # Gold
                    )

                # Also handle window-close (X button) gracefully.
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user.")
                    break

    except ModelNotFoundError as exc:
        logger.error("Fatal: %s", exc)
        print(f"ERROR: {exc}")
        return 1
    except CameraUnavailableError as exc:
        logger.error("Fatal: %s", exc)
        print(f"ERROR: {exc}\nCheck that a webcam is connected and not in use by another application.")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unhandled exception, shutting down safely: %s", exc)
        return 1
    finally:
        cv2.destroyAllWindows()
        logger.info("Application shut down. Camera released, no frames were persisted.")

    return 0


if __name__ == "__main__":
    sys.exit(run())