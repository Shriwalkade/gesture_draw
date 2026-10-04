"""
shopfloor.py
------------
Virtual-shopfloor PROTOTYPE: a faux-3D floor plan overlaid on the live camera
feed, driven by the locked owner's hand.

HONEST SCOPE
    * A webcam + overlay is a *prototype of the interaction pattern*, not real
      AR. Real AR (headset / tablet with spatial anchoring) is a different
      stack -- see README_PLM.md.
    * All station data below is MOCK. Replace `mock_stations()` with a
      connector to your PLM/MES (Teamcenter, Windchill, ...). Nothing here
      talks to a real system.

INTERACTIONS (all require the session lock to be ENGAGED; gesture-only, no keyboard needed)
    hover  : index fingertip over a station for SHOP_DWELL_SEC -> select
    pinch  : thumb+index pinch on a station, move, release -> what-if relocate
    buttons: dwell on LAYER / RESET / REFRESH / LOCK panel buttons (keys L,R remain as shortcuts)
What-if moves are never "committed" anywhere; there is deliberately no
destructive or write action reachable by gesture.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from plm_adapter import AdapterError, MockAdapter, PLMAdapter, Station  # noqa: F401 (Station re-exported)

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9

FLOOR_W_M, FLOOR_H_M = 24.0, 12.0   # floor size in metres
LAYERS = ["STATUS", "REVISION", "WIP"]

C_OK, C_WARN, C_BAD, C_IDLE = (80, 200, 80), (0, 170, 255), (60, 60, 230), (170, 170, 170)


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


class ShopfloorView:
    def __init__(self, frame_w: int, frame_h: int, adapter: Optional[PLMAdapter] = None) -> None:
        self.adapter: PLMAdapter = adapter or MockAdapter()
        self.stations = self.adapter.load_stations()
        self.status_msg = ""
        self._events: List[Tuple[str, Dict]] = []
        self._btn_fired = False
        self._grab_start: Tuple[float, float] = (0.0, 0.0)
        self.layer_idx = 0
        # Floor panel rectangle in frame pixels.
        self.px0, self.py0 = int(frame_w * 0.04), int(frame_h * 0.12)
        self.px1, self.py1 = int(frame_w * 0.70), int(frame_h * 0.93)   # right 26% reserved for the info card
        self.card_x = int(frame_w * 0.72)
        self.scale = min((self.px1 - self.px0) / FLOOR_W_M, (self.py1 - self.py0) / FLOOR_H_M)
        self.selected: Optional[Station] = None
        self._hover: Optional[Station] = None
        self._hover_since = 0.0
        self._pinching = False
        self._grab: Optional[Station] = None
        self._grab_offset = (0.0, 0.0)
        self.cursor: Optional[Tuple[float, float]] = None
        self.dwell_progress = 0.0
        bw, bh, gap, y = int(frame_w * 0.258), 44, 10, self.py0 + 200
        self.buttons = []  # (key, x0, y0, x1, y1)
        for i, key in enumerate(("LAYER", "RESET", "REFRESH", "LOCK")):
            self.buttons.append((key, self.card_x, y + i * (bh + gap), self.card_x + bw, y + i * (bh + gap) + bh))
        self._hover_is_button = False

    # ------------------------------------------------------- events / actions
    def pop_events(self) -> List[Tuple[str, Dict]]:
        ev, self._events = self._events, []
        return ev

    def _emit(self, name: str, **data) -> None:
        self._events.append((name, data))

    def refresh(self) -> bool:
        """Reload from the adapter. On failure keep the old data and say so."""
        try:
            fresh = self.adapter.load_stations()
        except AdapterError as exc:
            self.status_msg = f"Refresh failed: {exc}"[:110]
            self._emit("refresh_failed", error=str(exc)[:200])
            return False
        keep = self.selected.sid if self.selected else None
        self.stations = fresh
        self.selected = next((st for st in fresh if st.sid == keep), None)
        self._grab = None
        self.status_msg = f"Refreshed from {self.adapter.name} ({len(fresh)} stations; what-if cleared)"
        self._emit("refresh", source=self.adapter.name, stations=len(fresh))
        return True

    def _button_at(self, px: float, py: float) -> Optional[str]:
        for key, x0, y0, x1, y1 in self.buttons:
            if x0 <= px <= x1 and y0 <= py <= y1:
                return key
        return None

    def _fire(self, key: str) -> None:
        if key == "LAYER":
            self.cycle_layer()
        elif key == "RESET":
            self.reset_whatif()
        elif key == "REFRESH":
            self.refresh()
        elif key == "LOCK":
            self._emit("lock_request")
        if key in ("LAYER", "RESET"):
            self._emit("layer" if key == "LAYER" else "reset", **({"layer": self.layer} if key == "LAYER" else {}))

    # --------------------------------------------------------------- helpers
    @property
    def layer(self) -> str:
        return LAYERS[self.layer_idx]

    @property
    def interacting(self) -> bool:
        return self._grab is not None or self.dwell_progress > 0.0 or self._pinching

    def cycle_layer(self) -> None:
        self.layer_idx = (self.layer_idx + 1) % len(LAYERS)

    def reset_whatif(self) -> None:
        for s in self.stations:
            s.x, s.y = s.home
        self._grab = None

    @property
    def whatif_active(self) -> bool:
        return any(_dist((s.x, s.y), s.home) > 1e-6 for s in self.stations)

    def to_px(self, fx: float, fy: float) -> Tuple[int, int]:
        return int(self.px0 + fx * self.scale), int(self.py0 + fy * self.scale)

    def to_floor(self, px: float, py: float) -> Tuple[float, float]:
        return (px - self.px0) / self.scale, (py - self.py0) / self.scale

    def _station_at(self, px: float, py: float) -> Optional[Station]:
        fx, fy = self.to_floor(px, py)
        for s in reversed(self.stations):
            if abs(fx - s.x) <= s.w / 2 and abs(fy - s.y) <= s.h / 2:
                return s
        return None

    # ---------------------------------------------------------------- update
    def update(self, owner_hand, now: float) -> None:
        """Feed the owner's hand, or None when locked/paused (cancels everything)."""
        if owner_hand is None:
            self._cancel()
            return

        px = owner_hand.landmarks_px
        palm_px = max(_dist(px[WRIST], px[MIDDLE_MCP]), 1e-3)
        thumb, index = px[THUMB_TIP], px[INDEX_TIP]
        pinch_ratio = _dist(thumb, index) / palm_px

        # Hysteresis so a pinch doesn't flicker at the threshold.
        was_pinching = self._pinching
        if self._pinching:
            self._pinching = pinch_ratio < config.SHOP_PINCH_OFF
        else:
            self._pinching = pinch_ratio < config.SHOP_PINCH_ON

        self.cursor = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2) if self._pinching \
            else (float(index[0]), float(index[1]))

        if self._pinching:
            self._hover, self.dwell_progress = None, 0.0
            if not was_pinching:
                target = self._station_at(*self.cursor)
                if target is not None:
                    self._grab = target
                    self._grab_start = (target.x, target.y)
                    fx, fy = self.to_floor(*self.cursor)
                    self._grab_offset = (target.x - fx, target.y - fy)
                    self.selected = target
            if self._grab is not None:
                fx, fy = self.to_floor(*self.cursor)
                self._grab.x = float(np.clip(fx + self._grab_offset[0], 0, FLOOR_W_M))
                self._grab.y = float(np.clip(fy + self._grab_offset[1], 0, FLOOR_H_M))
            return

        self._end_grab()
        btn = self._button_at(*self.cursor)
        target = btn if btn is not None else self._station_at(*self.cursor)
        if target is not self._hover and target != self._hover:
            self._hover, self._hover_since, self._btn_fired = target, now, False
        if self._hover is None:
            self.dwell_progress = 0.0
        else:
            self.dwell_progress = min(1.0, (now - self._hover_since) / config.SHOP_DWELL_SEC)
            if self.dwell_progress >= 1.0:
                if isinstance(self._hover, str):          # a button: fire once per hover
                    if not self._btn_fired:
                        self._btn_fired = True
                        self._fire(self._hover)
                elif self.selected is not self._hover:
                    self.selected = self._hover
                    self._emit("select", station=self._hover.sid)

    def _end_grab(self) -> None:
        if self._grab is not None:
            moved = _dist((self._grab.x, self._grab.y), self._grab_start)
            if moved > 0.05:
                self._emit("whatif_move", station=self._grab.sid, x=round(self._grab.x, 2), y=round(self._grab.y, 2))
            self._grab = None

    def _cancel(self) -> None:
        self._end_grab()
        self._hover, self.dwell_progress, self._btn_fired = None, 0.0, False
        self._pinching, self.cursor = False, None

    # ------------------------------------------------------------------ draw
    def _color(self, s: Station) -> Tuple[int, int, int]:
        if self.layer == "STATUS":
            return {"RUNNING": C_OK, "IDLE": C_IDLE, "FAULT": C_BAD}[s.status]
        if self.layer == "REVISION":
            if s.rev_running == "-":
                return C_IDLE
            return C_WARN if s.rev_mismatch else C_OK
        # WIP: more WIP = hotter
        return C_BAD if s.wip >= 15 else C_WARN if s.wip >= 8 else C_OK

    def draw(self, frame: np.ndarray, locked: bool) -> np.ndarray:
        out = frame.copy()
        panel = out.copy()
        cv2.rectangle(panel, (self.px0, self.py0), (self.px1, self.py1), (30, 30, 30), -1)
        out = cv2.addWeighted(panel, 0.55, out, 0.45, 0)

        for gx in range(0, int(FLOOR_W_M) + 1, 2):  # floor grid
            cv2.line(out, self.to_px(gx, 0), self.to_px(gx, FLOOR_H_M), (70, 70, 70), 1)
        for gy in range(0, int(FLOOR_H_M) + 1, 2):
            cv2.line(out, self.to_px(0, gy), self.to_px(FLOOR_W_M, gy), (70, 70, 70), 1)

        for s in self.stations:
            self._draw_station(out, s)

        self._draw_header(out, locked)
        if self.selected is not None:
            self._draw_card(out, self.selected)
        self._draw_buttons(out, locked)
        if self.status_msg:
            cv2.putText(out, self.status_msg, (self.px0, self.py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if self.cursor is not None and not locked:
            self._draw_cursor(out)
        return out

    def _draw_station(self, out: np.ndarray, s: Station) -> None:
        color = self._color(s)
        x0, y0 = self.to_px(s.x - s.w / 2, s.y - s.h / 2)
        x1, y1 = self.to_px(s.x + s.w / 2, s.y + s.h / 2)
        lift = 24                                   # faux-3D extrusion height (px)
        top = np.array([[x0, y0 - lift], [x1, y0 - lift], [x1, y1 - lift], [x0, y1 - lift]], np.int32)
        side_r = np.array([[x1, y0 - lift], [x1, y0], [x1, y1], [x1, y1 - lift]], np.int32)
        front = np.array([[x0, y1 - lift], [x1, y1 - lift], [x1, y1], [x0, y1]], np.int32)
        dark = tuple(int(c * 0.55) for c in color)
        cv2.fillPoly(out, [front], dark)
        cv2.fillPoly(out, [side_r], tuple(int(c * 0.4) for c in color))
        cv2.fillPoly(out, [top], color)
        edge = (255, 255, 255) if s is self.selected else (20, 20, 20)
        cv2.polylines(out, [top], True, edge, 3 if s is self.selected else 1, cv2.LINE_AA)
        cv2.putText(out, s.sid, (x0 + 4, y0 - lift + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if self.layer == "REVISION" and s.rev_running != "-":
            tag = f"Rev {s.rev_running}" + (f" < {s.rev_released}" if s.rev_mismatch else "")
            cv2.putText(out, tag, (x0 + 4, y1 - lift - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        if self.layer == "WIP":
            cv2.putText(out, f"WIP {s.wip}", (x0 + 4, y1 - lift - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_header(self, out: np.ndarray, locked: bool) -> None:
        txt = f"VIRTUAL SHOPFLOOR (MOCK DATA)  |  Layer: {self.layer}  [L]  |  What-if: {'YES - not saved' if self.whatif_active else 'no'}  [R reset]"
        cv2.putText(out, txt, (self.px0, self.py0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if locked:
            ov = out.copy()
            cv2.rectangle(ov, (self.px0, self.py0), (self.px1, self.py1), (0, 0, 0), -1)
            out[:] = cv2.addWeighted(ov, 0.35, out, 0.65, 0)

    def _draw_card(self, out: np.ndarray, s: Station) -> None:
        lines = [
            f"{s.sid} - {s.name}", f"Status : {s.status}", f"WO     : {s.work_order}",
            f"Part   : {s.part_no}", f"Rev run: {s.rev_running}   Released: {s.rev_released}",
            f"BOM    : {s.bom_items} items   WIP: {s.wip}",
        ]
        if s.rev_mismatch and s.rev_running != "-":
            lines.append("!! Running a superseded revision")
        w, h = 330, 22 * len(lines) + 14
        x0, y0 = self.card_x, self.py0
        ov = out.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
        out[:] = cv2.addWeighted(ov, 0.8, out, 0.2, 0)
        for i, ln in enumerate(lines):
            col = C_WARN if ln.startswith("!!") else (255, 255, 255)
            cv2.putText(out, ln, (x0 + 10, y0 + 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    def _draw_buttons(self, out: np.ndarray, locked: bool) -> None:
        labels = {"LAYER": f"Layer: {self.layer}  (hover = next)", "RESET": "Reset what-if layout",
                  "REFRESH": f"Refresh from {self.adapter.name}", "LOCK": "Lock session"}
        for key, x0, y0, x1, y1 in self.buttons:
            ov = out.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), (50, 50, 50), -1)
            hot = (not locked) and self._hover == key
            if hot:  # dwell fill
                cv2.rectangle(ov, (x0, y0), (x0 + int((x1 - x0) * self.dwell_progress), y1), (0, 140, 0), -1)
            out[:] = cv2.addWeighted(ov, 0.85, out, 0.15, 0)
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255) if hot else (130, 130, 130), 1)
            cv2.putText(out, labels[key], (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_cursor(self, out: np.ndarray) -> None:
        cx, cy = int(self.cursor[0]), int(self.cursor[1])
        color = (0, 255, 255) if self._pinching else (255, 255, 255)
        cv2.circle(out, (cx, cy), 8, color, -1 if self._pinching else 2, cv2.LINE_AA)
        if self.dwell_progress > 0:
            cv2.ellipse(out, (cx, cy), (18, 18), -90, 0, int(360 * self.dwell_progress), (0, 220, 0), 3, cv2.LINE_AA)
