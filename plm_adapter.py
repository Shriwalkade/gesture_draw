"""
plm_adapter.py
--------------
The seam between the gesture UI and whatever PLM/MES you actually run.

HONEST SCOPE
    * The CSV column schema below is MY invention. No real PLM (Teamcenter,
      Windchill, Aras, SAP...) exports this layout. A real connector means
      writing one PLMAdapter subclass that maps that system's API to Station.
    * Adapters are READ-ONLY by design: there is no write method, so no
      gesture can ever modify PLM data.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

VALID_STATUS = ("RUNNING", "IDLE", "FAULT")
CSV_COLUMNS = (
    "station_id", "name", "x_m", "y_m", "w_m", "h_m", "status", "work_order",
    "part_no", "rev_running", "rev_released", "bom_items", "wip",
)


@dataclass
class Station:
    sid: str
    name: str
    x: float          # metres, floor coords (centre)
    y: float
    w: float
    h: float
    status: str       # RUNNING | IDLE | FAULT
    work_order: str
    part_no: str
    rev_running: str  # revision actually loaded on the station
    rev_released: str # latest released revision in PLM
    bom_items: int
    wip: int
    home: Tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        self.home = (self.x, self.y)

    @property
    def rev_mismatch(self) -> bool:
        return self.rev_running != self.rev_released


class AdapterError(RuntimeError):
    """Raised when station data cannot be loaded or is malformed."""


class PLMAdapter(ABC):
    name: str = "adapter"

    @abstractmethod
    def load_stations(self) -> List[Station]:
        """Return a fresh list of stations. Must raise AdapterError on failure."""


class MockAdapter(PLMAdapter):
    """MOCK DATA -- demo only."""
    name = "MOCK"

    def load_stations(self) -> List[Station]:
        return [
            Station("CNC-01", "CNC Mill 1", 4, 3, 3.0, 2.0, "RUNNING", "WO-10451", "BRKT-2231", "C", "C", 4, 12),
            Station("CNC-02", "CNC Mill 2", 4, 8, 3.0, 2.0, "RUNNING", "WO-10452", "BRKT-2231", "B", "C", 4, 7),
            Station("ASM-A", "Assembly A", 11, 3, 4.0, 2.5, "RUNNING", "WO-10460", "GBOX-0078", "D", "D", 37, 5),
            Station("WLD-01", "Weld Cell", 11, 8.5, 3.0, 2.5, "FAULT", "WO-10463", "FRAME-5510", "A", "B", 18, 3),
            Station("QC-01", "Quality Gate", 18, 3, 2.5, 2.0, "IDLE", "-", "-", "-", "-", 0, 0),
            Station("PKG-01", "Packaging", 18, 8.5, 3.0, 2.0, "RUNNING", "WO-10470", "GBOX-0078", "D", "D", 2, 21),
        ]


class CsvAdapter(PLMAdapter):
    """Reads stations from a CSV export. Strict: any bad row aborts the load
    (a half-loaded shopfloor that silently drops stations is worse than an error)."""

    def __init__(self, path: str, floor_w_m: float = 24.0, floor_h_m: float = 12.0) -> None:
        self.path = Path(path)
        self.name = f"CSV:{self.path.name}"
        self.floor_w, self.floor_h = floor_w_m, floor_h_m

    def load_stations(self) -> List[Station]:
        try:
            with open(self.path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
                if missing:
                    raise AdapterError(f"{self.path.name}: missing columns {missing}; expected {list(CSV_COLUMNS)}")
                rows = list(reader)
        except OSError as exc:
            raise AdapterError(f"Cannot read {self.path}: {exc}") from exc

        stations: List[Station] = []
        seen = set()
        for n, row in enumerate(rows, start=2):  # row 1 is the header
            try:
                sid = row["station_id"].strip()
                if not sid or sid in seen:
                    raise ValueError("empty or duplicate station_id")
                seen.add(sid)
                status = row["status"].strip().upper()
                if status not in VALID_STATUS:
                    raise ValueError(f"status must be one of {VALID_STATUS}, got '{status}'")
                x, y, w, h = (float(row[k]) for k in ("x_m", "y_m", "w_m", "h_m"))
                if w <= 0 or h <= 0:
                    raise ValueError("w_m and h_m must be > 0")
                if not (0 <= x <= self.floor_w and 0 <= y <= self.floor_h):
                    raise ValueError(f"position ({x},{y}) outside floor {self.floor_w}x{self.floor_h} m")
                bom, wip = int(row["bom_items"]), int(row["wip"])
                if bom < 0 or wip < 0:
                    raise ValueError("bom_items and wip must be >= 0")
                stations.append(Station(
                    sid, row["name"].strip(), x, y, w, h, status,
                    row["work_order"].strip() or "-", row["part_no"].strip() or "-",
                    row["rev_running"].strip() or "-", row["rev_released"].strip() or "-", bom, wip,
                ))
            except (ValueError, KeyError, AttributeError) as exc:
                raise AdapterError(f"{self.path.name} row {n}: {exc}") from exc
        if not stations:
            raise AdapterError(f"{self.path.name}: no station rows")
        return stations
