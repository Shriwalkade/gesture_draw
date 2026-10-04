"""
audit.py
--------
Append-only, hash-chained audit log (JSON Lines).

WHAT IT GIVES YOU
    Tamper-EVIDENT history: every record embeds the SHA-256 of the previous
    one, so editing or deleting a line is detected by AuditLog.verify().

WHAT IT DOES NOT GIVE YOU
    * No identity. The gesture lock cannot know WHO the operator is, so the
      only actor field is a random per-run session id. Real "who did what"
      needs the badge/SSO unlock from the roadmap.
    * Not tamper-PROOF. Someone with file access can delete the whole file or
      rewrite the entire chain. Ship logs off-box for real assurance.
    * Records no frames and no hand landmarks. Only event names + ids.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


class AuditLog:
    def __init__(self, directory: str, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"audit_{self.session_id}.jsonl"
        self._seq = 0
        self._prev = GENESIS
        self._fh = open(self.path, "a", encoding="utf-8")

    def record(self, event: str, **data: Any) -> None:
        body = {
            "seq": self._seq,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z",
            "session": self.session_id,
            "event": event,
            "data": data,
        }
        h = _digest(self._prev, body)
        self._fh.write(json.dumps({**body, "prev": self._prev, "hash": h}, separators=(",", ":")) + "\n")
        self._fh.flush()
        self._prev, self._seq = h, self._seq + 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def verify(path: str) -> Tuple[bool, str]:
        prev, expected_seq = GENESIS, 0
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                try:
                    rec = json.loads(line)
                    body = {k: rec[k] for k in ("seq", "ts", "session", "event", "data")}
                    if rec["prev"] != prev:
                        return False, f"line {n}: chain broken (prev hash mismatch)"
                    if rec["seq"] != expected_seq:
                        return False, f"line {n}: sequence gap/reorder"
                    if _digest(prev, body) != rec["hash"]:
                        return False, f"line {n}: record content modified"
                    prev, expected_seq = rec["hash"], expected_seq + 1
                except (KeyError, ValueError):
                    return False, f"line {n}: malformed record"
        return True, f"{expected_seq} records, chain intact"
