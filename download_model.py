"""
download_model.py
------------------
OPTIONAL. main.py downloads the model automatically on first run, so you
do not need to run this script for the app to work.

Use this only if you want to pre-fetch the model separately -- e.g. to
prepare an offline machine ahead of time by running this on a machine
with internet access, then copying the resulting `models/` folder over.

    python download_model.py
"""

from __future__ import annotations

import logging
import sys

import config
from hand_tracker import ModelNotFoundError, ensure_model_available


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("gesture_draw")
    try:
        ensure_model_available(config.MODEL_PATH, logger)
    except ModelNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model ready at {config.MODEL_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
