"""Logging + structured-output helpers (JSON / CSV) for reproducible results."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Union

PathLike = Union[str, os.PathLike]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "clearvad", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def write_json(data: Any, path: PathLike, indent: int = 2) -> Path:
    """Write any JSON-serializable object, creating parent dirs. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)
    return p


def write_csv(rows: Iterable[Mapping[str, Any]], path: PathLike,
              fieldnames: List[str] | None = None) -> Path:
    """Write a list of dict rows to CSV. Infers header from the first row if needed."""
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    if fieldnames is None:
        # union of keys, preserving first-seen order
        seen: Dict[str, None] = {}
        for r in rows:
            for k in r.keys():
                seen.setdefault(k, None)
        fieldnames = list(seen.keys())
    with open(p, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return p


def _json_default(obj: Any):
    """Make numpy scalars / arrays JSON serializable."""
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)
