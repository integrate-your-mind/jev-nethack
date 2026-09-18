"""Small, injectable local budget for recorder video backlog."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, NamedTuple
import stat

BACKLOG_CAP_BYTES = 128 * 1024 * 1024
RELEASE_OUTBOX_CAP_BYTES = 128 * 1024 * 1024
MINIMUM_FREE_BYTES = 64 * 1024 * 1024
ENCODER_RESERVATION_BYTES = 32 * 1024 * 1024
NEXT_SEGMENT_RESERVATION_BYTES = 32 * 1024 * 1024
NEXT_FRAME_HEADROOM_BYTES = 4 * 1024 * 1024


class Capacity(NamedTuple):
    backlog_bytes: int
    free_bytes: int
    reservation_bytes: int
    blocked: bool


def _regular_size(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    return info.st_size if stat.S_ISREG(info.st_mode) else 0


def measure_backlog_bytes(root: Path) -> int:
    """Count only regular MP4/PNG/segment JSONL files under broadcast-fragments."""
    root = Path(root)
    if root.name != "broadcast-fragments":
        root = root / "broadcast-fragments"
    total = 0
    if not root.is_dir():
        return 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames
                       if name not in {"training", "ttyrec"} and not (Path(directory) / name).is_symlink()]
        for name in filenames:
            path = Path(directory) / name
            if path.suffix == ".mp4" or path.suffix == ".png" or (path.suffix == ".jsonl" and path.name.startswith("segment-")):
                total += _regular_size(path)
    return total


def measure_release_outbox_bytes(root: Path) -> int:
    """Count regular tar/temp upload artifacts in the publisher outbox only."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return 0
    total = 0
    try:
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_file():
                continue
            if not (entry.name.endswith(".tar.gz") or entry.name.endswith((".tmp", ".part", ".partial"))):
                continue
            total += _regular_size(entry)
    except OSError:
        return total
    return total


def free_bytes(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def capacity(root: Path, *, inflight_jobs: int = 0,
             measure: Callable[[Path], int] = measure_backlog_bytes,
             free_measure: Callable[[Path], int] = free_bytes,
             backlog_cap: int = BACKLOG_CAP_BYTES,
             minimum_free: int = MINIMUM_FREE_BYTES) -> Capacity:
    jobs = max(0, int(inflight_jobs))
    reservation = jobs * ENCODER_RESERVATION_BYTES + NEXT_SEGMENT_RESERVATION_BYTES + NEXT_FRAME_HEADROOM_BYTES
    backlog = int(measure(Path(root)))
    available = int(free_measure(Path(root)))
    return Capacity(backlog, available, reservation,
                    backlog + reservation > backlog_cap or available < minimum_free + reservation)
