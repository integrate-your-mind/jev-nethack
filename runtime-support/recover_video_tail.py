#!/usr/bin/env python3
"""Recover the final recording segment of a proven-inactive broadcast.

The helper only derives an MP4 and a completed archive manifest from retained
PNG frames and the original segment JSONL.  It never posts a live frame or
deletes retained source data.  Callers can either hold the producer's
authoritative runtime lock or prove an exact mapping from an immutable old
broadcast directory to one dead producer process.  Exact-producer recovery
writes only to a separate publisher-owned derived directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
from typing import Any, BinaryIO, Callable, Iterator, Mapping, Sequence


_UNTIL_WIN = Path(__file__).resolve().parent.parent / "until-win"
if str(_UNTIL_WIN) not in sys.path:
    sys.path.insert(0, str(_UNTIL_WIN))

from broadcast import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    BroadcastError,
    probe_video,
    render_segment_ffmpeg,
    sha256_file,
)


MAX_JSONL_BYTES = 32 * 1024 * 1024
MAX_FRAME_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 10_000
MAX_SEGMENT_SPAN_SECONDS = 15 * 60
SESSION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SEGMENT_RE = re.compile(r"^segment-(\d{4})\.jsonl$")
BROADCAST_FRAGMENT_RE = re.compile(
    r"^broadcast-(\d{8}T\d{6}\.\d{6}Z)-([0-9a-f]{32})$"
)
RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
WRITER_KEYS = ("activeFragment", "active_fragment", "fragment", "path")


class TailRecoveryError(RuntimeError):
    """The source or inactivity proof is insufficient for safe recovery."""


class ActiveFragmentDeferred(TailRecoveryError):
    """The exact producer is current or alive and must be retried later."""


class NoRecoverableTail(TailRecoveryError):
    """The final producer segment contains no observed frame to recover."""


@dataclass(frozen=True)
class ActivityEvidence:
    """Explicit signals that are all required and all must prove inactivity.

    ``lock_files`` must contain the producer's authoritative exclusion lock,
    not a newly-created or unrelated lock selected only for recovery.
    """

    pids: tuple[int, ...] = ()
    pid_files: tuple[Path, ...] = ()
    lock_files: tuple[Path, ...] = ()
    current_writer_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ExactProducerEvidence:
    """Map an immutable old broadcast fragment to its dead writer process."""

    data_root: Path


@dataclass(frozen=True)
class ProducerIdentity:
    data_root: Path
    stamp: str
    pid: int
    stream_id: str
    accounting_dir: Path
    data_root_identity: DirectoryIdentity
    fragment_identity: DirectoryIdentity
    accounting_identity: DirectoryIdentity


@dataclass(frozen=True)
class DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class OutputLayout:
    fragment: Path
    separate: bool
    derived_root: Path | None
    derived_root_identity: DirectoryIdentity | None
    fragment_identity: DirectoryIdentity
    directory_fd: int | None


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class SegmentSource:
    events_path: Path
    frames: tuple[tuple[Path, float], ...]
    frame_snapshots: tuple[FileSnapshot, ...]
    events_snapshot: FileSnapshot
    broadcast_id: str
    started_at: str
    ended_at: str
    frame_count: int
    action_count: int
    first_sequence: int
    last_sequence: int


@dataclass(frozen=True)
class EventSummary:
    events_path: Path
    events_snapshot: FileSnapshot
    observed: tuple[Mapping[str, Any], ...]
    broadcast_id: str
    started_at: str
    ended_at: str
    frame_count: int
    action_count: int
    first_sequence: int
    last_sequence: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: Path, *, root: Path | None = None) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TailRecoveryError(f"required file is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TailRecoveryError(f"required path is not a regular non-symlink file: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TailRecoveryError(f"required file cannot be resolved: {path}") from exc
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise TailRecoveryError(f"source file escapes the fragment: {path}") from exc
    return resolved


def _real_directory(path: Path, *, root: Path | None = None) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TailRecoveryError(f"required directory is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise TailRecoveryError(f"required path is not a real non-symlink directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TailRecoveryError(f"required directory cannot be resolved: {path}") from exc
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise TailRecoveryError(f"directory escapes its allowed root: {path}") from exc
    return resolved


def _directory_identity(path: Path) -> DirectoryIdentity:
    resolved = _real_directory(path)
    metadata = resolved.stat()
    return DirectoryIdentity(metadata.st_dev, metadata.st_ino)


def _require_same_directory(path: Path, expected: DirectoryIdentity) -> None:
    if _directory_identity(path) != expected:
        raise TailRecoveryError(f"directory changed during recovery: {path}")


def _is_same_or_ancestor(candidate: Path, path: Path) -> bool:
    return candidate == path or candidate in path.parents


def _open_directory_fd(path: Path, expected: DirectoryIdentity) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TailRecoveryError(f"cannot safely open output directory: {path}") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or DirectoryIdentity(metadata.st_dev, metadata.st_ino) != expected
    ):
        os.close(descriptor)
        raise TailRecoveryError(f"output directory identity changed: {path}")
    return descriptor


@contextmanager
def _prepared_output(
    source_fragment: Path,
    *,
    output_fragment: Path | None,
    derived_root: Path | None,
    require_separate: bool,
    forbidden_runtime_root: Path | None = None,
) -> Iterator[OutputLayout]:
    source_fragment = _real_directory(source_fragment)
    if output_fragment is None:
        if derived_root is not None:
            raise TailRecoveryError("derived_root requires output_fragment")
        if require_separate:
            raise TailRecoveryError("exact-producer recovery requires output_fragment")
        yield OutputLayout(
            fragment=source_fragment,
            separate=False,
            derived_root=None,
            derived_root_identity=None,
            fragment_identity=_directory_identity(source_fragment),
            directory_fd=None,
        )
        return

    output_argument = Path(output_fragment).expanduser()
    if derived_root is None:
        if not output_argument.is_absolute():
            raise TailRecoveryError(
                "relative output_fragment requires an explicit derived_root"
            )
        derived_argument = output_argument.parent
    else:
        derived_argument = Path(derived_root).expanduser()
    resolved_root = _real_directory(derived_argument)
    if (
        _is_same_or_ancestor(resolved_root, source_fragment)
        or _is_same_or_ancestor(source_fragment, resolved_root)
    ):
        raise TailRecoveryError("derived output root must be disjoint from the source fragment")
    if forbidden_runtime_root is not None:
        runtime_root = _real_directory(forbidden_runtime_root)
        if (
            _is_same_or_ancestor(resolved_root, runtime_root)
            or _is_same_or_ancestor(runtime_root, resolved_root)
        ):
            raise TailRecoveryError(
                "publisher derived root must be disjoint from the runtime data root"
            )

    if output_argument.is_absolute():
        candidate = output_argument
        try:
            parent = candidate.parent.resolve(strict=True)
        except OSError as exc:
            raise TailRecoveryError("output_fragment parent is unavailable") from exc
        if parent != resolved_root:
            raise TailRecoveryError("output_fragment must be a direct child of derived_root")
    else:
        if len(output_argument.parts) != 1:
            raise TailRecoveryError("relative output_fragment must contain only one name")
        candidate = resolved_root / output_argument.name
    if FILE_RE.fullmatch(candidate.name) is None:
        raise TailRecoveryError("output_fragment has an unsafe name")

    root_identity = _directory_identity(resolved_root)
    root_fd = _open_directory_fd(resolved_root, root_identity)
    try:
        try:
            os.mkdir(candidate.name, mode=0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise TailRecoveryError("cannot create publisher output fragment") from exc
        output = _direct_real_child(
            candidate,
            resolved_root,
            kind="publisher output fragment",
        )
        output_identity = _directory_identity(output)
        output_fd = _open_directory_fd(output, output_identity)
    finally:
        os.close(root_fd)

    layout = OutputLayout(
        fragment=output,
        separate=True,
        derived_root=resolved_root,
        derived_root_identity=root_identity,
        fragment_identity=output_identity,
        directory_fd=output_fd,
    )
    try:
        _verify_output_layout(layout)
        yield layout
        _verify_output_layout(layout)
    finally:
        os.close(output_fd)


def _verify_output_layout(layout: OutputLayout) -> None:
    _require_same_directory(layout.fragment, layout.fragment_identity)
    if layout.derived_root is not None and layout.derived_root_identity is not None:
        _require_same_directory(layout.derived_root, layout.derived_root_identity)
        if layout.fragment.parent != layout.derived_root:
            raise TailRecoveryError("output fragment is no longer a direct child")
    if layout.directory_fd is not None:
        metadata = os.fstat(layout.directory_fd)
        identity = DirectoryIdentity(metadata.st_dev, metadata.st_ino)
        if not stat.S_ISDIR(metadata.st_mode) or identity != layout.fragment_identity:
            raise TailRecoveryError("open output directory identity changed")


def _snapshot(path: Path) -> FileSnapshot:
    path = _regular_file(path)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TailRecoveryError(f"file changed while it was being hashed: {path}")
    return FileSnapshot(before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, digest)


def _snapshot_optional(path: Path) -> FileSnapshot | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TailRecoveryError(f"cannot inspect derived output: {path}") from exc
    return _snapshot(path)


def _snapshot_optional_at(directory_fd: int, name: str) -> FileSnapshot | None:
    if Path(name).name != name or name in ("", ".", ".."):
        raise TailRecoveryError("derived output filename is unsafe")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TailRecoveryError(f"cannot inspect derived output safely: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TailRecoveryError(f"derived output is not a regular file: {name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TailRecoveryError(f"derived output changed while hashing: {name}")
    return FileSnapshot(
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        digest.hexdigest(),
    )


def _output_snapshot(path: Path, layout: OutputLayout) -> FileSnapshot | None:
    if path.parent != layout.fragment:
        raise TailRecoveryError("derived output is outside its output fragment")
    if layout.directory_fd is None:
        return _snapshot_optional(path)
    return _snapshot_optional_at(layout.directory_fd, path.name)


def _require_unchanged(path: Path, expected: FileSnapshot) -> None:
    if _snapshot(path) != expected:
        raise TailRecoveryError(f"source changed during recovery: {path}")


def _parse_json_file(path: Path) -> Any:
    path = _regular_file(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TailRecoveryError(f"cannot read evidence file: {path}") from exc
    if not raw or len(raw) > 1024 * 1024:
        raise TailRecoveryError(f"evidence file is empty or too large: {path}")
    try:
        return _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TailRecoveryError(f"evidence file is not valid JSON: {path}") from exc


def _read_stable_json_file(
    path: Path,
    *,
    root: Path | None = None,
) -> tuple[Any, FileSnapshot]:
    """Read one bounded JSON file through a non-following descriptor."""

    resolved = _regular_file(path, root=root)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise TailRecoveryError(f"cannot open evidence file safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TailRecoveryError(f"evidence path is not a regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, 1024 * 1024 + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 1024 * 1024:
                raise TailRecoveryError(f"evidence file is empty or too large: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise TailRecoveryError(f"evidence file changed while it was being read: {path}")
    raw = b"".join(chunks)
    if not raw:
        raise TailRecoveryError(f"evidence file is empty or too large: {path}")
    try:
        value = _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TailRecoveryError(f"evidence file is not valid JSON: {path}") from exc
    return value, FileSnapshot(
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        hashlib.sha256(raw).hexdigest(),
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_loads(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _pid_from_file(path: Path) -> int:
    path = _regular_file(path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise TailRecoveryError(f"cannot read PID evidence: {path}") from exc
    try:
        parsed = _strict_json_loads(text)
    except ValueError:
        parsed = text
    value = parsed.get("pid") if isinstance(parsed, Mapping) else parsed
    try:
        pid = int(value)
    except (TypeError, ValueError) as exc:
        raise TailRecoveryError(f"PID evidence is invalid: {path}") from exc
    if isinstance(value, bool) or pid <= 0 or str(pid) != str(value).strip():
        raise TailRecoveryError(f"PID evidence is invalid: {path}")
    return pid


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        raise TailRecoveryError("PID evidence must be a positive integer")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise TailRecoveryError(f"cannot determine whether PID {pid} is active") from exc
    return True


def _direct_real_child(path: Path, parent: Path, *, kind: str) -> Path:
    resolved_parent = _real_directory(parent)
    resolved = _real_directory(path, root=resolved_parent)
    if resolved.parent != resolved_parent:
        raise TailRecoveryError(f"{kind} must be a direct child of {resolved_parent}")
    return resolved


def _inspect_exact_producer(
    fragment: Path,
    evidence: ExactProducerEvidence,
) -> ProducerIdentity:
    data_root = _real_directory(evidence.data_root)
    broadcast_root = _direct_real_child(
        data_root / "broadcast-fragments",
        data_root,
        kind="broadcast root",
    )
    fragment = _direct_real_child(fragment, broadcast_root, kind="broadcast fragment")
    fragment_match = BROADCAST_FRAGMENT_RE.fullmatch(fragment.name)
    if fragment_match is None:
        raise TailRecoveryError(
            "broadcast fragment name must be broadcast-{stamp}-{32 lowercase hex}"
        )
    stamp, stream_id = fragment_match.groups()

    accounting_root = _direct_real_child(
        data_root / "jev-accounting",
        data_root,
        kind="accounting root",
    )
    process_re = re.compile(rf"^process-{re.escape(stamp)}-pid([1-9][0-9]*)$")
    matches: list[tuple[int, Path]] = []
    try:
        candidates = list(accounting_root.iterdir())
    except OSError as exc:
        raise TailRecoveryError("cannot enumerate exact producer accounting") from exc
    for candidate in candidates:
        match = process_re.fullmatch(candidate.name)
        if match is None:
            continue
        process_dir = _direct_real_child(
            candidate,
            accounting_root,
            kind="producer accounting directory",
        )
        pid_text = match.group(1)
        pid = int(pid_text)
        if str(pid) != pid_text:
            raise TailRecoveryError("producer accounting PID is not canonical")
        matches.append((pid, process_dir))
    if len(matches) != 1:
        raise TailRecoveryError(
            f"exact producer mapping requires one accounting directory; found {len(matches)}"
        )
    old_pid, accounting_dir = matches[0]
    if _pid_is_alive(old_pid):
        raise ActiveFragmentDeferred(f"broadcast producer PID is still active: {old_pid}")

    state_path = data_root / "state.json"
    state, _state_snapshot = _read_stable_json_file(state_path, root=data_root)
    if not isinstance(state, Mapping) or "activeFragment" not in state:
        raise TailRecoveryError("state.json must contain activeFragment")
    active_fragment = state["activeFragment"]
    if active_fragment is not None:
        if not isinstance(active_fragment, str) or not active_fragment:
            raise TailRecoveryError("state.activeFragment is invalid")
        active_path = Path(active_fragment)
        if (
            active_path.is_absolute()
            or len(active_path.parts) != 2
            or active_path.parts[0] != "fragments"
            or active_path.parts[1] in ("", ".", "..")
        ):
            raise TailRecoveryError("state.activeFragment is not a direct runtime fragment")
        fragments_root = _direct_real_child(
            data_root / "fragments",
            data_root,
            kind="runtime fragments root",
        )
        current_fragment = _direct_real_child(
            fragments_root / active_path.parts[1],
            fragments_root,
            kind="current runtime fragment",
        )
        config_path = current_fragment / "config.json"
        config, _config_snapshot = _read_stable_json_file(
            config_path,
            root=current_fragment,
        )
        current_pid = config.get("pid") if isinstance(config, Mapping) else None
        if (
            isinstance(current_pid, bool)
            or not isinstance(current_pid, int)
            or current_pid <= 0
        ):
            raise TailRecoveryError("current fragment config PID is invalid")
        if current_pid == old_pid:
            raise ActiveFragmentDeferred(
                "state.activeFragment still belongs to the broadcast producer"
            )

    return ProducerIdentity(
        data_root=data_root,
        stamp=stamp,
        pid=old_pid,
        stream_id=stream_id,
        accounting_dir=accounting_dir,
        data_root_identity=_directory_identity(data_root),
        fragment_identity=_directory_identity(fragment),
        accounting_identity=_directory_identity(accounting_dir),
    )


def inspect_exact_producer(
    fragment: Path,
    exact_producer: ExactProducerEvidence,
) -> ProducerIdentity:
    """Return the exact dead producer identity or raise a typed refusal."""

    return _inspect_exact_producer(fragment, exact_producer)


@contextmanager
def exact_producer_inactive(
    fragment: Path,
    evidence: ExactProducerEvidence,
) -> Iterator[tuple[Callable[[], None], ProducerIdentity]]:
    """Repeatedly prove one exact broadcast producer is no longer active."""

    initial = _inspect_exact_producer(fragment, evidence)

    def verify() -> None:
        current = _inspect_exact_producer(fragment, evidence)
        if current != initial:
            raise TailRecoveryError("exact producer mapping changed during recovery")

    yield verify, initial
    verify()


def _writer_reference(path: Path) -> str | None:
    value = _parse_json_file(path)
    if not isinstance(value, Mapping):
        raise TailRecoveryError(f"current-writer evidence must be a JSON object: {path}")
    for key in WRITER_KEYS:
        if key in value:
            writer = value[key]
            if writer is None:
                return None
            if not isinstance(writer, str) or not writer.strip():
                raise TailRecoveryError(f"current-writer reference is invalid: {path}")
            return writer.strip()
    raise TailRecoveryError(f"current-writer evidence has no recognized fragment field: {path}")


def _reference_names_fragment(reference: str, reference_file: Path, fragment: Path) -> bool:
    if reference == fragment.name or reference.rstrip("/").endswith("/" + fragment.name):
        return True
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = reference_file.parent / candidate
    candidate = candidate.resolve(strict=False)
    if candidate == fragment:
        return True
    try:
        candidate.relative_to(fragment)
        return True
    except ValueError:
        pass
    try:
        fragment.relative_to(candidate)
        return True
    except ValueError:
        return False


@contextmanager
def proven_inactive(fragment: Path, evidence: ActivityEvidence) -> Iterator[Callable[[], None]]:
    """Hold all supplied locks while every explicit activity signal is inactive."""

    if not evidence.lock_files:
        raise TailRecoveryError("at least one kernel lock file is required")
    if not evidence.pids and not evidence.pid_files:
        raise TailRecoveryError("at least one writer PID or PID file is required")
    if not evidence.current_writer_files:
        raise TailRecoveryError("at least one current-writer reference file is required")

    fragment = fragment.resolve(strict=True)
    held: list[BinaryIO] = []
    evidence_snapshots: list[tuple[Path, FileSnapshot]] = []
    try:
        for lock_path in sorted(set(evidence.lock_files), key=lambda item: str(item)):
            resolved = _regular_file(lock_path)
            stream = resolved.open("r+b")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.close()
                raise TailRecoveryError(f"writer lock is active: {resolved}") from exc
            held.append(stream)
            evidence_snapshots.append((resolved, _snapshot(resolved)))

        pids = list(evidence.pids)
        for path in evidence.pid_files:
            resolved = _regular_file(path)
            evidence_snapshots.append((resolved, _snapshot(resolved)))
            pids.append(_pid_from_file(resolved))
        for pid in pids:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise TailRecoveryError("PID evidence must contain positive integers")
            if _pid_is_alive(pid):
                raise TailRecoveryError(f"writer PID is still active: {pid}")

        for path in evidence.current_writer_files:
            resolved = _regular_file(path)
            evidence_snapshots.append((resolved, _snapshot(resolved)))
            reference = _writer_reference(resolved)
            if reference is not None and _reference_names_fragment(reference, resolved, fragment):
                raise TailRecoveryError(f"fragment is still named by current-writer evidence: {resolved}")

        def verify() -> None:
            for path, expected in evidence_snapshots:
                _require_unchanged(path, expected)
            for pid in pids:
                if _pid_is_alive(pid):
                    raise TailRecoveryError(f"writer PID became active during recovery: {pid}")
            for path in evidence.current_writer_files:
                resolved = _regular_file(path)
                reference = _writer_reference(resolved)
                if reference is not None and _reference_names_fragment(reference, resolved, fragment):
                    raise TailRecoveryError(f"fragment became active during recovery: {resolved}")

        yield verify
        verify()
    finally:
        for stream in reversed(held):
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


def _parse_time(value: Any, *, field: str) -> datetime:
    # datetime.fromisoformat accepts ISO week/basic dates and second-resolution
    # UTC offsets that JavaScript Date.parse rejects.  The Site validates with
    # Date.parse, so accept only the shared RFC 3339 subset emitted by utc_now.
    if not isinstance(value, str) or RFC3339_RE.fullmatch(value) is None:
        raise TailRecoveryError(f"{field} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TailRecoveryError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise TailRecoveryError(f"{field} must include a timezone")
    return parsed


def _validate_png(path: Path) -> None:
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise TailRecoveryError(f"retained frame has the wrong format or dimensions: {path}")
            image.verify()
    except TailRecoveryError:
        raise
    except (OSError, ValueError) as exc:
        raise TailRecoveryError(f"retained frame is not a valid PNG: {path}") from exc


def _load_event_summary(fragment: Path, events_path: Path) -> EventSummary:
    fragment = fragment.resolve(strict=True)
    events_path = _regular_file(events_path, root=fragment)
    segment_match = SEGMENT_RE.fullmatch(events_path.name)
    if segment_match is None or int(segment_match.group(1)) <= 0:
        raise TailRecoveryError("segment JSONL filename is invalid")
    _fsync_file(events_path)
    events_snapshot = _snapshot(events_path)
    if events_snapshot.size == 0:
        raise NoRecoverableTail("final segment has zero observed frames")
    if events_snapshot.size > MAX_JSONL_BYTES:
        raise TailRecoveryError("segment JSONL exceeds the 32 MiB archive limit")
    raw = events_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise TailRecoveryError("segment JSONL is not durably closed with a trailing newline")

    observed: list[Mapping[str, Any]] = []
    broadcast_id: str | None = None
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise TailRecoveryError(f"segment JSONL contains a blank line at {number}")
        try:
            event = _strict_json_loads(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise TailRecoveryError(f"segment JSONL is invalid at line {number}") from exc
        if not isinstance(event, Mapping):
            raise TailRecoveryError(f"segment JSONL event {number} is not an object")
        stream_id = event.get("stream_id")
        if stream_id is not None:
            if not isinstance(stream_id, str) or not SESSION_RE.fullmatch(stream_id):
                raise TailRecoveryError(f"segment JSONL has an invalid stream_id at line {number}")
            if broadcast_id is None:
                broadcast_id = stream_id
            elif broadcast_id != stream_id:
                raise TailRecoveryError("segment JSONL mixes broadcast stream IDs")
        if event.get("event_type") == "observed_frame":
            if stream_id is None:
                raise TailRecoveryError(f"observed frame has no stream_id at line {number}")
            observed.append(event)

    if not observed:
        raise NoRecoverableTail("final segment has zero observed frames")
    if broadcast_id is None:
        raise TailRecoveryError("segment JSONL has no stream ID")
    if len(observed) > MAX_FRAMES:
        raise TailRecoveryError("segment has too many retained frames")

    previous_sequence = -1
    previous_epoch = -math.inf
    previous_time: datetime | None = None
    for event in observed:
        sequence = event.get("sequence")
        epoch = event.get("wall_epoch")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0 or sequence <= previous_sequence:
            raise TailRecoveryError("observed frame sequences must be strictly increasing non-negative integers")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(float(epoch)) or float(epoch) < previous_epoch:
            raise TailRecoveryError("observed frame epochs must be finite and monotonic")
        timestamp = _parse_time(event.get("wall_time"), field="wall_time")
        if previous_time is not None and timestamp < previous_time:
            raise TailRecoveryError("observed frame wall_time values must be monotonic")
        if abs(timestamp.timestamp() - float(epoch)) > 5:
            raise TailRecoveryError("observed frame epoch and wall_time disagree")
        previous_sequence = sequence
        previous_epoch = float(epoch)
        previous_time = timestamp

    if float(observed[-1]["wall_epoch"]) - float(observed[0]["wall_epoch"]) > MAX_SEGMENT_SPAN_SECONDS:
        raise TailRecoveryError("segment duration exceeds the bounded recovery limit")

    action_count = sum(
        1 for event in observed
        if event.get("phase") == "after_action" and event.get("action") is not None
    )
    return EventSummary(
        events_path=events_path,
        events_snapshot=events_snapshot,
        observed=tuple(observed),
        broadcast_id=broadcast_id,
        started_at=str(observed[0]["wall_time"]),
        ended_at=str(observed[-1]["wall_time"]),
        frame_count=len(observed),
        action_count=action_count,
        first_sequence=int(observed[0]["sequence"]),
        last_sequence=int(observed[-1]["sequence"]),
    )


def _load_source(fragment: Path, events_path: Path) -> SegmentSource:
    fragment = fragment.resolve(strict=True)
    summary = _load_event_summary(fragment, events_path)
    segment_match = SEGMENT_RE.fullmatch(summary.events_path.name)
    assert segment_match is not None
    # BroadcastRecorder names retained frames with its current zero-based
    # segment_index, then increments that index before naming the finalized
    # JSONL/MP4.  Bind to that exact producer relationship; a wildcard prefix
    # could otherwise pair a valid event sequence with the wrong image.
    frame_prefix = f"segment-{int(segment_match.group(1)) - 1:04d}"
    frames_root = fragment / ".frames"
    if not frames_root.is_dir() or frames_root.is_symlink():
        raise TailRecoveryError("retained frame directory is missing or unsafe")
    frames: list[tuple[Path, float]] = []
    snapshots: list[FileSnapshot] = []
    total_frame_bytes = 0
    for event in summary.observed:
        sequence = int(event["sequence"])
        epoch = float(event["wall_epoch"])
        expected_frame = frames_root / f"{frame_prefix}-frame-{sequence:08d}.png"
        if not expected_frame.exists():
            raise TailRecoveryError(f"expected retained PNG is missing for sequence {sequence}")
        frame = _regular_file(expected_frame, root=fragment)
        _validate_png(frame)
        _fsync_file(frame)
        snapshot = _snapshot(frame)
        total_frame_bytes += snapshot.size
        if total_frame_bytes > MAX_FRAME_BYTES:
            raise TailRecoveryError("retained frames exceed the bounded recovery limit")
        frames.append((frame, epoch))
        snapshots.append(snapshot)

    return SegmentSource(
        events_path=summary.events_path,
        frames=tuple(frames),
        frame_snapshots=tuple(snapshots),
        events_snapshot=summary.events_snapshot,
        broadcast_id=summary.broadcast_id,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        frame_count=summary.frame_count,
        action_count=summary.action_count,
        first_sequence=summary.first_sequence,
        last_sequence=summary.last_sequence,
    )


def _source_unchanged(source: SegmentSource) -> None:
    _require_unchanged(source.events_path, source.events_snapshot)
    for (path, _timestamp), expected in zip(source.frames, source.frame_snapshots):
        _require_unchanged(path, expected)


def _video_is_valid(path: Path) -> bool:
    try:
        _regular_file(path)
        probe_video(path)
    except (TailRecoveryError, BroadcastError):
        return False
    return True


def _site_manifest(path: Path, *, mp4_name: str, jsonl_name: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        manifest = _parse_json_file(path)
        schema_version = manifest.get("schemaVersion") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict) or isinstance(schema_version, bool)
                or not isinstance(schema_version, int) or schema_version != 1
                or manifest.get("completed") is not True):
            return None
        session_id = manifest.get("sessionId")
        if not isinstance(session_id, str) or not SESSION_RE.fullmatch(session_id):
            return None
        broadcast_id = manifest.get("broadcastId")
        if broadcast_id is not None and (not isinstance(broadcast_id, str) or not SESSION_RE.fullmatch(broadcast_id)):
            return None
        if not isinstance(manifest.get("frameCount"), int) or isinstance(manifest.get("frameCount"), bool) or manifest["frameCount"] < 0:
            return None
        if not isinstance(manifest.get("actionCount"), int) or isinstance(manifest.get("actionCount"), bool) or manifest["actionCount"] < 0:
            return None
        started = _parse_time(manifest.get("startedAt"), field="startedAt")
        ended = _parse_time(manifest.get("endedAt"), field="endedAt")
        if ended < started:
            return None
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not 2 <= len(artifacts) <= 16:
            return None
        by_name: dict[str, Mapping[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("filename"), str):
                return None
            name = artifact["filename"]
            if not FILE_RE.fullmatch(name) or name == "manifest.json" or name in by_name:
                return None
            by_name[name] = artifact
        if mp4_name not in by_name or jsonl_name not in by_name:
            return None
        for name, artifact in by_name.items():
            if name.endswith(".mp4"):
                content_type = "video/mp4"
            elif name.endswith(".jsonl"):
                content_type = "application/x-ndjson"
            else:
                return None
            artifact_path = path.parent / name
            size = artifact.get("bytes")
            if artifact.get("contentType") != content_type or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                return None
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                return None
            snapshot = _snapshot(artifact_path)
            if snapshot.size != artifact["bytes"] or snapshot.sha256 != digest:
                return None
        if not _video_is_valid(path.parent / mp4_name):
            return None
        return manifest
    except (OSError, TailRecoveryError, BroadcastError, ValueError):
        return None


def _manifest_matches_events(manifest: Mapping[str, Any], summary: EventSummary, index: int) -> bool:
    expected_session = f"broadcast-{summary.broadcast_id}-seg{index:04d}"
    if (manifest.get("sessionId") != expected_session
            or manifest.get("broadcastId") != summary.broadcast_id
            or manifest.get("startedAt") != summary.started_at
            or manifest.get("endedAt") != summary.ended_at
            or manifest.get("frameCount") != summary.frame_count
            or manifest.get("actionCount") != summary.action_count):
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    event_artifact = next(
        (
            artifact for artifact in artifacts
            if isinstance(artifact, Mapping) and artifact.get("filename") == summary.events_path.name
        ),
        None,
    )
    if (event_artifact is None
            or event_artifact.get("bytes") != summary.events_snapshot.size
            or event_artifact.get("sha256") != summary.events_snapshot.sha256):
        return False
    for key, expected in (
        ("firstSequence", summary.first_sequence),
        ("lastSequence", summary.last_sequence),
    ):
        value = manifest.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value != expected):
            return False
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_temp(path: Path, data: bytes) -> Path:
    temporary = path.with_name(f".{path.stem}.recovery-{secrets.token_hex(8)}{path.suffix}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_output_temp(path: Path, data: bytes, layout: OutputLayout) -> Path:
    if layout.directory_fd is None:
        return _write_temp(path, data)
    if path.parent != layout.fragment:
        raise TailRecoveryError("temporary output target is outside its output fragment")
    temporary = path.with_name(
        f".{path.stem}.recovery-{secrets.token_hex(8)}{path.suffix}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            temporary.name,
            flags,
            0o600,
            dir_fd=layout.directory_fd,
        )
    except OSError as exc:
        raise TailRecoveryError("cannot create temporary derived output") from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TailRecoveryError("could not write temporary derived output")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary.name, dir_fd=layout.directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    return temporary


def _remove_output_temp(path: Path, layout: OutputLayout) -> None:
    if layout.directory_fd is None:
        path.unlink(missing_ok=True)
        return
    try:
        os.unlink(path.name, dir_fd=layout.directory_fd)
    except FileNotFoundError:
        pass


def _commit_temp(
    temporary: Path,
    target: Path,
    expected: FileSnapshot | None,
    *,
    precommit: Callable[[], None] | None = None,
    directory_fd: int | None = None,
) -> None:
    if precommit is not None:
        precommit()
    if directory_fd is None:
        current = _snapshot_optional(target)
    else:
        if temporary.parent != target.parent:
            raise TailRecoveryError("temporary output is outside the target directory")
        current = _snapshot_optional_at(directory_fd, target.name)
    if current != expected:
        raise TailRecoveryError(f"derived output changed during recovery: {target}")
    if precommit is not None:
        precommit()
    if expected is None:
        try:
            if directory_fd is None:
                os.link(temporary, target, follow_symlinks=False)
                temporary.unlink()
                _fsync_directory(target.parent)
            else:
                os.link(
                    temporary.name,
                    target.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                os.unlink(temporary.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
        except FileExistsError as exc:
            raise TailRecoveryError(
                f"derived output appeared during recovery: {target}"
            ) from exc
    elif directory_fd is None:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    else:
        os.replace(
            temporary.name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)


def _segment_index(fragment: Path, requested: int | None) -> int:
    candidates: list[int] = []
    for path in fragment.glob("segment-*.jsonl"):
        match = SEGMENT_RE.fullmatch(path.name)
        if match:
            candidates.append(int(match.group(1)))
    if not candidates:
        raise TailRecoveryError("fragment has no segment JSONL")
    index = max(candidates) if requested is None else requested
    if index <= 0 or index > 9999 or index not in candidates:
        raise TailRecoveryError("requested recovery segment does not exist")
    if any(candidate > index for candidate in candidates):
        raise TailRecoveryError("recovery is only allowed for the final segment JSONL")
    return index


@contextmanager
def _inactivity_guard(
    fragment: Path,
    *,
    evidence: ActivityEvidence | None,
    exact_producer: ExactProducerEvidence | None,
) -> Iterator[tuple[Callable[[], None], ProducerIdentity | None]]:
    if (evidence is None) == (exact_producer is None):
        raise TailRecoveryError(
            "supply exactly one inactivity mode: evidence or exact_producer"
        )
    if exact_producer is not None:
        with exact_producer_inactive(fragment, exact_producer) as guarded:
            yield guarded
        return
    assert evidence is not None
    with proven_inactive(fragment, evidence) as verify:
        yield verify, None


def _copy_event_stream(
    source: SegmentSource,
    target: Path,
    layout: OutputLayout,
    *,
    ensure_stable: Callable[[], None],
) -> tuple[FileSnapshot, bool]:
    if not layout.separate:
        if target != source.events_path:
            raise TailRecoveryError("in-place event target does not match the source")
        return source.events_snapshot, False

    initial = _output_snapshot(target, layout)
    if initial is not None:
        if (
            initial.size != source.events_snapshot.size
            or initial.sha256 != source.events_snapshot.sha256
        ):
            raise TailRecoveryError(
                "refusing to overwrite an existing non-matching segment JSONL"
            )
        ensure_stable()
        return initial, False

    raw = source.events_path.read_bytes()
    if (
        len(raw) != source.events_snapshot.size
        or hashlib.sha256(raw).hexdigest() != source.events_snapshot.sha256
    ):
        raise TailRecoveryError("source JSONL changed while preparing its exact copy")
    temporary = _write_output_temp(target, raw, layout)
    try:
        temporary_snapshot = (
            _snapshot(temporary)
            if layout.directory_fd is None
            else _snapshot_optional_at(layout.directory_fd, temporary.name)
        )
        if (
            temporary_snapshot is None
            or temporary_snapshot.size != source.events_snapshot.size
            or temporary_snapshot.sha256 != source.events_snapshot.sha256
        ):
            raise TailRecoveryError("temporary JSONL copy does not match the source")
        ensure_stable()
        _commit_temp(
            temporary,
            target,
            None,
            precommit=ensure_stable,
            directory_fd=layout.directory_fd,
        )
        installed = _output_snapshot(target, layout)
        if (
            installed is None
            or installed.size != source.events_snapshot.size
            or installed.sha256 != source.events_snapshot.sha256
        ):
            raise TailRecoveryError("installed JSONL copy does not match the source")
        ensure_stable()
        return installed, True
    finally:
        _remove_output_temp(temporary, layout)


def recover_video_tail(
    fragment: Path,
    *,
    evidence: ActivityEvidence | None = None,
    exact_producer: ExactProducerEvidence | None = None,
    output_fragment: Path | None = None,
    derived_root: Path | None = None,
    segment_index: int | None = None,
    render: Callable[[list[tuple[Path, float]], Path], None] = render_segment_ffmpeg,
    probe: Callable[[Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Recover one final segment and return a local, non-uploaded receipt."""

    try:
        fragment = _real_directory(fragment)
    except OSError as exc:
        raise TailRecoveryError("fragment does not exist") from exc

    with _inactivity_guard(
        fragment,
        evidence=evidence,
        exact_producer=exact_producer,
    ) as (verify_inactive, producer):
        index = _segment_index(fragment, segment_index)
        stem = f"segment-{index:04d}"
        source_events_path = fragment / f"{stem}.jsonl"
        summary = _load_event_summary(fragment, source_events_path)
        if producer is not None and summary.broadcast_id != producer.stream_id:
            raise TailRecoveryError(
                "segment stream_id does not match the broadcast fragment identity"
            )
        verify_inactive()
        with _prepared_output(
            fragment,
            output_fragment=output_fragment,
            derived_root=derived_root,
            require_separate=producer is not None,
            forbidden_runtime_root=producer.data_root if producer is not None else None,
        ) as output:
            output_events_path = output.fragment / f"{stem}.jsonl"
            mp4_path = output.fragment / f"{stem}.mp4"
            manifest_path = output.fragment / f"{stem}.manifest.json"

            _verify_output_layout(output)
            initial_manifest = _output_snapshot(manifest_path, output)

            existing = _site_manifest(
                manifest_path,
                mp4_name=mp4_path.name,
                jsonl_name=output_events_path.name,
            )
            if existing is not None:
                if not _manifest_matches_events(existing, summary, index):
                    raise TailRecoveryError(
                        "existing completed manifest does not match the original event stream"
                    )
                _require_unchanged(summary.events_path, summary.events_snapshot)
                verify_inactive()
                _verify_output_layout(output)
                rechecked = _site_manifest(
                    manifest_path,
                    mp4_name=mp4_path.name,
                    jsonl_name=output_events_path.name,
                )
                if rechecked != existing:
                    raise TailRecoveryError(
                        "existing completed segment changed during validation"
                    )
                rechecked_snapshot = _output_snapshot(manifest_path, output)
                if (
                    initial_manifest is None
                    or rechecked_snapshot != initial_manifest
                ):
                    raise TailRecoveryError(
                        "existing completed manifest bytes changed during validation"
                    )
                return {
                    "status": "already_complete",
                    "manifest": rechecked,
                    "manifestPath": str(manifest_path),
                    "manifestSha256": rechecked_snapshot.sha256,
                    "sourceFragment": str(fragment),
                    "outputFragment": str(output.fragment),
                    "sourceUnmodifiedByRecovery": output.separate,
                }
            if output.separate and initial_manifest is not None:
                raise TailRecoveryError(
                    "refusing to replace an existing invalid derived manifest "
                    "without an authoritative writer lock"
                )

            initial_mp4 = _output_snapshot(mp4_path, output)
            existing_mp4_valid = initial_mp4 is not None and _video_is_valid(mp4_path)
            if output.separate and initial_mp4 is not None and not existing_mp4_valid:
                raise TailRecoveryError(
                    "refusing to replace an existing invalid derived MP4 "
                    "without an authoritative writer lock"
                )
            source = _load_source(fragment, source_events_path)
            if producer is not None and source.broadcast_id != producer.stream_id:
                raise TailRecoveryError(
                    "segment stream_id does not match the broadcast fragment identity"
                )
            session_id = f"broadcast-{source.broadcast_id}-seg{index:04d}"
            if not SESSION_RE.fullmatch(session_id):
                raise TailRecoveryError("derived segment session ID is invalid for the Site")

            def ensure_stable() -> None:
                _source_unchanged(source)
                verify_inactive()
                _verify_output_layout(output)

            output_events_snapshot, copied_jsonl = _copy_event_stream(
                source,
                output_events_path,
                output,
                ensure_stable=ensure_stable,
            )
            temporary_mp4: Path | None = None
            temporary_manifest: Path | None = None
            installed_mp4 = False
            reused_mp4 = False
            try:
                ensure_stable()
                with tempfile.TemporaryDirectory(
                    prefix="jev-video-tail-recovery-"
                ) as staging_name:
                    staged_mp4 = Path(staging_name) / "recovered.mp4"
                    render(list(source.frames), staged_mp4)
                    if not staged_mp4.exists():
                        raise TailRecoveryError("renderer did not produce an MP4")
                    _fsync_file(staged_mp4)
                    probe(staged_mp4)
                    rendered = _snapshot(staged_mp4)
                    if rendered.size <= 0 or rendered.size > MAX_JSONL_BYTES:
                        raise TailRecoveryError(
                            "recovered MP4 is empty or exceeds the 32 MiB archive limit"
                        )
                    rendered_bytes = staged_mp4.read_bytes()
                    if (
                        len(rendered_bytes) != rendered.size
                        or hashlib.sha256(rendered_bytes).hexdigest()
                        != rendered.sha256
                    ):
                        raise TailRecoveryError(
                            "rendered MP4 changed while copying from private staging"
                        )
                    ensure_stable()
                    temporary_mp4 = _write_output_temp(
                        mp4_path,
                        rendered_bytes,
                        output,
                    )
                    copied_render = (
                        _snapshot(temporary_mp4)
                        if output.directory_fd is None
                        else _snapshot_optional_at(
                            output.directory_fd,
                            temporary_mp4.name,
                        )
                    )
                    if (
                        copied_render is None
                        or copied_render.size != rendered.size
                        or copied_render.sha256 != rendered.sha256
                    ):
                        raise TailRecoveryError(
                            "publisher MP4 temporary copy does not match the render"
                        )
                assert temporary_mp4 is not None
                ensure_stable()

                current_mp4 = _output_snapshot(mp4_path, output)
                if current_mp4 != initial_mp4:
                    raise TailRecoveryError("MP4 changed while recovery was rendering")
                if existing_mp4_valid:
                    if (
                        current_mp4 is None
                        or current_mp4.size != rendered.size
                        or current_mp4.sha256 != rendered.sha256
                    ):
                        raise TailRecoveryError(
                            "refusing to overwrite an existing valid MP4 without matching bytes"
                        )
                    _remove_output_temp(temporary_mp4, output)
                    reused_mp4 = True
                else:
                    _commit_temp(
                        temporary_mp4,
                        mp4_path,
                        initial_mp4,
                        precommit=ensure_stable,
                        directory_fd=output.directory_fd,
                    )
                    installed_mp4 = True
                ensure_stable()
                installed_snapshot = _output_snapshot(mp4_path, output)
                if installed_snapshot is None:
                    raise TailRecoveryError("installed MP4 is unavailable")
                probe(mp4_path)
                ensure_stable()
                if (
                    installed_snapshot.size != rendered.size
                    or installed_snapshot.sha256 != rendered.sha256
                ):
                    raise TailRecoveryError(
                        "installed MP4 does not match the validated render"
                    )

                artifacts = [
                    {
                        "filename": mp4_path.name,
                        "sha256": installed_snapshot.sha256,
                        "bytes": installed_snapshot.size,
                        "contentType": "video/mp4",
                    },
                    {
                        "filename": output_events_path.name,
                        "sha256": output_events_snapshot.sha256,
                        "bytes": output_events_snapshot.size,
                        "contentType": "application/x-ndjson",
                    },
                ]
                manifest = {
                    "schemaVersion": 1,
                    "sessionId": session_id,
                    "broadcastId": source.broadcast_id,
                    "completed": True,
                    "startedAt": source.started_at,
                    "endedAt": source.ended_at,
                    "frameCount": source.frame_count,
                    "actionCount": source.action_count,
                    "recoveredAfterCrash": True,
                    "recoveredAt": utc_now(),
                    "gameOutcome": "interrupted",
                    "firstSequence": source.first_sequence,
                    "lastSequence": source.last_sequence,
                    "artifacts": artifacts,
                }
                manifest_bytes = json.dumps(
                    manifest,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                temporary_manifest = _write_output_temp(
                    manifest_path,
                    manifest_bytes,
                    output,
                )
                ensure_stable()
                probe(mp4_path)
                if _output_snapshot(mp4_path, output) != installed_snapshot:
                    raise TailRecoveryError(
                        "installed MP4 changed before manifest publication"
                    )
                concurrently_completed = _site_manifest(
                    manifest_path,
                    mp4_name=mp4_path.name,
                    jsonl_name=output_events_path.name,
                )
                if concurrently_completed is not None:
                    if not _manifest_matches_events(
                        concurrently_completed,
                        summary,
                        index,
                    ):
                        raise TailRecoveryError(
                            "concurrent completed manifest does not match the original event stream"
                        )
                    _remove_output_temp(temporary_manifest, output)
                    temporary_manifest = None
                    ensure_stable()
                    concurrent_snapshot = _output_snapshot(manifest_path, output)
                    if concurrent_snapshot is None:
                        raise TailRecoveryError(
                            "concurrent completed manifest disappeared"
                        )
                    return {
                        "status": "already_complete",
                        "manifest": concurrently_completed,
                        "manifestPath": str(manifest_path),
                        "manifestSha256": concurrent_snapshot.sha256,
                        "sourceFragment": str(fragment),
                        "outputFragment": str(output.fragment),
                        "sourceUnmodifiedByRecovery": output.separate,
                    }
                ensure_stable()
                _commit_temp(
                    temporary_manifest,
                    manifest_path,
                    initial_manifest,
                    precommit=ensure_stable,
                    directory_fd=output.directory_fd,
                )
                temporary_manifest = None
                ensure_stable()
                verified = _site_manifest(
                    manifest_path,
                    mp4_name=mp4_path.name,
                    jsonl_name=output_events_path.name,
                )
                if verified is None:
                    raise TailRecoveryError(
                        "published recovery manifest failed local Site validation"
                    )
                verified_manifest_snapshot = _output_snapshot(manifest_path, output)
                if verified_manifest_snapshot is None:
                    raise TailRecoveryError(
                        "published recovery manifest disappeared after validation"
                    )
                return {
                    "status": "recovered",
                    "manifest": verified,
                    "manifestPath": str(manifest_path),
                    "manifestSha256": verified_manifest_snapshot.sha256,
                    "installedMp4": installed_mp4,
                    "reusedMatchingMp4": reused_mp4,
                    "copiedJsonl": copied_jsonl,
                    "originalFramesRetained": True,
                    "sourceFragment": str(fragment),
                    "outputFragment": str(output.fragment),
                    "sourceUnmodifiedByRecovery": output.separate,
                    "uploaded": False,
                }
            except BroadcastError as exc:
                raise TailRecoveryError(str(exc)) from exc
            finally:
                if temporary_mp4 is not None:
                    _remove_output_temp(temporary_mp4, output)
                if temporary_manifest is not None:
                    _remove_output_temp(temporary_manifest, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--segment-index", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--derived-root", type=Path)
    parser.add_argument("--output-fragment", type=Path)
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--pid-file", action="append", type=Path, default=[])
    parser.add_argument("--lock-file", action="append", type=Path, default=[])
    parser.add_argument("--current-writer-file", action="append", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    legacy_supplied = bool(
        args.pid or args.pid_file or args.lock_file or args.current_writer_file
    )
    if args.data_root is not None and legacy_supplied:
        print(
            json.dumps(
                {
                    "status": "refused",
                    "error": "--data-root cannot be combined with legacy evidence flags",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    evidence = None
    exact_producer = None
    if args.data_root is not None:
        exact_producer = ExactProducerEvidence(args.data_root)
    else:
        evidence = ActivityEvidence(
            pids=tuple(args.pid),
            pid_files=tuple(args.pid_file),
            lock_files=tuple(args.lock_file),
            current_writer_files=tuple(args.current_writer_file),
        )
    try:
        result = recover_video_tail(
            args.fragment,
            evidence=evidence,
            exact_producer=exact_producer,
            output_fragment=args.output_fragment,
            derived_root=args.derived_root,
            segment_index=args.segment_index,
        )
    except ActiveFragmentDeferred as exc:
        print(
            json.dumps(
                {"status": "deferred_active", "reason": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except NoRecoverableTail as exc:
        print(
            json.dumps(
                {"status": "no_recoverable_tail", "reason": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except TailRecoveryError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
