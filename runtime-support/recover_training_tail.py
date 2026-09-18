#!/usr/bin/env python3
"""Derive completed transition shards from a proven-inactive crash fragment.

The source fragment is never modified.  Recovery copies every complete,
hash-valid frame byte-for-byte into a separate derived root, reconstructs a
canonical index from the validated NPZ metadata, and commits the completion
marker last.  An incomplete final pack frame or final JSONL line is the only
source damage tolerated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np


_UNTIL_WIN = Path(__file__).resolve().parent.parent / "until-win"
if str(_UNTIL_WIN) not in sys.path:
    sys.path.insert(0, str(_UNTIL_WIN))

from transition_pack import (  # noqa: E402
    CONTENT_TYPE,
    FRAME_HEADER,
    INDEX_CONTENT_TYPE,
    MAGIC,
    MAX_FRAME_BYTES,
    SCHEMA,
    decode_transition,
)


SHARD_SCHEMA = "jev-nethack-transition-shard/v1"
RUNTIME_SCHEMA = "jev-nethack-until-win/v1"
PACK_RE = re.compile(r"^transitions-(\d{6})\.npzpack$")
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_SOURCE_PACK_BYTES = 64 * 1024 * 1024
MAX_SOURCE_INDEX_BYTES = 32 * 1024 * 1024
MAX_NPZ_MEMBERS = 256
MAX_NPZ_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 16 * 1024 * 1024
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class TrainingTailRecoveryError(RuntimeError):
    """The source cannot be proven safe and complete enough to recover."""


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class DirectoryChain:
    path: Path
    components: tuple[str, ...]
    fds: tuple[int, ...]

    @property
    def leaf_fd(self) -> int:
        return self.fds[-1]


@dataclass(frozen=True)
class Frame:
    offset: int
    payload_bytes: int
    sha256: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ShardPlan:
    number: int
    stem: str
    pack_path: Path
    index_path: Path
    marker_path: Path
    pack_snapshot: FileSnapshot
    index_snapshot: FileSnapshot
    pack_bytes: bytes
    index_bytes: bytes
    derived_pack: bytes
    derived_index: bytes
    marker_bytes: bytes
    records: int
    indexed_records: int
    pack_tail_bytes: int
    index_tail_bytes: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or Path(name).name != name or "/" in name:
        raise TrainingTailRecoveryError(f"unsafe path component: {name!r}")
    return name


def _open_child_directory(parent_fd: int, name: str, *, create: bool = False) -> int:
    name = _safe_name(name)
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise TrainingTailRecoveryError(f"cannot create output directory component: {name}") from exc
    try:
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise TrainingTailRecoveryError(f"directory component cannot be opened safely: {name}") from exc
    details = os.fstat(descriptor)
    if not stat.S_ISDIR(details.st_mode):
        os.close(descriptor)
        raise TrainingTailRecoveryError(f"opened component is not a directory: {name}")
    return descriptor


def _open_absolute_directory_chain(path: Path, *, create: bool = False) -> DirectoryChain:
    path = _lexical_absolute(path)
    if not path.is_absolute():
        raise TrainingTailRecoveryError("directory path must be absolute")
    components = tuple(part for part in path.parts[1:] if part)
    fds: list[int] = []
    try:
        root_fd = os.open("/", DIRECTORY_FLAGS)
        fds.append(root_fd)
        for component in components:
            fds.append(_open_child_directory(fds[-1], component, create=create))
        chain = DirectoryChain(path, components, tuple(fds))
        _require_directory_chain(chain)
        return chain
    except Exception:
        for descriptor in reversed(fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _require_child_identity(parent_fd: int, name: str, child_fd: int) -> None:
    name = _safe_name(name)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise TrainingTailRecoveryError(f"directory component changed during recovery: {name}") from exc
    opened = os.fstat(child_fd)
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise TrainingTailRecoveryError(f"directory component changed during recovery: {name}")


def _require_directory_chain(chain: DirectoryChain) -> None:
    for position, component in enumerate(chain.components):
        _require_child_identity(chain.fds[position], component, chain.fds[position + 1])


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(_safe_name(name), dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TrainingTailRecoveryError(f"cannot inspect directory entry: {name}") from exc
    return True


def _read_snapshot_at(
    directory_fd: int, name: str, *, limit: int, label: Path
) -> tuple[bytes, FileSnapshot]:
    name = _safe_name(name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise TrainingTailRecoveryError(f"required file cannot be opened safely: {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TrainingTailRecoveryError(f"required entry is not a regular file: {label}")
        if before.st_size < 0 or before.st_size > limit:
            raise TrainingTailRecoveryError(f"file exceeds recovery bound: {label}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise TrainingTailRecoveryError(f"file shortened while being read: {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TrainingTailRecoveryError(f"file grew while being read: {label}")
        body = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    first = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    second = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if first != second:
        raise TrainingTailRecoveryError(f"file changed while being read: {label}")
    return body, FileSnapshot(*first, _sha256(body))


def _require_snapshot_at(
    directory_fd: int, name: str, expected: FileSnapshot, *, limit: int, label: Path
) -> None:
    _, current = _read_snapshot_at(directory_fd, name, limit=limit, label=label)
    if current != expected:
        raise TrainingTailRecoveryError(f"source or evidence changed during recovery: {label}")


def _json_object(body: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrainingTailRecoveryError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TrainingTailRecoveryError(f"{label} is not a JSON object")
    return value


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise TrainingTailRecoveryError(f"cannot determine whether PID {pid} is active") from exc
    return True


def _fragment_reference_matches(reference: Any, *, state_path: Path, fragment: Path) -> bool:
    if reference is None:
        return False
    if not isinstance(reference, str) or not reference.strip():
        raise TrainingTailRecoveryError("state.activeFragment is invalid")
    text = reference.strip().rstrip("/")
    if text == fragment.name or text.endswith("/" + fragment.name):
        return True
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = state_path.parent / candidate
    return candidate.resolve(strict=False) == fragment


def _parse_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingTailRecoveryError(f"transition metadata {field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrainingTailRecoveryError(f"transition metadata {field} is invalid") from exc
    if parsed.tzinfo is None:
        raise TrainingTailRecoveryError(f"transition metadata {field} lacks a timezone")
    return value


def _preflight_npz(payload: bytes) -> None:
    """Bound ZIP expansion before NumPy allocates arrays from an NPZ payload."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TrainingTailRecoveryError("transition frame NPZ container is invalid") from exc
    try:
        members = archive.infolist()
        if not members or len(members) > MAX_NPZ_MEMBERS:
            raise TrainingTailRecoveryError("transition frame NPZ member count exceeds its bound")
        zip_total = 0
        declared_total = 0
        names: set[str] = set()
        for member in members:
            if (
                member.filename in names
                or member.is_dir()
                or not member.filename.endswith(".npy")
            ):
                raise TrainingTailRecoveryError(
                    "transition frame NPZ members are invalid or duplicated"
                )
            names.add(member.filename)
            if member.flag_bits & 0x1:
                raise TrainingTailRecoveryError("encrypted transition NPZ members are unsupported")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise TrainingTailRecoveryError("transition NPZ compression method is unsupported")
            if member.file_size < 0 or member.file_size > MAX_NPZ_MEMBER_BYTES:
                raise TrainingTailRecoveryError(
                    "transition NPZ member exceeds its decoded size bound"
                )
            zip_total += member.file_size
            if zip_total > MAX_NPZ_UNCOMPRESSED_BYTES:
                raise TrainingTailRecoveryError("transition NPZ decoded size exceeds its bound")
            try:
                with archive.open(member, "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise TrainingTailRecoveryError(
                            f"transition NPY format version is unsupported: {version}"
                        )
                    if dtype.hasobject:
                        raise TrainingTailRecoveryError("object arrays are forbidden in transitions")
                    elements = 1
                    for dimension in shape:
                        if not isinstance(dimension, int) or dimension < 0:
                            raise TrainingTailRecoveryError("transition NPY shape is invalid")
                        elements *= dimension
                        if elements * dtype.itemsize > MAX_NPZ_MEMBER_BYTES:
                            raise TrainingTailRecoveryError(
                                "transition NPY declared array exceeds its decoded size bound"
                            )
                    declared_bytes = elements * dtype.itemsize
                    if member.file_size - stream.tell() != declared_bytes:
                        raise TrainingTailRecoveryError(
                            "transition NPY declared shape disagrees with member size"
                        )
            except TrainingTailRecoveryError:
                raise
            except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
                raise TrainingTailRecoveryError("transition NPY header is invalid") from exc
            declared_total += declared_bytes
            if declared_total > MAX_NPZ_UNCOMPRESSED_BYTES:
                raise TrainingTailRecoveryError(
                    "transition NPZ declared arrays exceed their decoded size bound"
                )
    finally:
        archive.close()


def _scan_pack_bytes(body: bytes) -> tuple[list[Frame], int]:
    if not body.startswith(MAGIC):
        raise TrainingTailRecoveryError("transition pack magic is invalid")
    frames: list[Frame] = []
    event_ids: set[str] = set()
    offset = len(MAGIC)
    complete_end = offset
    while offset < len(body):
        remaining = len(body) - offset
        if remaining < FRAME_HEADER.size:
            break
        payload_bytes, expected = FRAME_HEADER.unpack_from(body, offset)
        if payload_bytes <= 0 or payload_bytes > MAX_FRAME_BYTES:
            raise TrainingTailRecoveryError("transition frame length is invalid")
        payload_start = offset + FRAME_HEADER.size
        payload_end = payload_start + payload_bytes
        if payload_end > len(body):
            break
        payload = body[payload_start:payload_end]
        actual = hashlib.sha256(payload).digest()
        if actual != expected:
            raise TrainingTailRecoveryError("transition frame digest mismatch")
        _preflight_npz(payload)
        try:
            _, metadata = decode_transition(payload)
        except Exception as exc:
            raise TrainingTailRecoveryError("transition frame is not a valid safe NPZ") from exc
        event_id = metadata.get("eventId") if isinstance(metadata, Mapping) else None
        if not isinstance(event_id, str) or not event_id:
            raise TrainingTailRecoveryError("transition frame eventId is invalid")
        if event_id in event_ids:
            raise TrainingTailRecoveryError(f"duplicate transition eventId: {event_id}")
        event_ids.add(event_id)
        frames.append(Frame(offset, payload_bytes, actual.hex(), dict(metadata)))
        offset = payload_end
        complete_end = offset
    if not frames:
        raise TrainingTailRecoveryError("unmarked shard contains no complete transition frames")
    return frames, complete_end


_INDEX_METADATA_FIELDS = (
    "episodeId",
    "seed",
    "step",
    "actionIndex",
    "keycode",
    "observationDigest",
    "nextObservationDigest",
    "reward",
    "terminated",
    "truncated",
    "isAscended",
    "verifiedAscension",
    "endStatus",
)


def _index_record(frame: Frame, *, shard: int, record: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "shard": shard,
        "record": record,
        "offset": frame.offset,
        "payloadBytes": frame.payload_bytes,
        "sha256": frame.sha256,
        "eventId": frame.metadata["eventId"],
    }
    for key in _INDEX_METADATA_FIELDS:
        result[key] = frame.metadata.get(key)
    return result


def _canonical_index(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def _validate_index_prefix(
    body: bytes, frames: Sequence[Frame], *, shard: int
) -> tuple[int, int]:
    complete_bytes = 0
    parsed: list[dict[str, Any]] = []
    cursor = 0
    for line in body.splitlines(keepends=True):
        cursor += len(line)
        if not line.endswith(b"\n"):
            break
        complete_bytes = cursor
        if not line.strip():
            raise TrainingTailRecoveryError("transition index contains a blank record")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise TrainingTailRecoveryError("transition index contains malformed complete JSON") from exc
        if not isinstance(value, dict):
            raise TrainingTailRecoveryError("transition index record is not an object")
        parsed.append(value)
    if len(parsed) > len(frames):
        raise TrainingTailRecoveryError("transition index is ahead of its pack")
    for number, (actual, frame) in enumerate(zip(parsed, frames), 1):
        expected = _index_record(frame, shard=shard, record=number)
        if any(actual.get(key) != value for key, value in expected.items()):
            raise TrainingTailRecoveryError(
                f"transition index disagrees with pack at record {number}"
            )
    return len(parsed), len(body) - complete_bytes


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _artifact(path: Path, body: bytes, content_type: str, records: int) -> dict[str, Any]:
    return {
        "filename": path.name,
        "contentType": content_type,
        "bytes": len(body),
        "sha256": _sha256(body),
        "records": records,
    }


def _build_plan(
    pack_name: str,
    *,
    training_fd: int,
    training_path: Path,
    config_path: Path,
    config_snapshot: FileSnapshot,
    fragment: Path,
) -> ShardPlan:
    match = PACK_RE.fullmatch(pack_name)
    if not match:
        raise TrainingTailRecoveryError(f"unexpected transition pack filename: {pack_name}")
    number = int(match.group(1))
    stem = pack_name.removesuffix(".npzpack")
    index_name = f"{stem}.index.jsonl"
    marker_name = f"{stem}.complete.json"
    pack_path = training_path / pack_name
    index_path = training_path / index_name
    marker_path = training_path / marker_name
    if _entry_exists(training_fd, marker_name):
        raise TrainingTailRecoveryError(f"source shard is already marked complete: {marker_path}")
    pack_body, pack_snapshot = _read_snapshot_at(
        training_fd, pack_name, limit=MAX_SOURCE_PACK_BYTES, label=pack_path
    )
    index_body, index_snapshot = _read_snapshot_at(
        training_fd, index_name, limit=MAX_SOURCE_INDEX_BYTES, label=index_path
    )
    frames, complete_end = _scan_pack_bytes(pack_body)
    indexed_records, index_tail_bytes = _validate_index_prefix(index_body, frames, shard=number)
    records = [_index_record(frame, shard=number, record=position) for position, frame in enumerate(frames, 1)]
    derived_pack = pack_body[:complete_end]
    derived_index = _canonical_index(records)
    started_at = _parse_timestamp(frames[0].metadata.get("recordedAt"), field="recordedAt")
    ended_at = _parse_timestamp(frames[-1].metadata.get("recordedAt"), field="recordedAt")
    if datetime.fromisoformat(ended_at.replace("Z", "+00:00")) < datetime.fromisoformat(
        started_at.replace("Z", "+00:00")
    ):
        raise TrainingTailRecoveryError("transition shard timestamps run backwards")

    output_pack = Path(pack_path.name)
    output_index = Path(index_path.name)
    artifacts = [
        _artifact(output_pack, derived_pack, CONTENT_TYPE, len(frames)),
        _artifact(output_index, derived_index, INDEX_CONTENT_TYPE, len(frames)),
    ]
    marker = {
        "schema": SHARD_SCHEMA,
        "completed": True,
        "records": len(frames),
        "transitionCount": len(frames),
        "startedAt": started_at,
        "endedAt": ended_at,
        "artifacts": artifacts,
        "provenance": {
            "recoveredAfterCrash": True,
            "sourcePaths": {
                "fragment": str(fragment.relative_to(fragment.parent.parent)),
                "config": str(config_path.relative_to(fragment.parent.parent)),
                "pack": str(pack_path.relative_to(fragment.parent.parent)),
                "index": str(index_path.relative_to(fragment.parent.parent)),
            },
            "sourceHashes": {
                "config": config_snapshot.sha256,
                "pack": pack_snapshot.sha256,
                "index": index_snapshot.sha256,
            },
            "sourceSizes": {
                "config": config_snapshot.size,
                "pack": pack_snapshot.size,
                "index": index_snapshot.size,
            },
            "indexedRecords": indexed_records,
            "partialTail": {
                "packBytesDiscarded": len(pack_body) - complete_end,
                "indexBytesDiscarded": index_tail_bytes,
            },
            "payloadBytesUnchanged": True,
        },
    }
    return ShardPlan(
        number=number,
        stem=stem,
        pack_path=pack_path,
        index_path=index_path,
        marker_path=marker_path,
        pack_snapshot=pack_snapshot,
        index_snapshot=index_snapshot,
        pack_bytes=pack_body,
        index_bytes=index_body,
        derived_pack=derived_pack,
        derived_index=derived_index,
        marker_bytes=_json_bytes(marker),
        records=len(frames),
        indexed_records=indexed_records,
        pack_tail_bytes=len(pack_body) - complete_end,
        index_tail_bytes=index_tail_bytes,
    )


def _install_exact_at(directory_fd: int, name: str, body: bytes, *, label: Path) -> bool:
    """Install bytes relative to a held directory without following symlinks."""
    name = _safe_name(name)
    if _entry_exists(directory_fd, name):
        existing, _ = _read_snapshot_at(
            directory_fd, name, limit=max(len(body), 1) + 1, label=label
        )
        if existing != body:
            raise TrainingTailRecoveryError(f"derived output conflicts with expected bytes: {label}")
        return False
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            created = True
        except FileExistsError:
            existing, _ = _read_snapshot_at(
                directory_fd, name, limit=max(len(body), 1) + 1, label=label
            )
            if existing != body:
                raise TrainingTailRecoveryError(
                    f"derived output appeared with conflicting bytes: {label}"
                )
            created = False
        os.fsync(directory_fd)
        return created
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass


def _acquire_lock_at(directory_fd: int, name: str):
    name = _safe_name(name)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise TrainingTailRecoveryError("recovery lock cannot be opened safely") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise TrainingTailRecoveryError("recovery lock is not a regular file")
    stream = os.fdopen(descriptor, "r+b", closefd=True)
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise TrainingTailRecoveryError("another training-tail recovery holds the lock") from exc
    return stream


def _require_lock_identity(directory_fd: int, name: str, stream) -> None:
    name = _safe_name(name)
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise TrainingTailRecoveryError("recovery lock changed during recovery") from exc
    opened = os.fstat(stream.fileno())
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise TrainingTailRecoveryError("recovery lock changed during recovery")


def recover_training_tail(
    fragment: Path,
    derived_root: Path,
    *,
    state_path: Path | None = None,
    lock_path: Path | None = None,
    shard_numbers: Sequence[int] | None = None,
    pid_is_alive: Callable[[int], bool] = _pid_is_alive,
    before_commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Recover all selected unmarked shards into ``derived_root``.

    ``derived_root`` is a data-root-shaped directory.  Results appear below
    ``fragments/<original-fragment-id>`` so publisher identities are identical
    to the identities a clean source close would have produced.  Source,
    destination, state, and custom-lock paths must use canonical absolute
    components; symlinked ancestors are rejected rather than followed.
    """
    fragment = _lexical_absolute(fragment)
    if fragment.parent.name != "fragments":
        raise TrainingTailRecoveryError("fragment must be directly below a fragments directory")
    data_root = fragment.parent.parent
    training = fragment / "training"
    config_path = fragment / "config.json"
    requested_state = _lexical_absolute(state_path or data_root / "state.json")
    if requested_state.parent != data_root:
        raise TrainingTailRecoveryError("state file must be a direct child of the source data root")
    state_path = requested_state

    derived_root = _lexical_absolute(derived_root)
    if (
        derived_root == data_root
        or derived_root == fragment
        or derived_root.is_relative_to(fragment)
        or fragment.is_relative_to(derived_root)
    ):
        raise TrainingTailRecoveryError("derived root must be separate from the source fragment")
    directory_fds: list[int] = []
    lock_stream = None
    lock_chain: DirectoryChain | None = None
    lock_parent_fd: int | None = None
    lock_name: str | None = None
    try:
        source_chain = _open_absolute_directory_chain(data_root)
        directory_fds.extend(source_chain.fds)
        data_root_fd = source_chain.leaf_fd
        source_fragments_fd = _open_child_directory(data_root_fd, "fragments")
        directory_fds.append(source_fragments_fd)
        source_fragment_fd = _open_child_directory(source_fragments_fd, fragment.name)
        directory_fds.append(source_fragment_fd)
        source_training_fd = _open_child_directory(source_fragment_fd, "training")
        directory_fds.append(source_training_fd)
        derived_chain = _open_absolute_directory_chain(derived_root, create=True)
        directory_fds.extend(derived_chain.fds)
        derived_root_fd = derived_chain.leaf_fd

        _require_directory_chain(source_chain)
        _require_child_identity(data_root_fd, "fragments", source_fragments_fd)
        _require_child_identity(source_fragments_fd, fragment.name, source_fragment_fd)
        _require_child_identity(source_fragment_fd, "training", source_training_fd)
        _require_directory_chain(derived_chain)

        if lock_path is None:
            lock_chain = derived_chain
            lock_parent_fd = derived_root_fd
            lock_name = ".recover-training-tail.lock"
            lock_stream = _acquire_lock_at(lock_parent_fd, lock_name)
        else:
            normalized_lock = _lexical_absolute(lock_path)
            custom_lock_chain = _open_absolute_directory_chain(normalized_lock.parent, create=True)
            directory_fds.extend(custom_lock_chain.fds)
            lock_chain = custom_lock_chain
            lock_parent_fd = custom_lock_chain.leaf_fd
            lock_name = normalized_lock.name
            lock_stream = _acquire_lock_at(lock_parent_fd, lock_name)

        def require_lock_held() -> None:
            assert lock_chain is not None and lock_parent_fd is not None and lock_name is not None
            assert lock_stream is not None
            _require_directory_chain(lock_chain)
            _require_lock_identity(lock_parent_fd, lock_name, lock_stream)

        require_lock_held()

        config_body, config_snapshot = _read_snapshot_at(
            source_fragment_fd, "config.json", limit=MAX_EVIDENCE_BYTES, label=config_path
        )
        config = _json_object(config_body, label="fragment config")
        if config.get("schema") != RUNTIME_SCHEMA or config.get("fragmentId") != fragment.name:
            raise TrainingTailRecoveryError("fragment config identity or schema is invalid")
        if config.get("transitionSchema") != SCHEMA:
            raise TrainingTailRecoveryError("fragment transition schema is invalid")
        pid = config.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise TrainingTailRecoveryError("fragment config PID is invalid")
        if pid_is_alive(pid):
            raise TrainingTailRecoveryError(f"source fragment PID is still active: {pid}")

        def require_state_excludes_source() -> dict[str, Any]:
            state_body, _ = _read_snapshot_at(
                data_root_fd, state_path.name, limit=MAX_EVIDENCE_BYTES, label=state_path
            )
            state_value = _json_object(state_body, label="runtime state")
            if state_value.get("schema") != RUNTIME_SCHEMA or "activeFragment" not in state_value:
                raise TrainingTailRecoveryError("runtime state schema or activeFragment is invalid")
            if _fragment_reference_matches(
                state_value["activeFragment"], state_path=state_path, fragment=fragment
            ):
                raise TrainingTailRecoveryError("source fragment is still the active fragment")
            return state_value

        require_state_excludes_source()

        selected = set(shard_numbers or ())
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in selected):
            raise TrainingTailRecoveryError("shard numbers must be positive integers")
        packs = sorted(name for name in os.listdir(source_training_fd) if PACK_RE.fullmatch(name))
        candidates: list[str] = []
        seen_numbers: set[int] = set()
        for name in packs:
            match = PACK_RE.fullmatch(name)
            if not match:
                continue
            number = int(match.group(1))
            marker_name = f"transitions-{number:06d}.complete.json"
            if _entry_exists(source_training_fd, marker_name):
                continue
            if selected and number not in selected:
                continue
            candidates.append(name)
            seen_numbers.add(number)
        missing = selected - seen_numbers
        if missing:
            raise TrainingTailRecoveryError(f"selected unmarked shards were not found: {sorted(missing)}")
        if not candidates:
            raise TrainingTailRecoveryError("fragment has no selected unmarked transition shards")

        plans = [
            _build_plan(
                name,
                training_fd=source_training_fd,
                training_path=training,
                config_path=config_path,
                config_snapshot=config_snapshot,
                fragment=fragment,
            )
            for name in candidates
        ]
        event_ids: set[str] = set()
        for plan in plans:
            frames, _ = _scan_pack_bytes(plan.derived_pack)
            for frame in frames:
                event_id = str(frame.metadata["eventId"])
                if event_id in event_ids:
                    raise TrainingTailRecoveryError(f"duplicate transition eventId across shards: {event_id}")
                event_ids.add(event_id)

        if before_commit is not None:
            before_commit()

        require_lock_held()
        _require_directory_chain(source_chain)
        _require_child_identity(data_root_fd, "fragments", source_fragments_fd)
        _require_child_identity(source_fragments_fd, fragment.name, source_fragment_fd)
        _require_child_identity(source_fragment_fd, "training", source_training_fd)
        _require_snapshot_at(
            source_fragment_fd,
            "config.json",
            config_snapshot,
            limit=MAX_EVIDENCE_BYTES,
            label=config_path,
        )
        require_state_excludes_source()
        if pid_is_alive(pid):
            raise TrainingTailRecoveryError(f"source fragment PID became active: {pid}")
        for plan in plans:
            _require_snapshot_at(
                source_training_fd,
                plan.pack_path.name,
                plan.pack_snapshot,
                limit=MAX_SOURCE_PACK_BYTES,
                label=plan.pack_path,
            )
            _require_snapshot_at(
                source_training_fd,
                plan.index_path.name,
                plan.index_snapshot,
                limit=MAX_SOURCE_INDEX_BYTES,
                label=plan.index_path,
            )
            if _entry_exists(source_training_fd, plan.marker_path.name):
                raise TrainingTailRecoveryError("source shard became complete during recovery")

        fragments_root = derived_root / "fragments"
        fragments_root_fd = _open_child_directory(derived_root_fd, "fragments", create=True)
        directory_fds.append(fragments_root_fd)
        output_fragment = fragments_root / fragment.name
        output_fragment_fd = _open_child_directory(
            fragments_root_fd, fragment.name, create=True
        )
        directory_fds.append(output_fragment_fd)
        output_training = output_fragment / "training"
        output_training_fd = _open_child_directory(output_fragment_fd, "training", create=True)
        directory_fds.append(output_training_fd)

        def require_output_tree() -> None:
            require_lock_held()
            _require_directory_chain(derived_chain)
            _require_child_identity(derived_root_fd, "fragments", fragments_root_fd)
            _require_child_identity(fragments_root_fd, fragment.name, output_fragment_fd)
            _require_child_identity(output_fragment_fd, "training", output_training_fd)

        require_output_tree()
        created_any = _install_exact_at(
            output_fragment_fd,
            "config.json",
            config_body,
            label=output_fragment / "config.json",
        )
        results: list[dict[str, Any]] = []
        for plan in plans:
            pack_output = output_training / f"{plan.stem}.npzpack"
            index_output = output_training / f"{plan.stem}.index.jsonl"
            marker_output = output_training / f"{plan.stem}.complete.json"
            require_output_tree()
            pack_created = _install_exact_at(
                output_training_fd, pack_output.name, plan.derived_pack, label=pack_output
            )
            require_output_tree()
            index_created = _install_exact_at(
                output_training_fd, index_output.name, plan.derived_index, label=index_output
            )
            _require_directory_chain(source_chain)
            _require_child_identity(data_root_fd, "fragments", source_fragments_fd)
            _require_child_identity(source_fragments_fd, fragment.name, source_fragment_fd)
            _require_child_identity(source_fragment_fd, "training", source_training_fd)
            _require_snapshot_at(
                source_fragment_fd,
                "config.json",
                config_snapshot,
                limit=MAX_EVIDENCE_BYTES,
                label=config_path,
            )
            require_state_excludes_source()
            _require_snapshot_at(
                source_training_fd,
                plan.pack_path.name,
                plan.pack_snapshot,
                limit=MAX_SOURCE_PACK_BYTES,
                label=plan.pack_path,
            )
            _require_snapshot_at(
                source_training_fd,
                plan.index_path.name,
                plan.index_snapshot,
                limit=MAX_SOURCE_INDEX_BYTES,
                label=plan.index_path,
            )
            require_output_tree()
            if pid_is_alive(pid) or _entry_exists(source_training_fd, plan.marker_path.name):
                raise TrainingTailRecoveryError("source activity changed before marker commit")
            marker_created = _install_exact_at(
                output_training_fd, marker_output.name, plan.marker_bytes, label=marker_output
            )
            created_any = created_any or pack_created or index_created or marker_created
            results.append(
                {
                    "shard": plan.number,
                    "records": plan.records,
                    "indexedRecords": plan.indexed_records,
                    "packTailBytesDiscarded": plan.pack_tail_bytes,
                    "indexTailBytesDiscarded": plan.index_tail_bytes,
                    "pack": str(pack_output),
                    "index": str(index_output),
                    "marker": str(marker_output),
                    "markerSha256": _sha256(plan.marker_bytes),
                }
            )
        os.fsync(output_training_fd)
        os.fsync(output_fragment_fd)
        os.fsync(fragments_root_fd)
        os.fsync(derived_root_fd)
        require_output_tree()
        return {
            "status": "recovered" if created_any else "already_recovered",
            "fragmentId": fragment.name,
            "sourceFragment": str(fragment),
            "outputFragment": str(output_fragment),
            "sourceUnchanged": True,
            "shards": results,
        }
    finally:
        if lock_stream is not None:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                lock_stream.close()
        for descriptor in reversed(directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--shard", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    try:
        result = recover_training_tail(
            args.fragment,
            args.derived_root,
            state_path=args.state,
            lock_path=args.lock,
            shard_numbers=args.shard,
        )
    except TrainingTailRecoveryError as exc:
        print(f"training tail recovery refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
