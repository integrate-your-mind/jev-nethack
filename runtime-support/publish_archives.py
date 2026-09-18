#!/usr/bin/env python3
"""Publish closed, checksummed Jev NetHack artifacts and nothing else."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tarfile
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit
import urllib.error
import urllib.request

from video_retention import (
    DEFAULT_RECORDING_CACHE_BYTES,
    DEFAULT_RELEASE_CACHE_BYTES,
    RetentionError,
    prune_recording_cache,
    prune_release_cache,
    recover_tombstones,
    tombstone_covers,
)


PACK_SCHEMA = "jev-nethack-transition-pack/v1"
SHARD_SCHEMA = "jev-nethack-transition-shard/v1"
RELEASE_SCHEMA = "jev-nethack-release-batch/v1"
MAGIC = b"JEVNHNPZ1\n"
FRAME_HEADER = struct.Struct(">Q32s")
PACK_TYPE = "application/vnd.jev-nethack.transitions+npz"
INDEX_TYPE = "application/x-ndjson"
JSON_TYPE = "application/json"
MAX_UPLOAD = 32 * 1024 * 1024
MAX_LOCAL_ARTIFACT = 1024 * 1024 * 1024
MAX_MANIFEST = 256 * 1024
RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
USER_AGENT = "Jev-NetHack-Archive-Publisher/1.0"
FULL_REMOTE_RECHECK_SECONDS = 24 * 60 * 60
REMOTE_EXISTENCE_RECHECK_SECONDS = 15 * 60
REMOTE_RECHECK_BUDGET = 8
FULL_LOCAL_SCRUB_SECONDS = 24 * 60 * 60
SITE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
RECORDING_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
MARKER_RE = re.compile(r"^transitions-(\d+)\.complete\.json$")
VIDEO_SEGMENT_RE = re.compile(r"^segment-(\d{4})\.jsonl$")


class PublishError(RuntimeError):
    pass


class DeferredPublish(PublishError):
    pass


class LockBusy(PublishError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_json(value))


def read_json_document(path: Path, *, limit: int = MAX_MANIFEST) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise PublishError(f"invalid JSON source: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublishError(f"invalid JSON source: {path}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or not 0 < details.st_size <= limit:
            raise PublishError(f"invalid JSON source: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != details.st_size:
            raise PublishError(f"JSON source changed while reading: {path}")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PublishError(f"invalid JSON source: {path}") from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise PublishError(f"JSON source is not an object: {path}")
    return value, raw


def read_json(path: Path, *, limit: int = MAX_MANIFEST) -> dict[str, Any]:
    return read_json_document(path, limit=limit)[0]


def read_token(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise PublishError("token file must be a private regular file")
    token = path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise PublishError("token file is empty or malformed")
    return token


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise PublishError("archive timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublishError("archive timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise PublishError("archive timestamp lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _inside(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise PublishError(f"source escapes configured root: {path}")
    return resolved


def _artifact_path(parent: Path, filename: str, root: Path) -> Path:
    if not FILE_RE.fullmatch(filename) or filename != Path(filename).name:
        raise PublishError("artifact filename is unsafe")
    candidate = parent / filename
    if candidate.is_symlink():
        raise PublishError("artifact may not be a symlink")
    resolved = _inside(candidate, root)
    if not resolved.is_file() or resolved.parent != parent.resolve():
        raise PublishError("artifact is missing or outside its marker directory")
    return resolved


def read_exact(path: Path, size: int, digest: str, *, limit: int = MAX_LOCAL_ARTIFACT) -> bytes:
    """Read the validated inode without following a replacement symlink."""
    if size <= 0 or size > limit:
        raise PublishError("artifact size is outside the supported bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublishError("artifact could not be opened safely") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != size:
            raise PublishError("artifact inode changed after validation")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) != size or sha256_bytes(data) != digest:
            raise PublishError("artifact bytes changed after validation")
        return data
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Artifact:
    filename: str
    path: Path
    content_type: str
    size: int
    sha256: str

    def public_spec(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "contentType": self.content_type,
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArchiveFile:
    name: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ClosedItem:
    kind: str
    item_id: str
    source_id: str
    source_path: Path
    source_sha256: str
    started_at: str
    ended_at: str
    artifacts: tuple[Artifact, ...]
    site_manifest: bytes
    release_files: tuple[ArchiveFile, ...]


@dataclass(frozen=True)
class Frame:
    source_offset: int
    payload_bytes: int
    sha256: str
    event_id: str
    index: Mapping[str, Any]


def _declared_artifact(
    raw: Any, *, parent: Path, root: Path, expected_type: str, suffix: str
) -> Artifact:
    if not isinstance(raw, dict):
        raise PublishError("artifact entry is not an object")
    filename, digest, size = raw.get("filename"), raw.get("sha256"), raw.get("bytes")
    if not isinstance(filename, str) or not filename.endswith(suffix):
        raise PublishError("artifact has the wrong filename")
    if raw.get("contentType") != expected_type:
        raise PublishError("artifact has the wrong content type")
    if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
        raise PublishError("artifact has an invalid SHA-256")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_LOCAL_ARTIFACT:
        raise PublishError("artifact has an invalid size")
    path = _artifact_path(parent, filename, root)
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise PublishError("artifact bytes do not match their completion marker")
    return Artifact(filename, path, expected_type, size, digest)


def validate_pack_index(pack: Artifact, index: Artifact, expected_count: int) -> tuple[Frame, ...]:
    """Validate every frame/NPZ and its exact JSONL counterpart."""
    try:
        import numpy as np
    except ImportError as exc:
        raise PublishError("numpy is required for transition verification") from exc
    frames: list[tuple[int, int, str, str]] = []
    event_ids: set[str] = set()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(pack.path, flags)
    except OSError as exc:
        raise PublishError("pack could not be opened safely") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != pack.size:
            raise PublishError("pack inode changed during validation")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            magic = stream.read(len(MAGIC))
            if magic != MAGIC:
                raise PublishError("transition pack magic is invalid")
            whole_digest = hashlib.sha256(magic)
            while True:
                offset = stream.tell()
                header = stream.read(FRAME_HEADER.size)
                if not header:
                    break
                if len(header) != FRAME_HEADER.size:
                    raise PublishError("transition pack has a truncated frame header")
                payload_size, expected_digest = FRAME_HEADER.unpack(header)
                if not 0 < payload_size <= MAX_LOCAL_ARTIFACT:
                    raise PublishError("transition frame size is invalid")
                payload = stream.read(payload_size)
                if len(payload) != payload_size or hashlib.sha256(payload).digest() != expected_digest:
                    raise PublishError("transition frame is truncated or corrupt")
                whole_digest.update(header)
                whole_digest.update(payload)
                try:
                    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                        arrays = {name: archive[name] for name in archive.files}
                    encoded = arrays.get("metadata_json")
                    if encoded is None or encoded.dtype != np.uint8:
                        raise ValueError("metadata_json missing")
                    metadata = json.loads(encoded.tobytes().decode("utf-8"))
                except (OSError, UnicodeError, ValueError, KeyError) as exc:
                    raise PublishError("transition payload is not a safe NPZ") from exc
                event_id = metadata.get("eventId") if isinstance(metadata, dict) else None
                if metadata.get("schema") != PACK_SCHEMA or not isinstance(event_id, str) or not event_id:
                    raise PublishError("transition payload metadata is invalid")
                if event_id in event_ids:
                    raise PublishError("transition event IDs are duplicated")
                event_ids.add(event_id)
                frames.append((offset, payload_size, expected_digest.hex(), event_id))
            final_position = stream.tell()
            if final_position != details.st_size or whole_digest.hexdigest() != pack.sha256:
                raise PublishError("transition pack changed during validation")
    finally:
        os.close(descriptor)
    raw_index = read_exact(index.path, index.size, index.sha256)
    if not raw_index.endswith(b"\n"):
        raise PublishError("completed transition index lacks its final newline")
    try:
        records = [json.loads(line) for line in raw_index.decode("utf-8").splitlines()]
    except (UnicodeError, ValueError) as exc:
        raise PublishError("transition index is not complete JSONL") from exc
    if len(frames) != expected_count or len(records) != expected_count:
        raise PublishError("transition pack/index count differs from marker")
    result: list[Frame] = []
    for number, (raw_frame, record) in enumerate(zip(frames, records), 1):
        if not isinstance(record, dict):
            raise PublishError("transition index entry is not an object")
        offset, payload_size, digest, event_id = raw_frame
        expected = {
            "schema": PACK_SCHEMA, "record": number, "offset": offset,
            "payloadBytes": payload_size, "sha256": digest, "eventId": event_id,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise PublishError(f"transition index disagrees with frame {number}")
        result.append(Frame(offset, payload_size, digest, event_id, record))
    return tuple(result)


def _source_shard_id(marker_path: Path, data_root: Path) -> str:
    fragment_dir = marker_path.parent.parent
    config_path = fragment_dir / "config.json"
    if config_path.is_symlink():
        raise PublishError("fragment config may not be a symlink")
    config = read_json(_inside(config_path, data_root))
    fragment_id = config.get("fragmentId")
    match = MARKER_RE.fullmatch(marker_path.name)
    if not isinstance(fragment_id, str) or fragment_id != fragment_dir.name or not match:
        raise PublishError("fragment/shard identity is invalid")
    identity = f"{fragment_id}:{int(match.group(1)):06d}".encode()
    shard_id = "shard-" + hashlib.sha256(identity).hexdigest()
    assert SITE_ID_RE.fullmatch(shard_id)
    return shard_id


def _public_manifest(
    *, item_id: str, started_at: str, ended_at: str, artifacts: Sequence[Artifact],
    count: int, provenance: Mapping[str, Any]
) -> bytes:
    return canonical_json(
        {
            "schema": PACK_SCHEMA, "schemaVersion": 1, "sessionId": item_id,
            "completed": True, "startedAt": started_at, "endedAt": ended_at,
            "transitionCount": count, "artifacts": [item.public_spec() for item in artifacts],
            "provenance": dict(provenance),
        }
    )


def _copy_frame_payload(source: Artifact, frame: Frame) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source.path, flags)
    except OSError as exc:
        raise PublishError("source pack could not be opened safely") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != source.size:
            raise PublishError("source pack changed during repackaging")
        payload = os.pread(descriptor, frame.payload_bytes, frame.source_offset + FRAME_HEADER.size)
    finally:
        os.close(descriptor)
    if len(payload) != frame.payload_bytes or sha256_bytes(payload) != frame.sha256:
        raise PublishError("source frame changed during repackaging")
    return payload


def _repackage_groups(source_id: str, frames: Sequence[Frame]) -> list[list[tuple[Frame, dict[str, Any], bytes]]]:
    groups: list[list[tuple[Frame, dict[str, Any], bytes]]] = []
    current: list[tuple[Frame, dict[str, Any], bytes]] = []
    pack_size, index_size = len(MAGIC), 0
    for frame in frames:
        def entry_for(local_number: int, local_offset: int) -> tuple[dict[str, Any], bytes]:
            entry = dict(frame.index)
            entry.update(
                {"record": local_number, "offset": local_offset,
                 "sourceShardId": source_id, "sourceRecord": frame.index["record"],
                 "sourceOffset": frame.source_offset}
            )
            line = canonical_json(entry)
            return entry, line
        entry, line = entry_for(len(current) + 1, pack_size)
        projected_pack = pack_size + FRAME_HEADER.size + frame.payload_bytes
        projected_index = index_size + len(line)
        if current and (projected_pack > MAX_UPLOAD or projected_index > MAX_UPLOAD):
            groups.append(current)
            current, pack_size, index_size = [], len(MAGIC), 0
            entry, line = entry_for(1, pack_size)
            projected_pack = pack_size + FRAME_HEADER.size + frame.payload_bytes
            projected_index = len(line)
        if projected_pack > MAX_UPLOAD or projected_index > MAX_UPLOAD:
            raise PublishError("one transition cannot fit the Site upload bound")
        current.append((frame, entry, line))
        pack_size, index_size = projected_pack, projected_index
    if current:
        groups.append(current)
    return groups


def parse_training_marker(path: Path, data_root: Path, derived_root: Path) -> list[ClosedItem]:
    data_root = data_root.resolve()
    path = _inside(path, data_root)
    marker, marker_bytes = read_json_document(path)
    if marker.get("schema") != SHARD_SCHEMA or marker.get("completed") is not True:
        raise PublishError("marker is not a completed transition shard")
    count = marker.get("transitionCount")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or marker.get("records") != count:
        raise PublishError("transition marker count is invalid")
    started, ended = _timestamp(marker.get("startedAt")), _timestamp(marker.get("endedAt"))
    if ended < started:
        raise PublishError("transition marker ends before it starts")
    raw_artifacts = marker.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 2:
        raise PublishError("transition marker requires exactly two artifacts")
    by_kind: dict[str, Mapping[str, Any]] = {}
    for raw in raw_artifacts:
        filename = raw.get("filename") if isinstance(raw, dict) else None
        if isinstance(filename, str) and filename.endswith(".npzpack"):
            by_kind["pack"] = raw
        elif isinstance(filename, str) and filename.endswith(".index.jsonl"):
            by_kind["index"] = raw
    if set(by_kind) != {"pack", "index"}:
        raise PublishError("transition marker must contain pack and index")
    pack = _declared_artifact(
        by_kind["pack"], parent=path.parent, root=data_root, expected_type=PACK_TYPE, suffix=".npzpack"
    )
    index = _declared_artifact(
        by_kind["index"], parent=path.parent, root=data_root, expected_type=INDEX_TYPE, suffix=".index.jsonl"
    )
    if any(raw.get("records") != count for raw in raw_artifacts):
        raise PublishError("artifact record count differs from marker")
    frames = validate_pack_index(pack, index, count)
    source_id = _source_shard_id(path, data_root)
    source_sha = sha256_bytes(marker_bytes)
    release_base = f"training/{source_id}"
    release_files = (
        ArchiveFile(f"{release_base}/{pack.filename}", pack.path, pack.size, pack.sha256),
        ArchiveFile(f"{release_base}/{index.filename}", index.path, index.size, index.sha256),
        ArchiveFile(f"{release_base}/{path.name}", path, len(marker_bytes), source_sha),
    )
    provenance = {
        "sourceShardId": source_id, "sourceMarkerSha256": source_sha,
        "sourceRecordStart": 1, "sourceRecordEnd": count, "payloadBytesUnchanged": True,
    }
    if pack.size <= MAX_UPLOAD and index.size <= MAX_UPLOAD:
        manifest = _public_manifest(
            item_id=source_id, started_at=marker["startedAt"], ended_at=marker["endedAt"],
            artifacts=(pack, index), count=count, provenance=provenance,
        )
        return [
            ClosedItem("training", source_id, source_id, path, source_sha, marker["startedAt"],
                       marker["endedAt"], (pack, index), manifest, release_files)
        ]

    groups = _repackage_groups(source_id, frames)
    items: list[ClosedItem] = []
    for part, group in enumerate(groups, 1):
        item_id = f"{source_id}-p{part:06d}"
        if not SITE_ID_RE.fullmatch(item_id):
            raise PublishError("derived Site shard ID is invalid")
        directory = derived_root / source_id / f"part-{part:06d}"
        pack_path = directory / f"transitions-{part:06d}.npzpack"
        index_path = directory / f"transitions-{part:06d}.index.jsonl"
        pack_buffer = io.BytesIO()
        pack_buffer.write(MAGIC)
        index_buffer = io.BytesIO()
        for frame, _, line in group:
            payload = _copy_frame_payload(pack, frame)
            pack_buffer.write(FRAME_HEADER.pack(frame.payload_bytes, bytes.fromhex(frame.sha256)))
            pack_buffer.write(payload)
            index_buffer.write(line)
        pack_bytes, index_bytes = pack_buffer.getvalue(), index_buffer.getvalue()
        if len(pack_bytes) > MAX_UPLOAD or len(index_bytes) > MAX_UPLOAD:
            raise PublishError("derived public shard exceeds Site limit")
        atomic_bytes(pack_path, pack_bytes)
        atomic_bytes(index_path, index_bytes)
        public_pack = Artifact(pack_path.name, pack_path, PACK_TYPE, len(pack_bytes), sha256_bytes(pack_bytes))
        public_index = Artifact(index_path.name, index_path, INDEX_TYPE, len(index_bytes), sha256_bytes(index_bytes))
        start_record = int(group[0][0].index["record"])
        end_record = int(group[-1][0].index["record"])
        part_provenance = {
            **provenance, "sourceRecordStart": start_record, "sourceRecordEnd": end_record,
            "part": part, "parts": len(groups),
        }
        manifest = _public_manifest(
            item_id=item_id, started_at=marker["startedAt"], ended_at=marker["endedAt"],
            artifacts=(public_pack, public_index), count=len(group), provenance=part_provenance,
        )
        items.append(
            ClosedItem(
                "training", item_id, source_id, path, source_sha, marker["startedAt"], marker["endedAt"],
                (public_pack, public_index), manifest, release_files if part == 1 else (),
            )
        )
    return items


def verify_mp4(path: Path, size: int, digest: str, ffprobe: str = "ffprobe") -> None:
    body = read_exact(path, size, digest, limit=MAX_UPLOAD)
    descriptor, temporary_name = tempfile.mkstemp(prefix="jev-nethack-mp4-", suffix=".mp4")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        probe_path = temporary
        command = [
            ffprobe, "-v", "error", "-show_entries", "stream=codec_type:format=duration",
            "-of", "json", str(probe_path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            value = json.loads(result.stdout)
            duration = float(value.get("format", {}).get("duration", 0))
        except (subprocess.SubprocessError, OSError, ValueError, TypeError) as exc:
            raise PublishError("recording MP4 failed ffprobe validation") from exc
        if duration <= 0 or not any(item.get("codec_type") == "video" for item in value.get("streams", [])):
            raise PublishError("recording MP4 has no nonempty video stream")
    finally:
        temporary.unlink(missing_ok=True)


def parse_recording_manifest(path: Path, recordings_root: Path) -> ClosedItem:
    root = recordings_root.resolve()
    path = _inside(path, root)
    manifest, manifest_bytes = read_json_document(path)
    session_id = manifest.get("sessionId")
    if manifest.get("schemaVersion") != 1 or manifest.get("completed") is not True:
        raise PublishError("recording manifest is not completed")
    if not isinstance(session_id, str) or not RECORDING_ID_RE.fullmatch(session_id):
        raise PublishError("recording session ID is invalid")
    for field in ("frameCount", "actionCount"):
        value = manifest.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PublishError(f"recording {field} is invalid")
    started, ended = _timestamp(manifest.get("startedAt")), _timestamp(manifest.get("endedAt"))
    if ended < started:
        raise PublishError("recording ends before it starts")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not 2 <= len(raw_artifacts) <= 16:
        raise PublishError("recording artifact list is invalid")
    artifacts: list[Artifact] = []
    names: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or not isinstance(raw.get("filename"), str):
            raise PublishError("recording artifact is invalid")
        filename = raw["filename"]
        if filename in names:
            raise PublishError("recording artifact filename is duplicated")
        names.add(filename)
        if filename.endswith(".mp4"):
            expected_type, suffix = "video/mp4", ".mp4"
        elif filename.endswith(".jsonl"):
            expected_type, suffix = INDEX_TYPE, ".jsonl"
        else:
            raise PublishError("recording artifact type is unsupported")
        artifact = _declared_artifact(
            raw, parent=path.parent, root=root, expected_type=expected_type, suffix=suffix
        )
        if suffix == ".mp4":
            verify_mp4(artifact.path, artifact.size, artifact.sha256)
        artifacts.append(artifact)
    if not any(item.filename.endswith(".mp4") for item in artifacts) or not any(
        item.filename.endswith(".jsonl") for item in artifacts
    ):
        raise PublishError("recording requires MP4 and JSONL")
    base = f"recordings/{session_id}"
    release_files = tuple(
        [ArchiveFile(f"{base}/{item.filename}", item.path, item.size, item.sha256) for item in artifacts]
        + [ArchiveFile(f"{base}/manifest.json", path, len(manifest_bytes), sha256_bytes(manifest_bytes))]
    )
    return ClosedItem(
        "recording", session_id, session_id, path, sha256_bytes(manifest_bytes),
        manifest["startedAt"], manifest["endedAt"], tuple(artifacts), manifest_bytes, release_files,
    )


def discover_training(root: Path) -> list[Path]:
    resolved = root.resolve()
    return sorted(
        path for path in resolved.rglob("transitions-*.complete.json")
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(resolved)
    )


def discover_recordings(root: Path) -> list[Path]:
    resolved = root.resolve()
    return sorted(
        path for path in resolved.rglob("segment-*.manifest.json")
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(resolved)
    )


class SiteClient:
    def __init__(self, origin: str, token: str, timeout: float = 30.0) -> None:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")
        ):
            raise PublishError("site URL must be a bare HTTPS origin")
        self.origin, self.token, self.timeout = origin.rstrip("/"), token, timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(
        self, method: str, path: str, body: bytes | None = None, *, content_type: str | None = None
    ) -> tuple[int, bytes]:
        if not path.startswith("/"):
            raise PublishError("site request path must be absolute")
        if body is not None and len(body) > MAX_UPLOAD:
            raise PublishError("upload exceeds Site limit")
        headers = {"Accept": "application/octet-stream", "User-Agent": USER_AGENT}
        if method == "PUT":
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers.update(
                {"Content-Length": str(len(body)), "X-Content-SHA256": sha256_bytes(body),
                 "Content-Type": content_type or "application/octet-stream"}
            )
        request = urllib.request.Request(self.origin + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = response.read(MAX_UPLOAD + 1)
                if len(data) > MAX_UPLOAD:
                    raise PublishError("site response exceeds bound")
                return response.status, data
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                return 409, b""
            raise PublishError(f"site HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PublishError("site transport failure") from exc

    def verify(self, path: str, body: bytes) -> bool:
        status, readback = self._request("GET", path)
        return status == 200 and len(readback) == len(body) and sha256_bytes(readback) == sha256_bytes(body)

    def verify_digest(self, path: str, size: int, digest: str) -> bool:
        status, readback = self._request("GET", path)
        return status == 200 and len(readback) == size and sha256_bytes(readback) == digest

    def head_size_matches(self, path: str, size: int) -> bool:
        request = urllib.request.Request(
            self.origin + path,
            headers={"Accept": "application/octet-stream", "User-Agent": USER_AGENT},
            method="HEAD",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.status == 200 and response.headers.get("Content-Length") == str(size)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise PublishError(f"site HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PublishError("site transport failure") from exc

    def put_verified(self, path: str, body: bytes, *, content_type: str) -> None:
        put_error: PublishError | None = None
        try:
            status, _ = self._request("PUT", path, body, content_type=content_type)
            if status not in (200, 201, 409):
                raise PublishError(f"unexpected Site PUT status {status}")
        except PublishError as exc:
            put_error = exc
        try:
            if self.verify(path, body):
                return
        except PublishError:
            if put_error is not None:
                raise put_error
            raise
        if put_error is not None:
            raise put_error
        raise PublishError("site readback checksum mismatch")


def _site_path(item: ClosedItem, filename: str) -> str:
    collection = "training" if item.kind == "training" else "recordings"
    return f"/api/{collection}/{quote(item.item_id, safe='')}/{quote(filename, safe='')}"


def _receipt(path: Path, item: ClosedItem, *, site: bool) -> bool:
    if not path.exists():
        return False
    value = read_json(path)
    expected_manifest = sha256_bytes(item.site_manifest) if site else None
    if (
        value.get("verified") is not True or value.get("sourceSha256") != item.source_sha256
        or (site and value.get("manifestSha256") != expected_manifest)
    ):
        raise PublishError(f"receipt conflicts with immutable item {item.source_id}")
    return True


def _staggered_next_epoch(identity: str, checked_epoch: float, interval: float) -> float:
    if interval < 0 or not math.isfinite(interval):
        raise PublishError("remote recheck interval must be finite and non-negative")
    if interval == 0:
        return checked_epoch
    encoded = identity.encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") / (2**64 - 1)
    return checked_epoch + interval + fraction * interval


def _next_remote_check_epoch(item: ClosedItem, checked_epoch: float, interval: float) -> float:
    return _staggered_next_epoch(f"{item.kind}:{item.item_id}", checked_epoch, interval)


def _remote_receipt_due(
    path: Path, item: ClosedItem, *, site: bool, now_epoch: float,
) -> tuple[bool, bool]:
    """Return ``(receipt_exists, remote_check_due)`` after identity validation."""
    if not _receipt(path, item, site=site):
        return False, True
    value = read_json(path)
    raw_next = value.get("nextRemoteCheckEpoch")
    if not isinstance(raw_next, (int, float)) or isinstance(raw_next, bool):
        return True, True
    next_epoch = float(raw_next)
    if not math.isfinite(next_epoch) or next_epoch < 0:
        raise PublishError("remote verification receipt has an invalid next check time")
    return True, next_epoch <= now_epoch


def publish_site_item(
    item: ClosedItem, client: SiteClient, receipts: Path, *,
    full_recheck_seconds: float = FULL_REMOTE_RECHECK_SECONDS,
    remote_recheck_seconds: float = REMOTE_EXISTENCE_RECHECK_SECONDS,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if full_recheck_seconds < 0 or not math.isfinite(full_recheck_seconds):
        raise PublishError("full remote recheck interval must be finite and non-negative")
    if remote_recheck_seconds < 0 or not math.isfinite(remote_recheck_seconds):
        raise PublishError("remote recheck interval must be finite and non-negative")
    receipt = receipts / "site" / f"{item.kind}-{item.item_id}.verified.json"
    manifest_path = _site_path(item, "manifest.json")
    checked_at = now()
    checked_epoch = checked_at.timestamp()
    receipt_exists, remote_due = _remote_receipt_due(
        receipt, item, site=True, now_epoch=checked_epoch,
    )
    if receipt_exists and not remote_due:
        return {"kind": item.kind, "id": item.item_id, "state": "receipt_current", "remoteChecked": False}
    if receipt_exists:
        try:
            receipt_value = read_json(receipt)
            last_full = _timestamp(receipt_value.get("lastFullVerifiedAt"))
            full_due = (checked_at - last_full).total_seconds() >= full_recheck_seconds
            artifacts_ok = True
            for artifact in item.artifacts:
                artifact_path = _site_path(item, artifact.filename)
                if full_due:
                    if artifact.path.exists():
                        body = read_exact(artifact.path, artifact.size, artifact.sha256, limit=MAX_UPLOAD)
                        artifacts_ok = artifacts_ok and client.verify(artifact_path, body)
                    else:
                        artifacts_ok = artifacts_ok and client.verify_digest(
                            artifact_path, artifact.size, artifact.sha256
                        )
                else:
                    artifacts_ok = artifacts_ok and client.head_size_matches(artifact_path, artifact.size)
            if artifacts_ok and client.verify(manifest_path, item.site_manifest):
                if full_due:
                    receipt_value["lastFullVerifiedAt"] = checked_at.isoformat()
                receipt_value["lastRemoteVerifiedAt"] = checked_at.isoformat()
                receipt_value["nextRemoteCheckEpoch"] = _next_remote_check_epoch(
                    item, checked_epoch, remote_recheck_seconds,
                )
                atomic_json(receipt, receipt_value)
                return {"kind": item.kind, "id": item.item_id, "state": "already_verified", "remoteChecked": True}
        except PublishError:
            pass
    for artifact in item.artifacts:
        body = read_exact(artifact.path, artifact.size, artifact.sha256, limit=MAX_UPLOAD)
        client.put_verified(_site_path(item, artifact.filename), body, content_type=artifact.content_type)
    client.put_verified(manifest_path, item.site_manifest, content_type=JSON_TYPE)
    verified_at = checked_at.isoformat()
    value = {
        "schema": "jev-nethack-site-publish-receipt/v1", "kind": item.kind, "id": item.item_id,
        "sourceSha256": item.source_sha256, "manifestSha256": sha256_bytes(item.site_manifest),
        "verified": True, "verifiedAt": verified_at, "lastFullVerifiedAt": verified_at,
        "lastRemoteVerifiedAt": verified_at,
        "nextRemoteCheckEpoch": _next_remote_check_epoch(
            item, checked_epoch, remote_recheck_seconds,
        ),
    }
    atomic_json(receipt, value)
    return {"kind": item.kind, "id": item.item_id, "state": "published", "remoteChecked": True}


def durable_retry(
    *, operation_id: str, outbox_path: Path, operation: Callable[[], Any],
    delays: Sequence[float] = RETRY_DELAYS, sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> Any:
    attempts_before = 0
    if outbox_path.exists():
        prior = read_json(outbox_path)
        attempts_before = int(prior.get("attempts", 0))
        if float(prior.get("nextAttemptEpoch", 0)) > now():
            raise DeferredPublish(f"{operation_id} is waiting for durable backoff")
    atomic_json(
        outbox_path,
        {"schema": "jev-nethack-publish-outbox/v1", "operationId": operation_id,
         "state": "pending", "attempts": attempts_before, "nextAttemptEpoch": 0},
    )
    for local_attempt in range(len(delays) + 1):
        try:
            result = operation()
            outbox_path.unlink(missing_ok=True)
            _fsync_directory(outbox_path.parent)
            return result
        except (PublishError, subprocess.SubprocessError, OSError) as exc:
            attempts = attempts_before + local_attempt + 1
            delay = delays[min(local_attempt, len(delays) - 1)] if delays else 0.0
            atomic_json(
                outbox_path,
                {"schema": "jev-nethack-publish-outbox/v1", "operationId": operation_id,
                 "state": "retry", "attempts": attempts, "nextAttemptEpoch": now() + delay,
                 "errorType": type(exc).__name__, "error": str(exc)[:500]},
            )
            if local_attempt == len(delays):
                raise PublishError(f"{operation_id} failed after bounded retries") from exc
            sleep(delay)
    raise AssertionError("retry loop fell through")


def _bucket(timestamp: str, seconds: int) -> datetime:
    if seconds <= 0:
        raise PublishError("batch duration must be positive")
    value = _timestamp(timestamp)
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, timezone.utc)


def _release_receipt_path(receipts: Path, item: ClosedItem) -> Path:
    return receipts / "github-items" / f"{item.kind}-{item.source_id}.verified.json"


def _release_pending(
    receipts: Path, item: ClosedItem, github: "GitHubClient | None" = None, *,
    now_epoch: float | None = None,
    remote_recheck_seconds: float = REMOTE_EXISTENCE_RECHECK_SECONDS,
) -> bool:
    path = _release_receipt_path(receipts, item)
    if not path.exists() or not _receipt(path, item, site=False):
        return True
    if github is None:
        return False
    now_epoch = time.time() if now_epoch is None else now_epoch
    _, remote_due = _remote_receipt_due(path, item, site=False, now_epoch=now_epoch)
    if not remote_due:
        return False
    value = read_json(path)
    tag, asset, archive_sha = value.get("tag"), value.get("asset"), value.get("archiveSha256")
    if (
        not isinstance(tag, str) or not tag or not isinstance(asset, str) or not FILE_RE.fullmatch(asset)
        or not isinstance(archive_sha, str) or not SHA_RE.fullmatch(archive_sha)
    ):
        raise PublishError("GitHub item receipt lacks valid remote identity")
    if not github.asset_verified(tag, asset, archive_sha):
        return True
    checked_at = datetime.fromtimestamp(now_epoch, timezone.utc).isoformat()
    value["lastRemoteVerifiedAt"] = checked_at
    value["nextRemoteCheckEpoch"] = _staggered_next_epoch(
        f"github:{tag}:{asset}", now_epoch, remote_recheck_seconds,
    )
    atomic_json(path, value)
    return False


def reconcile_github_item_receipts(receipts: Path, items: Sequence[ClosedItem]) -> None:
    """Finish local receipts after a crash between batch and per-item commits."""
    missing = {
        (item.kind, item.source_id, item.source_sha256): item
        for item in items
        if item.release_files and not _release_receipt_path(receipts, item).exists()
    }
    if not missing:
        return
    batch_root = receipts / "github-batches"
    if not batch_root.is_dir():
        return
    for path in sorted(batch_root.glob("*.verified.json")):
        value = read_json(path)
        if value.get("verified") is not True or value.get("schema") != "jev-nethack-github-publish-receipt/v1":
            continue
        tag, asset, archive_sha = value.get("tag"), value.get("asset"), value.get("sha256")
        if (
            not all(isinstance(field, str) and field for field in (tag, asset))
            or not FILE_RE.fullmatch(asset) or not asset.endswith(".tar.gz")
            or not isinstance(
            archive_sha, str
            ) or not SHA_RE.fullmatch(archive_sha)
        ):
            continue
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            continue
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            key = (raw.get("kind"), raw.get("id"), raw.get("sourceSha256"))
            item = missing.get(key)
            if item is None:
                continue
            atomic_json(
                _release_receipt_path(receipts, item),
                {"schema": "jev-nethack-github-item-receipt/v1", "kind": item.kind,
                 "id": item.source_id, "sourceSha256": item.source_sha256,
                 "tag": tag, "asset": asset, "archiveSha256": archive_sha,
                 "verified": True, "verifiedAt": value.get("verifiedAt"),
                 "lastRemoteVerifiedAt": value.get("verifiedAt"),
                 "nextRemoteCheckEpoch": _staggered_next_epoch(
                     f"github:{tag}:{asset}", _timestamp(value.get("verifiedAt")).timestamp(),
                     REMOTE_EXISTENCE_RECHECK_SECONDS,
                 )},
            )
            missing.pop(key, None)
        all_committed = True
        for raw in raw_items:
            if not isinstance(raw, dict):
                all_committed = False
                break
            kind, item_id, source_sha = raw.get("kind"), raw.get("id"), raw.get("sourceSha256")
            if kind not in ("training", "recording") or not isinstance(item_id, str) or not SITE_ID_RE.fullmatch(
                item_id
            ) or not isinstance(source_sha, str) or not SHA_RE.fullmatch(source_sha):
                all_committed = False
                break
            item_path = receipts / "github-items" / f"{kind}-{item_id}.verified.json"
            if not item_path.exists():
                all_committed = False
                break
            committed = read_json(item_path)
            if (
                committed.get("verified") is not True or committed.get("sourceSha256") != source_sha
                or committed.get("tag") != tag or committed.get("asset") != asset
                or committed.get("archiveSha256") != archive_sha
            ):
                all_committed = False
                break
        if all_committed:
            github_outbox = receipts / "outbox" / f"github-{asset}.json"
            if github_outbox.exists():
                github_outbox.unlink()
                _fsync_directory(github_outbox.parent)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _add_tar_file(tar: tarfile.TarFile, name: str, source: ArchiveFile) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source.path, flags)
    except OSError as exc:
        raise PublishError("release source could not be opened safely") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != source.size:
            raise PublishError("release source inode changed")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != source.sha256:
                raise PublishError("release source bytes changed")
            stream.seek(0)
            tar.addfile(_tar_info(name, source.size), stream)
    finally:
        os.close(descriptor)


def build_release_archive(
    items: Sequence[ClosedItem], output_root: Path, bucket: datetime, batch_seconds: int = 600
) -> dict[str, Any]:
    if not items:
        raise PublishError("cannot build an empty release batch")
    files: list[dict[str, Any]] = []
    by_name: dict[str, ArchiveFile] = {}
    for item in sorted(items, key=lambda value: (value.kind, value.source_id)):
        if not item.release_files:
            raise PublishError("release item has no closed source files")
        for source in item.release_files:
            if source.name.startswith("/") or ".." in Path(source.name).parts or source.name in by_name:
                raise PublishError("release member is unsafe or duplicated")
            by_name[source.name] = source
            files.append({"path": source.name, "bytes": source.size, "sha256": source.sha256})
    label = bucket.strftime("%Y%m%dT%H%M%SZ")
    prefix = f"jev-nethack-archive-{label}"
    manifest = {
        "schema": RELEASE_SCHEMA, "batchSeconds": batch_seconds,
        "bucketStartedAt": bucket.isoformat(),
        "items": [
            {"kind": item.kind, "id": item.source_id, "sourceSha256": item.source_sha256,
             "startedAt": item.started_at, "endedAt": item.ended_at}
            for item in sorted(items, key=lambda value: (value.kind, value.source_id))
        ],
        "files": sorted(files, key=lambda value: value["path"]),
    }
    manifest_bytes = canonical_json(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{prefix}.", suffix=".tar.gz", dir=output_root)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.addfile(_tar_info(f"{prefix}/SHA256SUMS.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
                    for member_name, source in sorted(by_name.items()):
                        _add_tar_file(tar, f"{prefix}/{member_name}", source)
            raw.flush()
            os.fsync(raw.fileno())
        digest = sha256_file(temporary)
        archive = output_root / f"{prefix}-{digest[:16]}.tar.gz"
        if archive.exists():
            if sha256_file(archive) != digest:
                raise PublishError("release outbox name collision")
            temporary.unlink()
        else:
            os.replace(temporary, archive)
            _fsync_directory(output_root)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(archive), "asset": archive.name, "sha256": digest,
        "bucket": label, "tag": f"jev-nethack-archive-{bucket:%Y-%m-%d}",
        "items": list(items), "manifest": manifest,
    }


class GitHubClient:
    def __init__(
        self, repo: str, gh: str = "gh", runner: Callable[..., Any] = subprocess.run,
        timeout: float = 120.0,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise PublishError("GitHub repository must be owner/name")
        if timeout <= 0:
            raise PublishError("GitHub command timeout must be positive")
        self.repo, self.gh, self.runner, self.timeout = repo, gh, runner, timeout
        self._release_cache: dict[str, dict[str, Any] | None] = {}

    def _run(self, arguments: Sequence[str]) -> Any:
        return self.runner(
            [self.gh, *arguments], capture_output=True, text=True, check=False, timeout=self.timeout
        )

    def view(self, tag: str, *, refresh: bool = False) -> dict[str, Any] | None:
        if not refresh and tag in self._release_cache:
            return self._release_cache[tag]
        result = self._run(["release", "view", tag, "--repo", self.repo, "--json", "assets,url"])
        if result.returncode == 0:
            try:
                value = json.loads(result.stdout)
            except ValueError as exc:
                raise PublishError("GitHub returned invalid release JSON") from exc
            if not isinstance(value, dict) or not isinstance(value.get("assets"), list):
                raise PublishError("GitHub release JSON lacks assets")
            self._release_cache[tag] = value
            return value
        message = (result.stderr or "") + (result.stdout or "")
        if re.search(r"release not found|HTTP 404|not found", message, re.IGNORECASE):
            self._release_cache[tag] = None
            return None
        raise PublishError("GitHub release inspection failed")

    def ensure_release(self, tag: str) -> dict[str, Any]:
        release = self.view(tag)
        if release is not None:
            return release
        try:
            self._run(
                ["release", "create", tag, "--repo", self.repo, "--title", tag,
                 "--notes", "Immutable checksummed Jev NetHack archive batches"]
            )
        except subprocess.TimeoutExpired:
            pass
        release = self.view(tag, refresh=True)
        if release is None:
            raise PublishError("GitHub release creation could not be verified")
        return release

    @staticmethod
    def _asset(release: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
        return next((item for item in release.get("assets", []) if item.get("name") == name), None)

    def _download_verified(self, tag: str, asset_name: str, expected_sha: str) -> None:
        with tempfile.TemporaryDirectory(prefix="jev-nethack-gh-readback-") as directory:
            result = self._run(
                ["release", "download", tag, "--repo", self.repo,
                 "--pattern", asset_name, "--dir", directory]
            )
            downloaded = Path(directory) / asset_name
            if result.returncode != 0 or not downloaded.is_file() or sha256_file(downloaded) != expected_sha:
                raise PublishError("GitHub asset download checksum mismatch")

    def asset_verified(self, tag: str, asset_name: str, expected_sha: str) -> bool:
        release = self.view(tag)
        if release is None:
            return False
        asset = self._asset(release, asset_name)
        if asset is None:
            return False
        remote_digest = asset.get("digest")
        if isinstance(remote_digest, str) and remote_digest:
            if remote_digest.removeprefix("sha256:") != expected_sha:
                raise PublishError("GitHub contains a conflicting asset digest")
            return True
        self._download_verified(tag, asset_name, expected_sha)
        return True

    def upload_verified(self, tag: str, archive: Path, expected_sha: str) -> None:
        release = self.ensure_release(tag)
        asset = self._asset(release, archive.name)
        if asset is None:
            try:
                self._run(["release", "upload", tag, str(archive), "--repo", self.repo])
            except subprocess.TimeoutExpired:
                pass
            release = self.view(tag, refresh=True)
            asset = None if release is None else self._asset(release, archive.name)
        if asset is None:
            raise PublishError("GitHub asset upload could not be verified")
        remote_digest = asset.get("digest")
        if isinstance(remote_digest, str) and remote_digest and remote_digest.removeprefix("sha256:") != expected_sha:
            raise PublishError("GitHub contains a conflicting asset digest")
        self._download_verified(tag, archive.name, expected_sha)


def publish_github_batch(
    batch: Mapping[str, Any], github: GitHubClient, receipts: Path, *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    remote_recheck_seconds: float = REMOTE_EXISTENCE_RECHECK_SECONDS,
) -> dict[str, Any]:
    archive = Path(str(batch["archive"]))
    github.upload_verified(str(batch["tag"]), archive, str(batch["sha256"]))
    verified_time = now()
    verified_at = verified_time.isoformat()
    next_remote = _staggered_next_epoch(
        f"github:{batch['tag']}:{archive.name}", verified_time.timestamp(), remote_recheck_seconds,
    )
    batch_items = [
        {"kind": item.kind, "id": item.source_id, "sourceSha256": item.source_sha256}
        for item in batch["items"]
    ]
    atomic_json(
        receipts / "github-batches" / f"{archive.name}.verified.json",
        {"schema": "jev-nethack-github-publish-receipt/v1", "tag": batch["tag"],
         "asset": archive.name, "sha256": batch["sha256"], "verified": True,
         "verifiedAt": verified_at, "items": batch_items},
    )
    for item in batch["items"]:
        atomic_json(
            _release_receipt_path(receipts, item),
            {"schema": "jev-nethack-github-item-receipt/v1", "kind": item.kind,
             "id": item.source_id, "sourceSha256": item.source_sha256,
             "tag": batch["tag"], "asset": archive.name, "archiveSha256": batch["sha256"],
             "verified": True, "verifiedAt": verified_at, "lastRemoteVerifiedAt": verified_at,
             "nextRemoteCheckEpoch": next_remote},
        )
    return {"tag": batch["tag"], "asset": archive.name, "state": "published"}


class PublisherLock:
    def __init__(self, path: Path) -> None:
        self.path, self.stream = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise PublishError("publisher lock parent may not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise PublishError("publisher lock could not be opened safely") from exc
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
                raise PublishError("publisher lock must be a caller-owned regular file")
            if stat.S_IMODE(details.st_mode) & 0o077:
                raise PublishError("publisher lock must not be group/world accessible")
            self.stream = os.fdopen(descriptor, "r+", encoding="utf-8", closefd=True)
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise LockBusy("archive publisher is already running") from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"{os.getpid()}\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise PublishError("catalog dependency may not be a symlink")
    try:
        details = path.stat()
    except OSError as exc:
        raise PublishError("catalog dependency is missing") from exc
    if not stat.S_ISREG(details.st_mode):
        raise PublishError("catalog dependency is not a regular file")
    return {
        "path": str(path.resolve()), "device": details.st_dev, "inode": details.st_ino,
        "bytes": details.st_size, "mtimeNs": details.st_mtime_ns,
    }


def _serialize_item(item: ClosedItem) -> dict[str, Any]:
    return {
        "kind": item.kind, "itemId": item.item_id, "sourceId": item.source_id,
        "sourcePath": str(item.source_path), "sourceSha256": item.source_sha256,
        "startedAt": item.started_at, "endedAt": item.ended_at,
        "siteManifest": item.site_manifest.decode("utf-8"),
        "artifacts": [
            {"filename": value.filename, "path": str(value.path), "contentType": value.content_type,
             "bytes": value.size, "sha256": value.sha256}
            for value in item.artifacts
        ],
        "releaseFiles": [
            {"name": value.name, "path": str(value.path), "bytes": value.size, "sha256": value.sha256}
            for value in item.release_files
        ],
    }


def _catalog_path_allowed(
    path: Path, allowed_roots: Sequence[Path], *,
    tombstone_root: Path | None = None, size: int | None = None, digest: str | None = None,
) -> Path:
    absolute = Path(os.path.abspath(path))
    resolved = absolute.parent.resolve() / absolute.name
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
        raise PublishError("catalog path escapes its allowed roots")
    if resolved.is_symlink():
        raise PublishError("catalog path is not a regular file")
    if not resolved.is_file():
        if (
            tombstone_root is None or size is None or digest is None
            or not tombstone_covers(resolved, size, digest, tombstone_root)
        ):
            raise PublishError("catalog path is not a regular file")
    return resolved


def _deserialize_item(
    raw: Any, allowed_roots: Sequence[Path], tombstone_root: Path | None = None,
) -> ClosedItem:
    if not isinstance(raw, dict) or raw.get("kind") not in ("training", "recording"):
        raise PublishError("source catalog item is invalid")
    kind, item_id, source_id = raw["kind"], raw.get("itemId"), raw.get("sourceId")
    if not isinstance(item_id, str) or not SITE_ID_RE.fullmatch(item_id):
        raise PublishError("source catalog Site ID is invalid")
    if not isinstance(source_id, str) or not SITE_ID_RE.fullmatch(source_id):
        raise PublishError("source catalog source ID is invalid")
    source_sha = raw.get("sourceSha256")
    if not isinstance(source_sha, str) or not SHA_RE.fullmatch(source_sha):
        raise PublishError("source catalog digest is invalid")
    started_at, ended_at = raw.get("startedAt"), raw.get("endedAt")
    if _timestamp(ended_at) < _timestamp(started_at):
        raise PublishError("source catalog timestamps are invalid")
    manifest_text = raw.get("siteManifest")
    if not isinstance(manifest_text, str) or len(manifest_text.encode()) > MAX_MANIFEST:
        raise PublishError("source catalog manifest is invalid")
    try:
        manifest_value = json.loads(manifest_text)
    except ValueError as exc:
        raise PublishError("source catalog manifest is invalid") from exc
    if not isinstance(manifest_value, dict) or manifest_value.get("sessionId") != item_id:
        raise PublishError("source catalog manifest identity is invalid")
    artifacts: list[Artifact] = []
    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise PublishError("source catalog artifacts are invalid")
    for value in raw_artifacts:
        if not isinstance(value, dict):
            raise PublishError("source catalog artifact is invalid")
        filename, content_type = value.get("filename"), value.get("contentType")
        size, digest = value.get("bytes"), value.get("sha256")
        if (
            not isinstance(filename, str) or not FILE_RE.fullmatch(filename)
            or not isinstance(content_type, str) or not content_type
            or not isinstance(size, int) or not 0 < size <= MAX_LOCAL_ARTIFACT
            or not isinstance(digest, str) or not SHA_RE.fullmatch(digest)
        ):
            raise PublishError("source catalog artifact metadata is invalid")
        artifact_path = _catalog_path_allowed(
            Path(str(value.get("path"))), allowed_roots,
            tombstone_root=tombstone_root, size=size, digest=digest,
        )
        artifacts.append(Artifact(filename, artifact_path, content_type, size, digest))
    release_files: list[ArchiveFile] = []
    raw_release = raw.get("releaseFiles")
    if not isinstance(raw_release, list):
        raise PublishError("source catalog release files are invalid")
    for value in raw_release:
        if not isinstance(value, dict):
            raise PublishError("source catalog release file is invalid")
        name, size, digest = value.get("name"), value.get("bytes"), value.get("sha256")
        if (
            not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts
            or not isinstance(size, int) or not 0 < size <= MAX_LOCAL_ARTIFACT
            or not isinstance(digest, str) or not SHA_RE.fullmatch(digest)
        ):
            raise PublishError("source catalog release metadata is invalid")
        release_path = _catalog_path_allowed(
            Path(str(value.get("path"))), allowed_roots,
            tombstone_root=tombstone_root, size=size, digest=digest,
        )
        release_files.append(ArchiveFile(name, release_path, size, digest))
    source_path = _catalog_path_allowed(Path(str(raw.get("sourcePath"))), allowed_roots)
    return ClosedItem(
        kind, item_id, source_id, source_path, source_sha, started_at, ended_at,
        tuple(artifacts), manifest_text.encode("utf-8"), tuple(release_files),
    )


def _catalog_items(
    *, source: Path, kind: str, catalog_root: Path, allowed_roots: Sequence[Path],
    parse: Callable[[], list[ClosedItem]], now_epoch: float, scrub_seconds: float,
    tombstone_root: Path | None = None,
) -> list[ClosedItem]:
    if scrub_seconds < 0:
        raise PublishError("local scrub interval must not be negative")
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    cache_path = catalog_root / f"{kind}-{key}.json"
    if cache_path.exists():
        try:
            cached = read_json(cache_path, limit=4 * 1024 * 1024)
            last_scrub = float(cached.get("lastFullValidatedEpoch", -1))
            dependencies = cached.get("dependencies")
            raw_items = cached.get("items")
            cached_items = (
                [_deserialize_item(item, allowed_roots, tombstone_root) for item in raw_items]
                if isinstance(raw_items, list) and raw_items else []
            )
            has_pruned = any(
                not value.path.exists()
                for item in cached_items for value in (*item.artifacts, *item.release_files)
            )
            dependencies_current = (
                isinstance(dependencies, list) and dependencies
                and all(
                    (
                        _file_fingerprint(Path(str(item.get("path")))) == item
                        if Path(str(item.get("path"))).exists() else
                        any(
                            Path(str(item.get("path"))) == value.path
                            and tombstone_root is not None
                            and tombstone_covers(value.path, value.size, value.sha256, tombstone_root)
                            for closed in cached_items
                            for value in (*closed.artifacts, *closed.release_files)
                        )
                    )
                    for item in dependencies if isinstance(item, dict)
                )
                and len([item for item in dependencies if isinstance(item, dict)]) == len(dependencies)
            )
            if (
                cached.get("schema") == "jev-nethack-source-catalog/v1"
                and cached.get("sourcePath") == str(source.resolve())
                and (0 <= now_epoch - last_scrub < scrub_seconds or has_pruned)
                and dependencies_current
            ):
                return cached_items
        except (PublishError, OSError, TypeError, ValueError):
            pass
    items = parse()
    dependency_paths = {
        item.source_path.resolve() for item in items
    } | {
        value.path.resolve() for item in items for value in (*item.artifacts, *item.release_files)
    }
    atomic_json(
        cache_path,
        {"schema": "jev-nethack-source-catalog/v1", "sourcePath": str(source.resolve()),
         "lastFullValidatedEpoch": now_epoch,
         "dependencies": [_file_fingerprint(path) for path in sorted(dependency_paths)],
         "items": [_serialize_item(item) for item in items]},
    )
    return items


def collect_items(
    data_root: Path, recordings_root: Path | None, derived_root: Path,
    *, catalog_root: Path | None = None, now_epoch: float | None = None,
    scrub_seconds: float = FULL_LOCAL_SCRUB_SECONDS, tombstone_root: Path | None = None,
) -> tuple[list[ClosedItem], list[dict[str, str]]]:
    items: list[ClosedItem] = []
    errors: list[dict[str, str]] = []
    catalog_root = catalog_root or derived_root.parent / "source-catalog"
    now_epoch = time.time() if now_epoch is None else now_epoch
    allowed_roots = [data_root, derived_root]
    if recordings_root is not None:
        allowed_roots.append(recordings_root)
    for marker in discover_training(data_root):
        try:
            items.extend(
                _catalog_items(
                    source=marker, kind="training", catalog_root=catalog_root,
                    allowed_roots=allowed_roots,
                    parse=lambda marker=marker: parse_training_marker(marker, data_root, derived_root),
                    now_epoch=now_epoch, scrub_seconds=scrub_seconds,
                    tombstone_root=tombstone_root,
                )
            )
        except PublishError as exc:
            errors.append({"path": str(marker), "error": str(exc)})
    if recordings_root is not None:
        for manifest in discover_recordings(recordings_root):
            try:
                items.extend(
                    _catalog_items(
                        source=manifest, kind="recording", catalog_root=catalog_root,
                        allowed_roots=allowed_roots,
                        parse=lambda manifest=manifest: [parse_recording_manifest(manifest, recordings_root)],
                        now_epoch=now_epoch, scrub_seconds=scrub_seconds,
                        tombstone_root=tombstone_root,
                    )
                )
            except PublishError as exc:
                errors.append({"path": str(manifest), "error": str(exc)})
    identities: set[tuple[str, str]] = set()
    for item in items:
        key = (item.kind, item.item_id)
        if key in identities:
            raise PublishError(f"duplicate immutable item identity: {key}")
        identities.add(key)
    return sorted(items, key=lambda item: (item.ended_at, item.kind, item.item_id)), errors


def recover_inactive_training_tails(
    data_root: Path, recovered_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Close complete prefixes from dead fragments into a separate data root."""
    fragments_root = data_root / "fragments"
    if not fragments_root.is_dir() or fragments_root.is_symlink():
        return [], []
    pending: dict[Path, list[int]] = {}
    for fragment in sorted(fragments_root.iterdir()):
        if fragment.is_symlink() or not fragment.is_dir():
            continue
        training = fragment / "training"
        if training.is_symlink() or not training.is_dir():
            continue
        for pack in sorted(training.glob("transitions-*.npzpack")):
            match = re.fullmatch(r"transitions-(\d{6})\.npzpack", pack.name)
            if not match or pack.is_symlink() or not pack.is_file():
                continue
            number = int(match.group(1))
            source_marker = training / f"transitions-{number:06d}.complete.json"
            recovered_marker = (
                recovered_root / "fragments" / fragment.name / "training"
                / f"transitions-{number:06d}.complete.json"
            )
            if not source_marker.exists() and not source_marker.is_symlink() and not recovered_marker.exists():
                pending.setdefault(fragment, []).append(number)
    if not pending:
        return [], []
    state_path = data_root / "state.json"
    try:
        state = read_json(state_path, limit=1024 * 1024)
        active_reference = state.get("activeFragment")
        active_fragment = (
            _inside(data_root / active_reference, data_root)
            if isinstance(active_reference, str) and active_reference else None
        )
    except PublishError as exc:
        return [], [{"path": str(state_path), "error": f"training recovery state: {exc}"}]
    try:
        from recover_training_tail import TrainingTailRecoveryError, recover_training_tail
    except (ImportError, OSError) as exc:
        return [], [{"path": str(data_root), "error": f"training recovery unavailable: {exc}"}]
    recovered: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for fragment, shard_numbers in pending.items():
        if active_fragment is not None and fragment.resolve() == active_fragment:
            continue
        try:
            recovered.append(
                recover_training_tail(
                    fragment, recovered_root, state_path=state_path,
                    shard_numbers=shard_numbers,
                )
            )
        except TrainingTailRecoveryError as exc:
            errors.append({"path": str(fragment), "error": f"training recovery refused: {exc}"})
    return recovered, errors


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _recovery_root_path(value: str | None, runtime_root: Path, default_name: str) -> Path:
    is_default = value is None
    raw = runtime_root / default_name if is_default else Path(value).expanduser()
    raw_absolute = Path(os.path.abspath(raw))
    try:
        leaf_details = raw_absolute.lstat()
    except FileNotFoundError:
        leaf_details = None
    except OSError as exc:
        raise PublishError("recovery root path cannot be inspected") from exc
    if leaf_details is not None:
        if stat.S_ISLNK(leaf_details.st_mode):
            raise PublishError("recovery root path contains a symlink")
        if not stat.S_ISDIR(leaf_details.st_mode):
            raise PublishError("recovery root path contains a non-directory")

    candidate = (
        raw_absolute if is_default
        else raw_absolute.parent.resolve() / raw_absolute.name
    )
    if is_default and (
        candidate.parent != runtime_root or candidate.parent.resolve() != runtime_root
    ):
        raise PublishError("default recovery root escapes the runtime root")
    return candidate


def _video_tail_index(fragment: Path) -> int | None:
    indexes: list[int] = []
    try:
        entries = list(fragment.iterdir())
    except OSError as exc:
        raise PublishError("recording fragment cannot be scanned") from exc
    for entry in entries:
        match = VIDEO_SEGMENT_RE.fullmatch(entry.name)
        if match is None:
            continue
        try:
            details = entry.lstat()
        except OSError as exc:
            raise PublishError("recording event stream cannot be inspected") from exc
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise PublishError("recording event stream is not a regular non-symlink file")
        indexes.append(int(match.group(1)))
    return max(indexes) if indexes else None


def _ensure_private_recovery_root(path: Path) -> None:
    try:
        created = not path.exists()
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise PublishError("video recovery root is not a real directory")
        if created:
            path.chmod(0o700)
            _fsync_directory(path.parent)
    except OSError as exc:
        raise PublishError("video recovery root cannot be prepared") from exc


def _closed_recording_marker_exists(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PublishError("recording completion marker cannot be inspected") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise PublishError("recording completion marker is not a regular non-symlink file")
    return True


def recover_inactive_video_tails(
    data_root: Path, recovered_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Derive the final closed video segment for exactly mapped dead producers."""
    fragments_root = data_root / "broadcast-fragments"
    if not fragments_root.is_dir() or fragments_root.is_symlink():
        return [], []
    try:
        from recover_video_tail import (
            ActiveFragmentDeferred,
            ExactProducerEvidence,
            NoRecoverableTail,
            TailRecoveryError,
            inspect_exact_producer,
            recover_video_tail,
        )
    except (ImportError, OSError) as exc:
        return [], [{"path": str(data_root), "error": f"video recovery unavailable: {exc}"}]

    recovered: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        fragments = sorted(fragments_root.iterdir())
    except OSError as exc:
        return [], [{"path": str(fragments_root), "error": f"video recovery scan: {exc}"}]
    for fragment in fragments:
        if fragment.is_symlink() or not fragment.is_dir():
            continue
        try:
            segment_index = _video_tail_index(fragment)
            if segment_index is None:
                continue
            stem = f"segment-{segment_index:04d}"
            source_manifest = fragment / f"{stem}.manifest.json"
            if _closed_recording_marker_exists(source_manifest):
                continue

            output_fragment = recovered_root / fragment.name
            recovered_manifest = output_fragment / f"{stem}.manifest.json"
            if _closed_recording_marker_exists(recovered_manifest):
                continue

            _ensure_private_recovery_root(recovered_root)
            evidence = ExactProducerEvidence(data_root)
            producer = inspect_exact_producer(fragment, evidence)
            result = recover_video_tail(
                fragment,
                exact_producer=evidence,
                output_fragment=output_fragment,
                derived_root=recovered_root,
                segment_index=segment_index,
            )
            recovered.append(
                {
                    **result,
                    "fragment": fragment.name,
                    "segmentIndex": segment_index,
                    "producerPid": producer.pid,
                    "producerStamp": producer.stamp,
                }
            )
        except ActiveFragmentDeferred:
            recovered.append(
                {"status": "deferred_active", "fragment": fragment.name,
                 "segmentIndex": segment_index}
            )
        except NoRecoverableTail:
            recovered.append(
                {"status": "no_recoverable_tail", "fragment": fragment.name,
                 "segmentIndex": segment_index}
            )
        except (PublishError, TailRecoveryError) as exc:
            errors.append({"path": str(fragment), "error": f"video recovery refused: {exc}"})
    return recovered, errors


def _merge_closed_items(groups: Sequence[Sequence[ClosedItem]]) -> list[ClosedItem]:
    merged: list[ClosedItem] = []
    identities: set[tuple[str, str]] = set()
    for group in groups:
        for item in group:
            key = (item.kind, item.item_id)
            if key in identities:
                raise PublishError(f"duplicate immutable item identity: {key}")
            identities.add(key)
            merged.append(item)
    return sorted(merged, key=lambda item: (item.ended_at, item.kind, item.item_id))


def run_once(
    args: argparse.Namespace, *, delays: Sequence[float] = RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
    now_epoch: Callable[[], float] = time.time,
) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    recordings_root = Path(args.recordings_root).resolve() if args.recordings_root else None
    recovered_training_root = _recovery_root_path(
        getattr(args, "recovered_training_root", None), runtime_root, "recovered-training"
    )
    recovered_recordings_root = _recovery_root_path(
        getattr(args, "recovered_recordings_root", None), runtime_root, "recovered-recordings"
    )
    if recovered_training_root == data_root or recovered_training_root.is_relative_to(data_root):
        raise PublishError("recovered training root must be separate from the primary data root")
    if _paths_overlap(recovered_recordings_root, data_root) or (
        recordings_root is not None and _paths_overlap(recovered_recordings_root, recordings_root)
    ):
        raise PublishError("recovered recordings root must be separate from source data")
    if _paths_overlap(recovered_recordings_root, recovered_training_root):
        raise PublishError("recovered training and recording roots must be separate")
    receipts, outbox = runtime_root / "publish-receipts", runtime_root / "publish-receipts" / "outbox"
    tombstone_root = runtime_root / "recording-tombstones"
    recording_roots = tuple(
        root for root in (recordings_root, recovered_recordings_root)
        if root is not None
    )
    retention_recovered: list[str] = []
    retention_recovery_error: str | None = None
    if recordings_root is not None:
        try:
            retention_recovered = recover_tombstones(tombstone_root, recording_roots)
        except RetentionError as exc:
            retention_recovery_error = str(exc)
    outbox.mkdir(parents=True, exist_ok=True)
    recovered, recovery_errors = recover_inactive_training_tails(data_root, recovered_training_root)
    recovered_video: list[dict[str, Any]] = []
    video_recovery_errors: list[dict[str, str]] = []
    if recordings_root is not None:
        recovered_video, video_recovery_errors = recover_inactive_video_tails(
            data_root, recovered_recordings_root
        )
    primary_items, source_errors = collect_items(
        data_root, recordings_root, runtime_root / "repackaged",
        catalog_root=runtime_root / "source-catalog",
        scrub_seconds=float(getattr(args, "local_scrub_seconds", FULL_LOCAL_SCRUB_SECONDS)),
        tombstone_root=tombstone_root,
    )
    recovered_items: list[ClosedItem] = []
    if recovered_training_root.is_dir():
        recovered_items, recovered_errors = collect_items(
            recovered_training_root, None, runtime_root / "repackaged",
            catalog_root=runtime_root / "source-catalog",
            scrub_seconds=float(getattr(args, "local_scrub_seconds", FULL_LOCAL_SCRUB_SECONDS)),
            tombstone_root=tombstone_root,
        )
        source_errors.extend(recovered_errors)
    recovered_recording_items: list[ClosedItem] = []
    if recordings_root is not None and recovered_recordings_root.is_dir():
        recovered_recording_items, recovered_recording_errors = collect_items(
            recovered_recordings_root, recovered_recordings_root, runtime_root / "repackaged",
            catalog_root=runtime_root / "source-catalog",
            scrub_seconds=float(getattr(args, "local_scrub_seconds", FULL_LOCAL_SCRUB_SECONDS)),
            tombstone_root=tombstone_root,
        )
        source_errors.extend(recovered_recording_errors)
    items = _merge_closed_items((primary_items, recovered_items, recovered_recording_items))
    cycle_epoch = now_epoch()
    cycle_time = datetime.fromtimestamp(cycle_epoch, timezone.utc)
    report: dict[str, Any] = {
        "sources": len(items), "trainingRecovery": recovered, "videoRecovery": recovered_video,
        "site": [], "github": [], "recordingRetention": {
            "budgetBytes": int(getattr(args, "recording_cache_bytes", DEFAULT_RECORDING_CACHE_BYTES)),
            "recovered": retention_recovered, "state": "pending" if recordings_root else "not_configured",
        },
        "releaseCache": {
            "budgetBytes": int(getattr(args, "release_cache_bytes", DEFAULT_RELEASE_CACHE_BYTES)),
            "state": "pending",
        },
        "errors": [*recovery_errors, *video_recovery_errors, *source_errors],
    }
    if retention_recovery_error is not None:
        report["errors"].append(
            {"path": str(tombstone_root), "error": f"recording retention recovery: {retention_recovery_error}"}
        )
    site: SiteClient | None = None
    github: GitHubClient | None = None
    if args.site_url:
        site = SiteClient(args.site_url, read_token(Path(args.token_file)))
        remote_budget = int(getattr(args, "remote_recheck_budget", REMOTE_RECHECK_BUDGET))
        remote_interval = float(
            getattr(args, "remote_recheck_seconds", REMOTE_EXISTENCE_RECHECK_SECONDS)
        )
        if remote_budget <= 0:
            raise PublishError("remote recheck budget must be positive when Site publishing is enabled")
        if remote_interval < 0 or not math.isfinite(remote_interval):
            raise PublishError("remote recheck interval must be finite and non-negative")
        remote_checks = 0
        for item in items:
            try:
                receipt_path = receipts / "site" / f"{item.kind}-{item.item_id}.verified.json"
                operation_outbox = outbox / f"site-{item.kind}-{item.item_id}.json"
                receipt_exists, remote_due = _remote_receipt_due(
                    receipt_path, item, site=True, now_epoch=cycle_epoch,
                )
                if receipt_exists and not operation_outbox.exists():
                    if not remote_due:
                        report["site"].append(
                            {"kind": item.kind, "id": item.item_id, "state": "receipt_current",
                             "remoteChecked": False}
                        )
                        continue
                    if remote_checks >= remote_budget:
                        report["site"].append(
                            {"kind": item.kind, "id": item.item_id, "state": "recheck_deferred",
                             "remoteChecked": False}
                        )
                        continue
                    remote_checks += 1
                report["site"].append(
                    durable_retry(
                        operation_id=f"site:{item.kind}:{item.item_id}",
                        outbox_path=operation_outbox,
                        operation=lambda item=item: publish_site_item(
                            item, site, receipts, remote_recheck_seconds=remote_interval,
                            now=lambda: cycle_time,
                        ),
                        delays=delays, sleep=sleep,
                    )
                )
            except PublishError as exc:
                report["errors"].append({"path": str(item.source_path), "error": str(exc)})
    if args.github_repo:
        github = GitHubClient(args.github_repo, gh=args.gh)
        release_items: dict[tuple[str, str], ClosedItem] = {}
        for item in items:
            if item.release_files:
                release_items[(item.kind, item.source_id)] = item
        reconcile_github_item_receipts(receipts, list(release_items.values()))
        pending: list[ClosedItem] = []
        github_remote_tags: set[str] = set()
        remote_budget = int(getattr(args, "remote_recheck_budget", REMOTE_RECHECK_BUDGET))
        remote_interval = float(
            getattr(args, "remote_recheck_seconds", REMOTE_EXISTENCE_RECHECK_SECONDS)
        )
        for item in release_items.values():
            try:
                item_receipt = _release_receipt_path(receipts, item)
                receipt_exists, remote_due = _remote_receipt_due(
                    item_receipt, item, site=False, now_epoch=cycle_epoch,
                )
                if receipt_exists:
                    if not remote_due:
                        continue
                    receipt_value = read_json(item_receipt)
                    remote_tag = receipt_value.get("tag")
                    if not isinstance(remote_tag, str) or not remote_tag:
                        raise PublishError("GitHub item receipt lacks a release tag")
                    if remote_tag not in github_remote_tags and len(github_remote_tags) >= remote_budget:
                        continue
                    github_remote_tags.add(remote_tag)
                if _release_pending(
                    receipts, item, github, now_epoch=cycle_epoch,
                    remote_recheck_seconds=remote_interval,
                ):
                    missing_pruned = [
                        source for source in item.release_files
                        if not source.path.exists()
                        and tombstone_covers(
                            source.path, source.size, source.sha256, tombstone_root
                        )
                    ]
                    if missing_pruned:
                        raise PublishError(
                            "GitHub asset is unavailable and its local recording source "
                            "was intentionally pruned; archive rebuild is impossible"
                        )
                    pending.append(item)
            except PublishError as exc:
                report["errors"].append({"path": str(item.source_path), "error": str(exc)})
        groups: dict[datetime, list[ClosedItem]] = {}
        for item in pending:
            groups.setdefault(_bucket(item.ended_at, args.batch_seconds), []).append(item)
        for bucket, grouped in sorted(groups.items()):
            try:
                batch = build_release_archive(
                    grouped, runtime_root / "release-outbox", bucket, args.batch_seconds
                )
                report["github"].append(
                    durable_retry(
                        operation_id=f"github:{batch['asset']}",
                        outbox_path=outbox / f"github-{batch['asset']}.json",
                        operation=lambda batch=batch: publish_github_batch(
                            batch, github, receipts, now=lambda: cycle_time,
                            remote_recheck_seconds=remote_interval,
                        ),
                        delays=delays, sleep=sleep,
                    )
                )
            except PublishError as exc:
                report["errors"].append({"path": bucket.isoformat(), "error": str(exc)})
    if recordings_root is not None:
        budget = int(getattr(args, "recording_cache_bytes", DEFAULT_RECORDING_CACHE_BYTES))
        try:
            retention = prune_recording_cache(
                items, recording_roots, budget_bytes=budget,
                tombstone_root=tombstone_root, catalog_root=runtime_root / "source-catalog",
                receipts=receipts,
                site_verify=(
                    site.verify_digest if site is not None
                    else lambda _path, _size, _digest: False
                ),
                github_verify=(
                    github.asset_verified if github is not None
                    else lambda _tag, _asset, _digest: False
                ),
                site_path=_site_path,
            )
            retention["recovered"] = [
                *retention_recovered, *retention.get("recovered", []),
            ]
            report["recordingRetention"] = retention
            if not retention["withinBudget"] and not retention["disabled"]:
                report["errors"].append(
                    {
                        "path": str(recordings_root),
                        "error": (
                            f"recording cache remains above {budget} bytes "
                            f"({retention['afterBytes']} bytes retained)"
                        ),
                    }
                )
        except RetentionError as exc:
            report["recordingRetention"] = {
                "budgetBytes": budget, "state": "failed", "error": str(exc),
                "recovered": retention_recovered,
            }
            report["errors"].append(
                {"path": str(recordings_root), "error": f"recording retention: {exc}"}
            )
    release_budget = int(getattr(args, "release_cache_bytes", DEFAULT_RELEASE_CACHE_BYTES))
    try:
        release_cache = prune_release_cache(
            runtime_root / "release-outbox", receipts, budget_bytes=release_budget,
            github_verify=(
                github.asset_verified if github is not None
                else lambda _tag, _asset, _digest: False
            ),
        )
        report["releaseCache"] = release_cache
        if not release_cache["withinBudget"] and not release_cache["disabled"]:
            report["errors"].append(
                {
                    "path": str(runtime_root / "release-outbox"),
                    "error": (
                        f"verified release cache remains above {release_budget} bytes "
                        f"({release_cache['afterBytes']} bytes retained)"
                    ),
                }
            )
    except RetentionError as exc:
        report["releaseCache"] = {
            "budgetBytes": release_budget, "state": "failed", "error": str(exc),
        }
        report["errors"].append(
            {"path": str(runtime_root / "release-outbox"), "error": f"release cache: {exc}"}
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--recordings-root")
    parser.add_argument("--recovered-training-root")
    parser.add_argument("--recovered-recordings-root")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--site-url")
    parser.add_argument("--token-file")
    parser.add_argument("--github-repo")
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--batch-seconds", type=int, default=600)
    parser.add_argument("--local-scrub-seconds", type=float, default=FULL_LOCAL_SCRUB_SECONDS)
    parser.add_argument("--remote-recheck-seconds", type=float, default=REMOTE_EXISTENCE_RECHECK_SECONDS)
    parser.add_argument("--remote-recheck-budget", type=int, default=REMOTE_RECHECK_BUDGET)
    parser.add_argument(
        "--recording-cache-bytes", type=int, default=DEFAULT_RECORDING_CACHE_BYTES,
        help="local closed MP4/segment JSONL budget; zero disables pruning",
    )
    parser.add_argument(
        "--release-cache-bytes", type=int, default=DEFAULT_RELEASE_CACHE_BYTES,
        help="local verified release archive budget; zero disables cleanup",
    )
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.site_url) != bool(args.token_file):
        parser.error("--site-url and --token-file must be supplied together")
    if (
        args.interval <= 0 or args.batch_seconds <= 0 or args.local_scrub_seconds < 0
        or args.remote_recheck_seconds < 0 or not math.isfinite(args.remote_recheck_seconds)
        or args.remote_recheck_budget <= 0 or args.recording_cache_bytes < 0
        or args.release_cache_bytes < 0
    ):
        parser.error("intervals, batch size, and remote budget must be positive; scrub/check/cache values must be non-negative")
    runtime_root = Path(args.runtime_root).resolve()
    with PublisherLock(runtime_root / "publisher.lock"):
        while True:
            report = run_once(args)
            print(json.dumps(report, sort_keys=True), flush=True)
            if args.once:
                return 1 if report["errors"] else 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
