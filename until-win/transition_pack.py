"""Crash-recoverable, checksummed transition shards for Jev NetHack.

Each pack starts with ``JEVNHNPZ1\n`` and contains independently readable
frames. A frame is an unsigned 64-bit big-endian payload length, the SHA-256
of the payload, and one ordinary ``numpy.savez_compressed`` payload. A crash
can damage only the trailing frame; all complete, hash-valid frames remain
readable without trusting the companion index.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import struct
from typing import Any, BinaryIO, Iterator, Mapping

import numpy as np


SCHEMA = "jev-nethack-transition-pack/v1"
MAGIC = b"JEVNHNPZ1\n"
FRAME_HEADER = struct.Struct(">Q32s")
CONTENT_TYPE = "application/vnd.jev-nethack.transitions+npz"
INDEX_CONTENT_TYPE = "application/x-ndjson"
PUBLIC_ARTIFACT_MAX_BYTES = 32 * 1024 * 1024
MAX_FRAME_BYTES = PUBLIC_ARTIFACT_MAX_BYTES - len(MAGIC) - FRAME_HEADER.size


class TransitionPackError(Exception):
    """The pack is malformed or cannot be written safely."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON receipt durably without exposing a partial valid-looking file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    data = json.dumps(dict(value), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _observation_arrays(prefix: str, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if not isinstance(key, str) or not key:
            raise TransitionPackError("observation keys must be non-empty strings")
        # NLE reuses its buffers. np.array(copy=True) is required before the
        # next environment step and keeps object arrays out of the artifact.
        array = np.array(value, copy=True)
        if array.dtype.hasobject:
            raise TransitionPackError(f"observation {key!r} has object dtype")
        arrays[f"{prefix}__{key}"] = array
    return arrays


def encode_transition(
    *,
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bytes:
    """Encode one complete transition as an ordinary compressed NPZ payload."""
    normalized = dict(metadata)
    normalized["schema"] = SCHEMA
    if not isinstance(normalized.get("eventId"), str) or not normalized["eventId"]:
        raise TransitionPackError("metadata.eventId must be a non-empty string")
    arrays = _observation_arrays("obs", observation)
    arrays.update(_observation_arrays("next_obs", next_observation))
    arrays["metadata_json"] = np.frombuffer(_canonical_json(normalized), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def decode_transition(payload: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Decode a payload with pickle disabled and return arrays plus metadata."""
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise TransitionPackError("transition NPZ payload is invalid") from exc
    encoded = arrays.pop("metadata_json", None)
    if encoded is None or encoded.dtype != np.uint8:
        raise TransitionPackError("transition metadata_json is missing or invalid")
    try:
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TransitionPackError("transition metadata JSON is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise TransitionPackError("transition schema is unsupported")
    return arrays, metadata


@dataclass(frozen=True)
class ScannedFrame:
    offset: int
    payload_bytes: int
    sha256: str
    payload: bytes
    metadata: Mapping[str, Any]


def scan_pack(path: Path) -> Iterator[ScannedFrame]:
    """Yield complete frames; ignore only an incomplete trailing frame.

    A fully present frame with a bad digest or invalid payload is corruption,
    not a recoverable crash tail, and raises :class:`TransitionPackError`.
    """
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise TransitionPackError("transition pack magic is invalid")
        while True:
            offset = stream.tell()
            header = stream.read(FRAME_HEADER.size)
            if not header:
                return
            if len(header) < FRAME_HEADER.size:
                return
            payload_length, expected_digest = FRAME_HEADER.unpack(header)
            if payload_length <= 0 or payload_length > MAX_FRAME_BYTES:
                raise TransitionPackError("transition frame length is invalid")
            payload = stream.read(payload_length)
            if len(payload) < payload_length:
                return
            actual_digest = hashlib.sha256(payload).digest()
            if actual_digest != expected_digest:
                raise TransitionPackError("transition frame digest mismatch")
            _, metadata = decode_transition(payload)
            yield ScannedFrame(
                offset=offset,
                payload_bytes=payload_length,
                sha256=actual_digest.hex(),
                payload=payload,
                metadata=metadata,
            )


def validate_shard(pack_path: Path) -> dict[str, Any]:
    """Validate a pack, its durable index prefix, and an optional completion marker.

    An unmarked shard may be the result of a hard crash. Its index may lag the
    pack because the pack frame is fsynced first, but it may never describe a
    frame that is absent or different. A marked shard is immutable and must
    match both artifacts, their hashes, sizes, and record counts exactly.
    """
    if pack_path.suffix != ".npzpack":
        raise TransitionPackError("transition pack filename is invalid")
    stem = pack_path.name.removesuffix(".npzpack")
    index_path = pack_path.with_name(f"{stem}.index.jsonl")
    marker_path = pack_path.with_name(f"{stem}.complete.json")
    if not index_path.is_file():
        raise TransitionPackError("transition pack index is missing")

    frames = list(scan_pack(pack_path))
    indexed: list[dict[str, Any]] = []
    raw_index = index_path.read_bytes()
    lines = raw_index.splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            if not marker_path.exists() and number == len(lines) and not line.endswith(b"\n"):
                break
            raise TransitionPackError("transition index contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise TransitionPackError("transition index record is not an object")
        indexed.append(value)
    if len(indexed) > len(frames):
        raise TransitionPackError("transition index is ahead of its pack")
    for offset, (index_record, frame) in enumerate(zip(indexed, frames), 1):
        expected = {
            "record": offset,
            "offset": frame.offset,
            "payloadBytes": frame.payload_bytes,
            "sha256": frame.sha256,
            "eventId": frame.metadata.get("eventId"),
        }
        if any(index_record.get(key) != value for key, value in expected.items()):
            raise TransitionPackError(f"transition index disagrees with pack at record {offset}")

    frame_end = len(MAGIC)
    if frames:
        final = frames[-1]
        frame_end = final.offset + FRAME_HEADER.size + final.payload_bytes
    partial_tail = pack_path.stat().st_size != frame_end
    completed = False
    marker: dict[str, Any] | None = None
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text())
        except ValueError as exc:
            raise TransitionPackError("transition completion marker is invalid JSON") from exc
        if not isinstance(marker, dict) or marker.get("completed") is not True:
            raise TransitionPackError("transition completion marker is invalid")
        if partial_tail or len(indexed) != len(frames):
            raise TransitionPackError("completed transition shard has an incomplete tail or index")
        if marker.get("records") != len(frames) or marker.get("transitionCount") != len(frames):
            raise TransitionPackError("transition completion marker count is invalid")
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 2:
            raise TransitionPackError("transition completion marker artifacts are invalid")
        by_name = {
            item.get("filename"): item
            for item in artifacts
            if isinstance(item, dict) and isinstance(item.get("filename"), str)
        }
        for artifact_path in (pack_path, index_path):
            artifact = by_name.get(artifact_path.name)
            if not isinstance(artifact, dict):
                raise TransitionPackError("transition completion marker is missing an artifact")
            if artifact.get("bytes") != artifact_path.stat().st_size:
                raise TransitionPackError("transition artifact size does not match completion marker")
            if artifact.get("sha256") != _sha256_file(artifact_path):
                raise TransitionPackError("transition artifact hash does not match completion marker")
            if artifact.get("records") != len(frames):
                raise TransitionPackError("transition artifact record count does not match completion marker")
        completed = True

    return {
        "pack": pack_path,
        "index": index_path,
        "marker": marker_path if completed else None,
        "frames": frames,
        "records": len(frames),
        "indexedRecords": len(indexed),
        "partialTail": partial_tail,
        "completed": completed,
    }


class TransitionPackWriter:
    """Append durable transitions and rotate bounded shards."""

    def __init__(
        self,
        root: Path,
        *,
        max_records_per_shard: int = 128,
        max_bytes_per_shard: int = PUBLIC_ARTIFACT_MAX_BYTES,
    ) -> None:
        if max_records_per_shard <= 0 or max_bytes_per_shard <= len(MAGIC) + FRAME_HEADER.size:
            raise TransitionPackError("transition shard bounds must be positive")
        if max_bytes_per_shard > PUBLIC_ARTIFACT_MAX_BYTES:
            raise TransitionPackError("transition shard bound exceeds the public artifact limit")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
        self.shard_number = 0
        self.record_number = 0
        self.total_records = 0
        self._pack: BinaryIO | None = None
        self._index: Any = None
        self.pack_path: Path | None = None
        self.index_path: Path | None = None
        self.artifacts: list[dict[str, Any]] = []
        self._shard_opened_at: str | None = None
        self._shard_first_at: str | None = None
        self._shard_last_at: str | None = None
        self._open_shard()

    def _open_shard(self) -> None:
        self.shard_number += 1
        stem = f"transitions-{self.shard_number:06d}"
        self.pack_path = self.root / f"{stem}.npzpack"
        self.index_path = self.root / f"{stem}.index.jsonl"
        self._pack = self.pack_path.open("xb")
        self._pack.write(MAGIC)
        self._pack.flush()
        os.fsync(self._pack.fileno())
        self._index = self.index_path.open("x", encoding="utf-8")
        self.record_number = 0
        self._shard_opened_at = datetime.now(timezone.utc).isoformat()
        self._shard_first_at = None
        self._shard_last_at = None
        _fsync_directory(self.root)

    def _finish_shard(self) -> None:
        if self._pack is None or self._index is None or self.pack_path is None or self.index_path is None:
            return
        self._pack.flush()
        os.fsync(self._pack.fileno())
        self._index.flush()
        os.fsync(self._index.fileno())
        self._pack.close()
        self._index.close()
        shard_artifacts = [
            {
                "filename": self.pack_path.name,
                "contentType": CONTENT_TYPE,
                "bytes": self.pack_path.stat().st_size,
                "sha256": _sha256_file(self.pack_path),
                "records": self.record_number,
            },
            {
                "filename": self.index_path.name,
                "contentType": INDEX_CONTENT_TYPE,
                "bytes": self.index_path.stat().st_size,
                "sha256": _sha256_file(self.index_path),
                "records": self.record_number,
            },
        ]
        self.artifacts.extend(shard_artifacts)
        marker = {
            "schema": "jev-nethack-transition-shard/v1",
            "completed": True,
            "records": self.record_number,
            "transitionCount": self.record_number,
            "startedAt": self._shard_first_at or self._shard_opened_at,
            "endedAt": self._shard_last_at or datetime.now(timezone.utc).isoformat(),
            "artifacts": shard_artifacts,
        }
        marker_path = self.root / f"transitions-{self.shard_number:06d}.complete.json"
        _write_json_atomic(marker_path, marker)
        self._pack = None
        self._index = None

    def write(
        self,
        *,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self._pack is None or self._index is None or self.pack_path is None:
            raise TransitionPackError("transition writer is closed")
        payload = encode_transition(
            observation=observation,
            next_observation=next_observation,
            metadata=metadata,
        )
        if len(payload) > MAX_FRAME_BYTES:
            raise TransitionPackError("transition payload exceeds the per-frame size cap")
        digest = hashlib.sha256(payload).digest()
        frame_bytes = FRAME_HEADER.size + len(payload)

        def build_index_record() -> tuple[int, dict[str, Any], bytes]:
            assert self._pack is not None and self.index_path is not None
            offset_value = self._pack.tell()
            record = {
                "schema": SCHEMA,
                "shard": self.shard_number,
                "record": self.record_number + 1,
                "offset": offset_value,
                "payloadBytes": len(payload),
                "sha256": digest.hex(),
                "eventId": metadata["eventId"],
                "episodeId": metadata.get("episodeId"),
                "seed": metadata.get("seed"),
                "step": metadata.get("step"),
                "actionIndex": metadata.get("actionIndex"),
                "keycode": metadata.get("keycode"),
                "observationDigest": metadata.get("observationDigest"),
                "nextObservationDigest": metadata.get("nextObservationDigest"),
                "reward": metadata.get("reward"),
                "terminated": metadata.get("terminated"),
                "truncated": metadata.get("truncated"),
                "isAscended": metadata.get("isAscended"),
                "verifiedAscension": metadata.get("verifiedAscension"),
                "endStatus": metadata.get("endStatus"),
            }
            encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            return offset_value, record, encoded

        offset, index_record, encoded_index = build_index_record()
        projected_pack_bytes = offset + frame_bytes
        projected_index_bytes = self.index_path.stat().st_size + len(encoded_index)
        if self.record_number and (
            projected_pack_bytes > self.max_bytes_per_shard
            or projected_index_bytes > PUBLIC_ARTIFACT_MAX_BYTES
        ):
            self._finish_shard()
            self._open_shard()
            assert self._pack is not None and self._index is not None and self.index_path is not None
            offset, index_record, encoded_index = build_index_record()
            projected_pack_bytes = offset + frame_bytes
            projected_index_bytes = len(encoded_index)
        if projected_pack_bytes > self.max_bytes_per_shard:
            raise TransitionPackError("one transition frame exceeds the configured shard limit")
        if projected_index_bytes > PUBLIC_ARTIFACT_MAX_BYTES:
            raise TransitionPackError("one transition index record exceeds the public artifact limit")

        self._pack.write(FRAME_HEADER.pack(len(payload), digest))
        self._pack.write(payload)
        self._pack.flush()
        os.fsync(self._pack.fileno())
        self.record_number += 1
        self.total_records += 1
        recorded_at = metadata.get("recordedAt")
        timestamp = recorded_at if isinstance(recorded_at, str) else datetime.now(timezone.utc).isoformat()
        if self._shard_first_at is None:
            self._shard_first_at = timestamp
        self._shard_last_at = timestamp
        self._index.write(encoded_index.decode("utf-8"))
        self._index.flush()
        os.fsync(self._index.fileno())
        receipt = dict(index_record)
        receipt["pack"] = self.pack_path.name
        if (
            self.record_number >= self.max_records_per_shard
            or self._pack.tell() == self.max_bytes_per_shard
        ):
            self._finish_shard()
            self._open_shard()
        return receipt

    def close(self, *, completed: bool = True, reason: str = "clean_close") -> dict[str, Any]:
        if self._pack is not None and self.record_number == 0:
            # Rotation opens the next shard eagerly. Remove only this writer's
            # empty, rebuildable tail after closing its exact files.
            assert self.pack_path is not None and self.index_path is not None and self._index is not None
            self._pack.close()
            self._index.close()
            self.pack_path.unlink()
            self.index_path.unlink()
            self._pack = None
            self._index = None
        else:
            self._finish_shard()
        manifest = {
            "schema": SCHEMA,
            "completed": bool(completed),
            "reason": reason,
            "records": self.total_records,
            "artifacts": self.artifacts,
        }
        manifest_path = self.root / "training-manifest.json"
        _write_json_atomic(manifest_path, manifest)
        return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
