"""Terminal score and death metrics with bound native xlog provenance."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

MAX_XLOG_BYTES = 64 * 1024
_EPISODE_DIR_RE = re.compile(r"^episode-(?P<episode>[0-9]{8})-seed-(?P<seed>[0-9]+)-[A-Za-z0-9_-]+$")
_XLOG_NAME_RE = re.compile(r"^nle\.(?P<pid>[1-9][0-9]*)\.xlogfile$")
_TTYREC_NAME_RE = re.compile(r"^nle\.(?P<pid>[1-9][0-9]*)\.[0-9]+\.ttyrec3\.bz2$")
_INT_FIELDS = {"points", "deathdnum", "deathlev", "maxlvl", "hp", "maxhp", "deaths", "turns"}


class MetricEvidenceError(ValueError):
    """A stable native metric could not be bound to this episode."""


def _has_symlink_component(path: Path, *, trusted_root: Path | None) -> bool:
    if trusted_root is None:
        return path.is_symlink()
    try:
        relative = path.relative_to(trusted_root)
    except ValueError:
        return True
    if trusted_root.is_symlink():
        return True
    current = trusted_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_regular_nofollow(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MetricEvidenceError("evidence path cannot be opened safely") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise MetricEvidenceError("evidence path is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise MetricEvidenceError("evidence file exceeds the size limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise MetricEvidenceError("evidence file changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise MetricEvidenceError("evidence file grew while reading")
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise MetricEvidenceError("evidence file changed while reading")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def _stat_regular_nofollow(path: Path) -> os.stat_result:
    """Inspect a retained binary artifact without loading it into memory."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MetricEvidenceError("evidence path cannot be opened safely") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise MetricEvidenceError("evidence path is not a regular file")
        return file_stat
    finally:
        os.close(fd)


def _parse_xlog(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="strict").strip()
    if not text or "\n" in text or "\r" in text:
        raise MetricEvidenceError("xlogfile must contain exactly one record")
    fields: dict[str, Any] = {}
    for token in text.split("\t"):
        if "=" not in token:
            raise MetricEvidenceError("xlogfile field is missing '='")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise MetricEvidenceError("xlogfile has a duplicate or empty field")
        if len(key) > 64 or len(value) > 4096:
            raise MetricEvidenceError("xlogfile field exceeds the size limit")
        if key in _INT_FIELDS:
            if not re.fullmatch(r"-?[0-9]+", value):
                raise MetricEvidenceError(f"xlogfile integer field {key!r} is malformed")
            fields[key] = int(value)
        else:
            fields[key] = value
    if not isinstance(fields.get("points"), int) or fields["points"] < 0:
        raise MetricEvidenceError("xlogfile points is missing or negative")
    if not isinstance(fields.get("ttyrecname"), str):
        raise MetricEvidenceError("xlogfile ttyrecname is missing")
    return fields


def _unknown(error: str, *, candidate_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "officialFinalScore": None,
        "deathCause": None,
        "deathWhile": None,
        "officialScoreEvidence": None,
        "officialScoreError": error,
    }
    if candidate_count is not None:
        result["xlogfileCandidateCount"] = candidate_count
    return result


def terminal_metrics(
    episode_dir: Path,
    *,
    expected_episode_id: int | None,
    expected_seed: int | None,
    trusted_root: Path | None = None,
) -> dict[str, Any]:
    """Return an authoritative native terminal metric or an explicit unknown.

    An xlog record is authoritative only when it is bound to the expected
    episode directory and to a retained ttyrec from the same NLE process.
    """

    if _has_symlink_component(episode_dir, trusted_root=trusted_root) or not episode_dir.is_dir():
        return _unknown("unsafe_episode_directory")
    identity = _EPISODE_DIR_RE.fullmatch(episode_dir.name)
    if identity is None or expected_episode_id is None or expected_seed is None:
        return _unknown("unbound_episode_identity")
    if int(identity.group("episode")) != expected_episode_id:
        return _unknown("episode_id_mismatch")
    if int(identity.group("seed")) != expected_seed:
        return _unknown("episode_seed_mismatch")

    ttyrec_dir = episode_dir / "nle-ttyrec"
    if ttyrec_dir.is_symlink() or not ttyrec_dir.is_dir():
        return _unknown("unsafe_ttyrec_directory")
    try:
        candidates = sorted(
            (entry for entry in ttyrec_dir.iterdir() if entry.name.endswith(".xlogfile")),
            key=lambda entry: entry.name,
        )
    except OSError:
        candidates = []
    if len(candidates) != 1:
        return _unknown("missing_xlogfile" if not candidates else "ambiguous_xlogfile", candidate_count=len(candidates))

    xlog_path = candidates[0]
    if xlog_path.is_symlink() or _XLOG_NAME_RE.fullmatch(xlog_path.name) is None:
        return _unknown("unsafe_xlogfile", candidate_count=1)
    try:
        raw, xlog_stat = _read_regular_nofollow(xlog_path, max_bytes=MAX_XLOG_BYTES)
        fields = _parse_xlog(raw)
    except (OSError, UnicodeError, MetricEvidenceError):
        return _unknown("malformed_xlogfile", candidate_count=1)

    xlog_match = _XLOG_NAME_RE.fullmatch(xlog_path.name)
    assert xlog_match is not None
    ttyrec_name = fields["ttyrecname"]
    ttyrec_match = _TTYREC_NAME_RE.fullmatch(ttyrec_name)
    if ttyrec_match is None or ttyrec_match.group("pid") != xlog_match.group("pid"):
        return _unknown("xlog_ttyrec_binding_mismatch", candidate_count=1)
    ttyrec_path = ttyrec_dir / ttyrec_name
    if ttyrec_path.parent != ttyrec_dir or ttyrec_path.is_symlink():
        return _unknown("unsafe_ttyrec", candidate_count=1)
    try:
        ttyrec_stat = _stat_regular_nofollow(ttyrec_path)
    except (OSError, MetricEvidenceError):
        return _unknown("missing_bound_ttyrec", candidate_count=1)

    return {
        "officialFinalScore": fields["points"],
        "deathCause": fields.get("death"),
        "deathWhile": fields.get("while"),
        "xlogfileCandidateCount": 1,
        "officialScoreEvidence": {
            "source": "native_xlogfile",
            "relativePath": str(xlog_path.relative_to(episode_dir)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "xlogBytes": int(xlog_stat.st_size),
            "record": "single_native_xlog_record",
            "ttyrecRelativePath": str(ttyrec_path.relative_to(episode_dir)),
            "ttyrecBytes": int(ttyrec_stat.st_size),
            "episodeId": expected_episode_id,
            "seed": expected_seed,
        },
    }


def choose_compat_score(*, official_final_score: int | None, last_observed_score: int | None) -> tuple[int | None, str]:
    """Keep legacy ``score`` useful while recording its exact semantics."""
    if official_final_score is not None:
        return official_final_score, "official_final"
    return last_observed_score, "last_observed"


def episode_metric_fields(
    *,
    episode_dir: Path,
    expected_episode_id: int,
    expected_seed: int,
    trusted_root: Path | None = None,
    last_observed_stats: dict[str, Any],
    max_observed_score: int | None,
    total_reward: float,
) -> dict[str, Any]:
    """Compose additive metrics without merging independent measurements."""
    terminal = terminal_metrics(
        episode_dir,
        expected_episode_id=expected_episode_id,
        expected_seed=expected_seed,
        trusted_root=trusted_root,
    )
    last_score = last_observed_stats.get("score")
    score, semantics = choose_compat_score(
        official_final_score=terminal.get("officialFinalScore"),
        last_observed_score=last_score,
    )
    return {
        "score": score,
        "scoreSemantics": semantics,
        "officialFinalScore": terminal.get("officialFinalScore"),
        "lastObservedScore": last_score,
        "maxObservedScore": max_observed_score,
        "totalReward": float(total_reward),
        "deathCause": terminal.get("deathCause"),
        "deathWhile": terminal.get("deathWhile"),
        "officialScoreEvidence": terminal.get("officialScoreEvidence"),
        "officialScoreError": terminal.get("officialScoreError"),
        "xlogfileCandidateCount": terminal.get("xlogfileCandidateCount"),
    }
