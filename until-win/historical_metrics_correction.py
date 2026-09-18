"""Fail-closed, additive correction of one retained terminal episode result."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping

from metrics import terminal_metrics


CORRECTION_SCHEMA = "jev-nethack-historical-metrics-correction/v1"
RUNTIME_SCHEMA = "jev-nethack-until-win/v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
TERMINAL_STATUSES = {"game_end", "verified_ascension"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid unambiguous JSON") from exc


def _assert_no_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence path escapes the data root") from exc
    current = root
    if current.is_symlink():
        raise ValueError("data root may not be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("evidence path contains a symlink")


@contextmanager
def _regular_reader(path: Path, *, max_bytes: int | None = None) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open {path.name}") from exc
    stream = os.fdopen(fd, "rb")
    try:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path.name} is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"{path.name} exceeds its size limit")
        yield stream, before
        after = os.fstat(stream.fileno())
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ValueError(f"{path.name} changed while it was read")
    finally:
        stream.close()


def _stable_bytes(path: Path, *, root: Path, max_bytes: int = MAX_JSON_BYTES) -> tuple[bytes, os.stat_result]:
    _assert_no_symlink_components(path, root)
    with _regular_reader(path, max_bytes=max_bytes) as (stream, before):
        raw = stream.read(max_bytes + 1)
        if len(raw) != before.st_size:
            raise ValueError(f"{path.name} changed while it was read")
        return raw, before


def _artifact(path: Path, *, root: Path, raw: bytes, file_stat: os.stat_result) -> dict[str, Any]:
    return {
        "relativePath": path.relative_to(root).as_posix(),
        "bytes": int(file_stat.st_size),
        "sha256": _sha256_bytes(raw),
    }


def _read_json_artifact(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, file_stat = _stable_bytes(path, root=root)
    value = _parse_json(raw, label=path.name)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value, _artifact(path, root=root, raw=raw, file_stat=file_stat)


def _read_state(state_path: Path) -> tuple[dict[str, Any], bytes]:
    raw, _ = _stable_bytes(state_path, root=state_path.parent)
    state = _parse_json(raw, label="state.json")
    if not isinstance(state, dict):
        raise ValueError("state.json must contain an object")
    if state.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("state.json schema does not match the runtime contract")
    return state, raw


def _row_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "episodeId",
            "seed",
            "status",
            "steps",
            "terminated",
            "truncated",
            "isAscended",
            "maxDepth",
        )
    }


def _select_terminal_row(state: Mapping[str, Any], episode_id: int) -> tuple[int, dict[str, Any]]:
    rows = state.get("episodeResults")
    if not isinstance(rows, list):
        raise ValueError("state has no retained episode results")
    eligible: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("episodeId") != episode_id:
            continue
        if row.get("status") in TERMINAL_STATUSES and row.get("terminated") is True:
            eligible.append((index, row))
    if len(eligible) != 1:
        raise ValueError("completed terminal episode result is missing or ambiguous")
    index, row = eligible[0]
    if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
        raise ValueError("terminal episode result has an invalid seed")
    if not isinstance(row.get("steps"), int) or isinstance(row.get("steps"), bool) or row["steps"] <= 0:
        raise ValueError("terminal episode result has an invalid step count")
    if row.get("truncated") is not False:
        raise ValueError("historical native terminal correction does not accept a truncated result")
    return index, row


def _matching_attempt_dirs(data_root: Path, *, episode_id: int, seed: int) -> list[Path]:
    fragments_root = data_root / "fragments"
    pattern = f"episode-{episode_id:08d}-seed-{seed}-*"
    attempts: list[Path] = []
    if not fragments_root.is_dir() or fragments_root.is_symlink():
        raise ValueError("fragments directory is missing or unsafe")
    for fragment in sorted(fragments_root.iterdir(), key=lambda item: item.name):
        if not fragment.is_dir() or fragment.is_symlink():
            continue
        episodes = fragment / "episodes"
        if not episodes.is_dir() or episodes.is_symlink():
            continue
        for attempt in sorted(episodes.glob(pattern), key=lambda item: item.name):
            if attempt.is_dir() and not attempt.is_symlink():
                _assert_no_symlink_components(attempt, data_root)
                attempts.append(attempt)
    if not attempts:
        raise ValueError("episode attempt directories are missing")
    return attempts


def _summary_matches_row(summary: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if summary.get("schema") != RUNTIME_SCHEMA:
        return False
    for key, expected in _row_binding(row).items():
        if summary.get(key) != expected:
            return False
    return isinstance(summary.get("endStatus"), str) and bool(summary["endStatus"])


def _select_terminal_summary(
    data_root: Path,
    *,
    attempts: list[Path],
    row: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for attempt in attempts:
        summary_path = attempt / "summary.json"
        if not summary_path.exists():
            continue
        summary, artifact = _read_json_artifact(summary_path, root=data_root)
        if _summary_matches_row(summary, row):
            matches.append((attempt, summary, artifact))
    if len(matches) != 1:
        raise ValueError("completed terminal summary is missing or ambiguous")
    return matches[0]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fragment_sources(
    data_root: Path,
    *,
    attempts: list[Path],
) -> list[tuple[Path, Path, dict[str, Any]]]:
    by_fragment: dict[Path, tuple[Path, Path, dict[str, Any]]] = {}
    for attempt in attempts:
        fragment = attempt.parent.parent
        config_path = fragment / "config.json"
        journal_path = fragment / "transitions.jsonl"
        config, artifact = _read_json_artifact(config_path, root=data_root)
        if config.get("fragmentId") != fragment.name:
            raise ValueError("fragment config identity does not match its directory")
        pid = config.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("fragment config has an invalid producer PID")
        if _pid_is_alive(pid):
            raise ValueError("episode evidence belongs to an active producer")
        _assert_no_symlink_components(journal_path, data_root)
        by_fragment[fragment] = (config_path, journal_path, artifact)
    return [by_fragment[key] for key in sorted(by_fragment, key=lambda item: item.name)]


def _player_score(state: Any, *, label: str) -> int:
    if not isinstance(state, Mapping) or not isinstance(state.get("player"), Mapping):
        raise ValueError(f"{label} has no player state")
    value = state["player"].get("score")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} has an invalid score")
    return value


def _transition_fingerprint(event: Mapping[str, Any]) -> dict[str, Any]:
    reward = event.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(float(reward)):
        raise ValueError("transition has an invalid reward")
    return {
        "eventId": event.get("eventId"),
        "episodeId": event.get("episodeId"),
        "seed": event.get("seed"),
        "step": event.get("step"),
        "actionIndex": event.get("actionIndex"),
        "keycode": event.get("keycode"),
        "observationDigest": event.get("observationDigest"),
        "nextObservationDigest": event.get("nextObservationDigest"),
        "reward": float(reward),
        "terminated": event.get("terminated"),
        "truncated": event.get("truncated"),
        "isAscended": event.get("isAscended"),
        "verifiedAscension": event.get("verifiedAscension"),
        "endStatus": event.get("endStatus"),
        "stateScore": _player_score(event.get("state"), label="transition state"),
        "nextStateScore": _player_score(event.get("nextState"), label="transition next state"),
    }


def _scan_journals(
    data_root: Path,
    *,
    sources: list[tuple[Path, Path, dict[str, Any]]],
    episode_id: int,
    seed: int,
    summary: Mapping[str, Any],
    summary_attempt: Path,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    transitions: dict[int, dict[str, Any]] = {}
    summary_events: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    terminal_fragment = summary_attempt.parent.parent
    terminal_journal = terminal_fragment / "transitions.jsonl"
    for config_path, journal_path, config_artifact in sources:
        artifacts.append(config_artifact)
        digest = hashlib.sha256()
        with _regular_reader(journal_path) as (stream, before):
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                if len(raw_line) > MAX_EVENT_LINE_BYTES:
                    raise ValueError("transition journal contains an oversized line")
                if not raw_line.endswith(b"\n"):
                    raise ValueError("transition journal has an incomplete trailing line")
                if not raw_line.strip():
                    continue
                event = _parse_json(raw_line, label=f"{journal_path.name}:{line_number}")
                if not isinstance(event, dict):
                    raise ValueError("transition journal entry must be an object")
                if event.get("episodeId") != episode_id or event.get("seed") != seed:
                    continue
                if event.get("schema") != RUNTIME_SCHEMA:
                    raise ValueError("episode evidence schema does not match the runtime contract")
                event_type = event.get("eventType")
                if event_type == "transition_committed":
                    step = event.get("step")
                    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                        raise ValueError("committed transition has an invalid step")
                    expected_id = f"episode-{episode_id:08d}-step-{step:09d}"
                    event_id = event.get("eventId")
                    legacy_id = re.fullmatch(
                        rf"legacy-[0-9a-f]{{32}}-episode-{episode_id}-step-{step:09d}",
                        event_id if isinstance(event_id, str) else "",
                    )
                    provenance = event.get("provenance")
                    regenerated_legacy = (
                        isinstance(provenance, Mapping)
                        and provenance.get("origin") == "deterministically_reconstructed_legacy"
                        and provenance.get("rawArrays") == "regenerated_by_verified_seed_action_replay"
                    )
                    if event_id != expected_id and not (legacy_id is not None and regenerated_legacy):
                        raise ValueError("committed transition event ID does not match its step provenance")
                    fingerprint = _transition_fingerprint(event)
                    evidence = {
                        "fingerprint": fingerprint,
                        "relativePath": journal_path.relative_to(data_root).as_posix(),
                        "line": line_number,
                        "lineSha256": _sha256_bytes(raw_line),
                    }
                    existing = transitions.get(step)
                    if existing is not None and existing["fingerprint"] != fingerprint:
                        raise ValueError("duplicate committed transition step conflicts with retained evidence")
                    if existing is None:
                        transitions[step] = evidence
                elif event_type == "episode_summary":
                    event_summary = dict(event)
                    event_summary.pop("eventType", None)
                    if event_summary == dict(summary):
                        summary_events.append(
                            {
                                "relativePath": journal_path.relative_to(data_root).as_posix(),
                                "line": line_number,
                                "lineSha256": _sha256_bytes(raw_line),
                            }
                        )
            journal_artifact = {
                "relativePath": journal_path.relative_to(data_root).as_posix(),
                "bytes": int(before.st_size),
                "sha256": digest.hexdigest(),
            }
            artifacts.append(journal_artifact)
    if len(summary_events) != 1:
        raise ValueError("terminal summary event is missing or ambiguous")
    if summary_events[0]["relativePath"] != terminal_journal.relative_to(data_root).as_posix():
        raise ValueError("terminal summary event is not in the summary fragment")
    return transitions, summary_events, artifacts


def _derive_event_metrics(
    *,
    transitions: Mapping[int, Mapping[str, Any]],
    summary_events: list[dict[str, Any]],
    summary: Mapping[str, Any],
    episode_id: int,
) -> tuple[int, int, float, dict[str, Any]]:
    steps = int(summary["steps"])
    if sorted(transitions) != list(range(steps)):
        raise ValueError("committed terminal episode transitions are not contiguous")
    terminal = [
        evidence
        for evidence in transitions.values()
        if evidence["fingerprint"]["terminated"] is True or evidence["fingerprint"]["truncated"] is True
    ]
    if len(terminal) != 1:
        raise ValueError("terminal transition is missing or ambiguous")
    terminal_evidence = terminal[0]
    fingerprint = terminal_evidence["fingerprint"]
    if fingerprint["step"] != steps - 1:
        raise ValueError("terminal transition is not the final committed step")
    for key in ("terminated", "truncated", "isAscended", "endStatus"):
        if fingerprint[key] != summary.get(key):
            raise ValueError(f"terminal transition {key} does not match the summary")
    summary_event = summary_events[0]
    if (
        summary_event["relativePath"] != terminal_evidence["relativePath"]
        or summary_event["line"] != terminal_evidence["line"] + 1
    ):
        raise ValueError("terminal summary event does not immediately follow the terminal transition")
    ordered = [transitions[index]["fingerprint"] for index in range(steps)]
    for current, following in zip(ordered, ordered[1:]):
        if current["nextObservationDigest"] != following["observationDigest"]:
            raise ValueError("committed transition observation chain is discontinuous")
        if current["nextStateScore"] != following["stateScore"]:
            raise ValueError("committed transition score chain is discontinuous")
    last_observed = ordered[-1]["stateScore"]
    observed_scores = [item["stateScore"] for item in ordered]
    observed_scores.extend(
        item["nextStateScore"]
        for item in ordered
        if item["terminated"] is not True and item["truncated"] is not True
    )
    maximum_observed = max(observed_scores)
    total_reward = math.fsum(item["reward"] for item in ordered)
    summary_reward = summary.get("totalReward")
    if (
        not isinstance(summary_reward, (int, float))
        or isinstance(summary_reward, bool)
        or not math.isfinite(float(summary_reward))
        or not math.isclose(total_reward, float(summary_reward), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("deduplicated transition reward does not match the terminal summary")
    terminal_public = {
        key: terminal_evidence[key]
        for key in ("relativePath", "line", "lineSha256")
    }
    terminal_public["eventId"] = f"episode-{episode_id:08d}-step-{steps - 1:09d}"
    return last_observed, maximum_observed, total_reward, terminal_public


def _source_artifact(path: Path, *, data_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    _assert_no_symlink_components(path, data_root)
    with _regular_reader(path) as (stream, before):
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "relativePath": path.relative_to(data_root).as_posix(),
        "bytes": int(before.st_size),
        "sha256": digest.hexdigest(),
    }


def _deduplicate_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        relative = artifact["relativePath"]
        existing = by_path.get(relative)
        if existing is not None and existing != artifact:
            raise ValueError("source artifact identity changed during correction build")
        by_path[relative] = artifact
    return [by_path[key] for key in sorted(by_path)]


def _build_from_state(*, data_root: Path, state: Mapping[str, Any], episode_id: int) -> dict[str, Any]:
    target_index, row = _select_terminal_row(state, episode_id)
    seed = int(row["seed"])
    attempts = _matching_attempt_dirs(data_root, episode_id=episode_id, seed=seed)
    summary_attempt, summary, summary_artifact = _select_terminal_summary(
        data_root,
        attempts=attempts,
        row=row,
    )
    sources = _fragment_sources(data_root, attempts=attempts)
    transitions, summary_events, artifacts = _scan_journals(
        data_root,
        sources=sources,
        episode_id=episode_id,
        seed=seed,
        summary=summary,
        summary_attempt=summary_attempt,
    )
    last_observed, maximum_observed, total_reward, terminal_evidence = _derive_event_metrics(
        transitions=transitions,
        summary_events=summary_events,
        summary=summary,
        episode_id=episode_id,
    )
    native = terminal_metrics(
        summary_attempt,
        expected_episode_id=episode_id,
        expected_seed=seed,
        trusted_root=data_root,
    )
    official_score = native.get("officialFinalScore")
    official_evidence = native.get("officialScoreEvidence")
    if not isinstance(official_score, int) or isinstance(official_score, bool) or not isinstance(official_evidence, dict):
        raise ValueError(f"native terminal score is not bound: {native.get('officialScoreError')}")
    xlog_path = summary_attempt / official_evidence["relativePath"]
    ttyrec_path = summary_attempt / official_evidence["ttyrecRelativePath"]
    xlog_artifact = _source_artifact(xlog_path, data_root=data_root)
    ttyrec_artifact = _source_artifact(ttyrec_path, data_root=data_root)
    if (
        xlog_artifact["sha256"] != official_evidence.get("sha256")
        or xlog_artifact["bytes"] != official_evidence.get("xlogBytes")
        or ttyrec_artifact["bytes"] != official_evidence.get("ttyrecBytes")
    ):
        raise ValueError("native score evidence changed while the correction was built")
    artifacts.extend(
        [
            summary_artifact,
            xlog_artifact,
            ttyrec_artifact,
        ]
    )
    source_artifacts = _deduplicate_artifacts(artifacts)
    return {
        "schema": CORRECTION_SCHEMA,
        "episodeId": episode_id,
        "seed": seed,
        "target": {
            "episodeResultIndex": target_index,
            "rowBinding": _row_binding(row),
            "summaryRelativePath": (summary_attempt / "summary.json").relative_to(data_root).as_posix(),
            "summarySha256": summary_artifact["sha256"],
        },
        "officialFinalScore": official_score,
        "lastObservedScore": last_observed,
        "maxObservedScore": maximum_observed,
        "totalReward": total_reward,
        "endStatus": summary["endStatus"],
        "deathCause": native.get("deathCause"),
        "deathWhile": native.get("deathWhile"),
        "officialScoreEvidence": official_evidence,
        "evidence": {
            "terminalTransition": terminal_evidence,
            "episodeSummaryEvent": summary_events[0],
            "transitionCount": len(transitions),
            "sourceArtifacts": source_artifacts,
            "referencedSourceArtifactsReadOnly": True,
        },
    }


def build_correction(*, data_root: Path, state_path: Path, episode_id: int) -> dict[str, Any]:
    """Build a read-only correction from immutable closed-episode evidence."""
    state, raw = _read_state(state_path)
    correction = _build_from_state(data_root=data_root, state=state, episode_id=episode_id)
    target_index, target_row = _select_terminal_row(state, episode_id)
    expected_post_state, _, expected_post_row = _updated_state(state, correction)
    expected_post_raw = _json_bytes(expected_post_state)
    correction["reviewState"] = {
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "targetEpisodeResultIndex": target_index,
        "targetRowSha256": _sha256_bytes(_canonical_bytes(target_row)),
        "expectedPostBytes": len(expected_post_raw),
        "expectedPostSha256": _sha256_bytes(expected_post_raw),
        "expectedPostTargetRowSha256": _sha256_bytes(_canonical_bytes(expected_post_row)),
    }
    return correction


def _with_state_evidence(
    correction: dict[str, Any],
    *,
    state: Mapping[str, Any],
    state_raw: bytes,
    episode_id: int,
) -> dict[str, Any]:
    target_index, target_row = _select_terminal_row(state, episode_id)
    expected_post_state, _, expected_post_row = _updated_state(state, correction)
    expected_post_raw = _json_bytes(expected_post_state)
    return {
        **correction,
        "reviewState": {
            "bytes": len(state_raw),
            "sha256": _sha256_bytes(state_raw),
            "targetEpisodeResultIndex": target_index,
            "targetRowSha256": _sha256_bytes(_canonical_bytes(target_row)),
            "expectedPostBytes": len(expected_post_raw),
            "expectedPostSha256": _sha256_bytes(expected_post_raw),
            "expectedPostTargetRowSha256": _sha256_bytes(_canonical_bytes(expected_post_row)),
        },
    }


def _read_expected_dry_run(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("expected dry-run SHA-256 is invalid")
    raw, file_stat = _stable_bytes(path, root=path.parent)
    if _sha256_bytes(raw) != expected_sha256:
        raise RuntimeError("expected dry-run file hash does not match")
    value = _parse_json(raw, label=path.name)
    if not isinstance(value, dict) or value.get("phase") != "dry_run" or value.get("applied") is not False:
        raise RuntimeError("expected evidence is not a dry-run result")
    correction = dict(value)
    correction.pop("phase", None)
    correction.pop("applied", None)
    if correction.get("schema") != CORRECTION_SCHEMA:
        raise RuntimeError("expected dry-run schema does not match")
    artifact = {
        "filename": path.name,
        "bytes": int(file_stat.st_size),
        "sha256": expected_sha256,
    }
    return correction, artifact


def _validated_global_lock(data_root: Path, supplied_lock: Path) -> tuple[Path, dict[str, Any]]:
    handoff_path = data_root / "handoff-receipt.json"
    handoff, artifact = _read_json_artifact(handoff_path, root=data_root)
    recorded = handoff.get("globalLock")
    if (
        handoff.get("schema") != RUNTIME_SCHEMA
        or handoff.get("accepted") is not True
        or handoff.get("conflicts") != []
        or not isinstance(recorded, str)
    ):
        raise RuntimeError("runtime handoff receipt does not bind an accepted global lock")
    supplied = supplied_lock.expanduser().resolve()
    if Path(recorded).expanduser().resolve() != supplied:
        raise RuntimeError("supplied global lock does not match the runtime handoff receipt")
    return supplied, artifact


def _find_bound_target(state: Mapping[str, Any], correction: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    rows = state.get("episodeResults")
    target = correction.get("target")
    if not isinstance(rows, list) or not isinstance(target, Mapping):
        raise ValueError("correction target is invalid")
    binding = target.get("rowBinding")
    if not isinstance(binding, Mapping):
        raise ValueError("correction row binding is invalid")
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if isinstance(row, dict) and _row_binding(row) == dict(binding)
    ]
    if len(matches) != 1 or matches[0][0] != target.get("episodeResultIndex"):
        raise ValueError("bound terminal result row changed or became ambiguous")
    return matches[0]


def _updated_state(state: Mapping[str, Any], correction: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    index, original_row = _find_bound_target(state, correction)
    updated = dict(state)
    rows = list(state["episodeResults"])
    row = dict(original_row)
    for key in (
        "officialFinalScore",
        "lastObservedScore",
        "maxObservedScore",
        "totalReward",
        "endStatus",
        "deathCause",
        "deathWhile",
        "officialScoreEvidence",
    ):
        row[key] = correction[key]
    row.pop("officialScoreError", None)
    row["score"] = correction["officialFinalScore"]
    row["scoreSemantics"] = "official_final"
    rows[index] = row
    updated["episodeResults"] = rows
    updated["bestScore"] = max(int(state.get("bestScore", 0)), int(correction["officialFinalScore"]))
    return updated, original_row, row


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("refusing to replace a symlink")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


class _KernelLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> "_KernelLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError("cannot safely open the gameplay lock") from exc
        self.stream = os.fdopen(fd, "r+b")
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RuntimeError("gameplay lock is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as exc:
            self.stream.close()
            self.stream = None
            if isinstance(exc, OSError) and exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError("gameplay lock is held by an active process") from None
            raise
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.stream is not None:
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def _require_apply_gate(state: Mapping[str, Any], stop_path: Path) -> None:
    if stop_path.parent.is_symlink() or stop_path.is_symlink() or not stop_path.is_file():
        raise RuntimeError("correction requires a regular STOP latch")
    if state.get("status") != "paused":
        raise RuntimeError("correction requires paused runtime state")
    if state.get("activeEpisode") is not None or state.get("activeFragment") is not None:
        raise RuntimeError("correction refuses active episode or fragment state")


def _verify_source_artifacts(data_root: Path, expected: list[dict[str, Any]]) -> None:
    for artifact in expected:
        relative = artifact.get("relativePath")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("correction source artifact path is invalid")
        current = _source_artifact(data_root / relative, data_root=data_root)
        if current != artifact:
            raise RuntimeError(f"correction source changed: {relative}")


def _verify_correction_membership(
    *,
    data_root: Path,
    state: Mapping[str, Any],
    episode_id: int,
    expected_source_correction: Mapping[str, Any],
) -> None:
    rebuilt = _build_from_state(data_root=data_root, state=state, episode_id=episode_id)
    if rebuilt != dict(expected_source_correction):
        raise RuntimeError("correction evidence membership changed after review")


def _read_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw, _ = _stable_bytes(path, root=path.parent)
    value = _parse_json(raw, label=path.name)
    if not isinstance(value, dict):
        raise RuntimeError("existing correction receipt is invalid")
    return value


def _validate_handoff_artifact(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"relativePath", "bytes", "sha256"}:
        raise RuntimeError("existing correction receipt has an invalid handoff artifact")
    if value.get("relativePath") != "handoff-receipt.json":
        raise RuntimeError("existing correction receipt has an invalid handoff artifact path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError("existing correction receipt has invalid handoff artifact metadata")


def _validate_existing_receipt(
    existing: Mapping[str, Any],
    *,
    correction: Mapping[str, Any],
    correction_digest: str,
    expected_evidence: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
) -> None:
    phase = existing.get("phase")
    if phase not in {"prepared", "completed"}:
        raise RuntimeError("existing correction receipt has an invalid phase")
    expected_keys = set(correction) | {
        "phase",
        "applied",
        "preparedAt",
        "correctionDigest",
        "expectedDryRunEvidence",
        "lockBinding",
        "preparedHandoffReceipt",
        "stateSha256Before",
        "stateSha256After",
        "targetRowSha256Before",
        "targetRowSha256After",
    }
    if phase == "completed":
        expected_keys.update(
            {"completedAt", "completedHandoffReceipt", "sourceArtifactsVerifiedAfterStateWrite"}
        )
    if set(existing) != expected_keys:
        raise RuntimeError("existing correction receipt has unexpected or missing fields")
    for key, expected in correction.items():
        if existing.get(key) != expected:
            raise RuntimeError(f"existing correction receipt conflicts at {key}")
    if (
        existing.get("correctionDigest") != correction_digest
        or existing.get("expectedDryRunEvidence") != dict(expected_evidence)
        or existing.get("lockBinding") != dict(lock_binding)
        or existing.get("stateSha256Before") != correction["reviewState"]["sha256"]
        or existing.get("targetRowSha256Before") != correction["reviewState"]["targetRowSha256"]
        or existing.get("stateSha256After") != correction["reviewState"]["expectedPostSha256"]
        or existing.get("targetRowSha256After")
        != correction["reviewState"]["expectedPostTargetRowSha256"]
    ):
        raise RuntimeError("existing correction receipt does not match reviewed evidence")
    _validate_handoff_artifact(existing.get("preparedHandoffReceipt"))
    for key in ("stateSha256After", "targetRowSha256After"):
        value = existing.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"existing correction receipt has an invalid {key}")
    if not isinstance(existing.get("preparedAt"), str) or not existing["preparedAt"]:
        raise RuntimeError("existing correction receipt has no preparation timestamp")
    if phase == "prepared":
        if existing.get("applied") is not False:
            raise RuntimeError("prepared correction receipt has invalid applied state")
    else:
        _validate_handoff_artifact(existing.get("completedHandoffReceipt"))
        if (
            existing.get("applied") is not True
            or existing.get("sourceArtifactsVerifiedAfterStateWrite") is not True
            or not isinstance(existing.get("completedAt"), str)
            or not existing["completedAt"]
        ):
            raise RuntimeError("completed correction receipt is invalid")

def _state_contains_correction(
    state: Mapping[str, Any],
    correction: Mapping[str, Any],
    *,
    expected_row_sha256: str,
) -> bool:
    try:
        _, row = _find_bound_target(state, correction)
    except ValueError:
        return False
    desired = dict(row)
    for key in (
        "officialFinalScore",
        "lastObservedScore",
        "maxObservedScore",
        "totalReward",
        "endStatus",
        "deathCause",
        "deathWhile",
        "officialScoreEvidence",
    ):
        if desired.get(key) != correction.get(key):
            return False
    if desired.get("score") != correction.get("officialFinalScore") or desired.get("scoreSemantics") != "official_final":
        return False
    if _sha256_bytes(_canonical_bytes(desired)) != expected_row_sha256:
        return False
    return int(state.get("bestScore", 0)) >= int(correction["officialFinalScore"])


def apply_correction(
    *,
    data_root: Path,
    state_path: Path,
    episode_id: int,
    receipt_path: Path,
    global_lock: Path,
    expected_dry_run: Path,
    expected_dry_run_sha256: str,
) -> dict[str, Any]:
    """Apply one reviewed correction under STOP and both runner locks.

    The expected dry-run must have been generated from the exact paused state
    being corrected. Its caller-supplied digest binds both the immutable raw
    evidence and the pre-mutation state used for compare-and-swap.
    """
    stop_path = state_path.parent / "control" / "STOP"
    runtime_lock = data_root / "runtime.lock"
    receipt_parent = receipt_path.parent
    if receipt_parent.is_symlink() or not receipt_parent.is_dir():
        raise RuntimeError("correction receipt parent must be a pre-existing regular directory")
    distinct_paths = {
        path.expanduser().resolve()
        for path in (state_path, receipt_path, expected_dry_run, global_lock, runtime_lock)
    }
    if len(distinct_paths) != 5:
        raise RuntimeError("state, receipt, evidence, and lock paths must differ")

    # Read the handoff once to identify the lock, then read it again while both
    # locks are held. A concurrent runner handoff can therefore only make the
    # operation fail before any receipt or state write.
    bound_global_lock, initial_handoff_artifact = _validated_global_lock(data_root, global_lock)
    with _KernelLock(bound_global_lock), _KernelLock(runtime_lock):
        checked_global_lock, handoff_artifact = _validated_global_lock(data_root, global_lock)
        if checked_global_lock != bound_global_lock or handoff_artifact != initial_handoff_artifact:
            raise RuntimeError("runtime handoff changed while correction locks were acquired")

        expected_correction, expected_evidence = _read_expected_dry_run(
            expected_dry_run,
            expected_sha256=expected_dry_run_sha256,
        )
        expected_review = expected_correction.get("reviewState")
        if not isinstance(expected_review, Mapping):
            raise RuntimeError("expected dry-run has no reviewed state binding")
        for key in (
            "sha256",
            "targetRowSha256",
            "expectedPostSha256",
            "expectedPostTargetRowSha256",
        ):
            value = expected_review.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise RuntimeError("expected dry-run has an invalid reviewed state binding")
        for key in ("bytes", "targetEpisodeResultIndex", "expectedPostBytes"):
            value = expected_review.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError("expected dry-run has an invalid reviewed state binding")

        state, state_raw = _read_state(state_path)
        _require_apply_gate(state, stop_path)
        fresh_source_correction = _build_from_state(data_root=data_root, state=state, episode_id=episode_id)
        expected_source_correction = dict(expected_correction)
        expected_source_correction.pop("reviewState", None)
        if fresh_source_correction != expected_source_correction:
            raise RuntimeError("current correction evidence does not match the reviewed dry-run")

        correction_digest = _sha256_bytes(_canonical_bytes(expected_correction))
        lock_binding = {
            "globalLock": str(bound_global_lock),
            "runtimeLock": str(runtime_lock.expanduser().resolve()),
        }
        existing = _read_receipt(receipt_path)
        state_before = _sha256_bytes(state_raw)

        # A retry after the atomic state write no longer has the reviewed
        # pre-state hash. It is accepted only through a fully validated
        # prepared/completed receipt and exact post-state/row hashes.
        if state_before != expected_review["sha256"]:
            if existing is None:
                raise RuntimeError("state does not match the reviewed dry-run")
            _validate_existing_receipt(
                existing,
                correction=expected_correction,
                correction_digest=correction_digest,
                expected_evidence=expected_evidence,
                lock_binding=lock_binding,
            )
            if state_before != existing["stateSha256After"] or not _state_contains_correction(
                state,
                expected_correction,
                expected_row_sha256=existing["targetRowSha256After"],
            ):
                raise RuntimeError("state no longer matches the reviewed correction")
            _verify_source_artifacts(data_root, expected_correction["evidence"]["sourceArtifacts"])
            _verify_correction_membership(
                data_root=data_root,
                state=state,
                episode_id=episode_id,
                expected_source_correction=expected_source_correction,
            )
            _, final_handoff_artifact = _validated_global_lock(data_root, global_lock)
            if final_handoff_artifact != handoff_artifact:
                raise RuntimeError("runtime handoff changed during correction verification")
            _read_expected_dry_run(expected_dry_run, expected_sha256=expected_dry_run_sha256)
            if existing["phase"] == "completed":
                return dict(existing)
            completed = {
                **existing,
                "phase": "completed",
                "applied": True,
                "completedAt": utc_now(),
                "completedHandoffReceipt": final_handoff_artifact,
                "sourceArtifactsVerifiedAfterStateWrite": True,
            }
            _atomic_json(receipt_path, completed)
            return completed

        correction = _with_state_evidence(
            fresh_source_correction,
            state=state,
            state_raw=state_raw,
            episode_id=episode_id,
        )
        if correction != expected_correction:
            raise RuntimeError("state does not match the reviewed dry-run")

        updated, original_row, updated_row = _updated_state(state, correction)
        updated_raw = _json_bytes(updated)
        state_after = _sha256_bytes(updated_raw)
        if (
            state_after != correction["reviewState"]["expectedPostSha256"]
            or len(updated_raw) != correction["reviewState"]["expectedPostBytes"]
            or _sha256_bytes(_canonical_bytes(updated_row))
            != correction["reviewState"]["expectedPostTargetRowSha256"]
        ):
            raise RuntimeError("derived post-state does not match the reviewed dry-run")
        prepared = {
            **correction,
            "phase": "prepared",
            "applied": False,
            "preparedAt": utc_now(),
            "correctionDigest": correction_digest,
            "expectedDryRunEvidence": expected_evidence,
            "lockBinding": lock_binding,
            "preparedHandoffReceipt": handoff_artifact,
            "stateSha256Before": state_before,
            "stateSha256After": state_after,
            "targetRowSha256Before": _sha256_bytes(_canonical_bytes(original_row)),
            "targetRowSha256After": _sha256_bytes(_canonical_bytes(updated_row)),
        }
        if existing is not None:
            if existing.get("phase") == "completed":
                raise RuntimeError("completed correction receipt conflicts with the reviewed pre-state")
            _validate_existing_receipt(
                existing,
                correction=correction,
                correction_digest=correction_digest,
                expected_evidence=expected_evidence,
                lock_binding=lock_binding,
            )
            if (
                existing["stateSha256After"] != state_after
                or existing["targetRowSha256After"] != prepared["targetRowSha256After"]
            ):
                raise RuntimeError("existing correction receipt has conflicting post-state hashes")
            prepared = dict(existing)
        else:
            _atomic_json(receipt_path, prepared)

        _verify_source_artifacts(data_root, correction["evidence"]["sourceArtifacts"])
        _, current_handoff_artifact = _validated_global_lock(data_root, global_lock)
        if current_handoff_artifact != handoff_artifact:
            raise RuntimeError("runtime handoff changed after correction preparation")
        current_expected, current_expected_evidence = _read_expected_dry_run(
            expected_dry_run,
            expected_sha256=expected_dry_run_sha256,
        )
        if current_expected != expected_correction or current_expected_evidence != expected_evidence:
            raise RuntimeError("reviewed dry-run changed after correction preparation")
        current_state, current_raw = _read_state(state_path)
        _require_apply_gate(current_state, stop_path)
        current_hash = _sha256_bytes(current_raw)
        _verify_correction_membership(
            data_root=data_root,
            state=current_state,
            episode_id=episode_id,
            expected_source_correction=expected_source_correction,
        )
        if current_hash == state_before:
            _atomic_bytes(state_path, updated_raw)
        elif current_hash != state_after:
            raise RuntimeError("state changed after correction preparation")
        final_state, final_raw = _read_state(state_path)
        _require_apply_gate(final_state, stop_path)
        if _sha256_bytes(final_raw) != state_after:
            raise RuntimeError("corrected state readback does not match prepared bytes")
        if not _state_contains_correction(
            final_state,
            correction,
            expected_row_sha256=prepared["targetRowSha256After"],
        ):
            raise RuntimeError("corrected result row does not match prepared evidence")
        _verify_source_artifacts(data_root, correction["evidence"]["sourceArtifacts"])
        _verify_correction_membership(
            data_root=data_root,
            state=final_state,
            episode_id=episode_id,
            expected_source_correction=expected_source_correction,
        )
        _, final_handoff_artifact = _validated_global_lock(data_root, global_lock)
        if final_handoff_artifact != handoff_artifact:
            raise RuntimeError("runtime handoff changed after state replacement")
        _read_expected_dry_run(expected_dry_run, expected_sha256=expected_dry_run_sha256)
        completed = {
            **prepared,
            "phase": "completed",
            "applied": True,
            "completedAt": utc_now(),
            "completedHandoffReceipt": final_handoff_artifact,
            "sourceArtifactsVerifiedAfterStateWrite": True,
        }
        _atomic_json(receipt_path, completed)
        return completed

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--global-lock", type=Path)
    parser.add_argument("--expected-dry-run", type=Path)
    parser.add_argument("--expected-dry-run-sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        if (
            args.receipt is None
            or args.global_lock is None
            or args.expected_dry_run is None
            or args.expected_dry_run_sha256 is None
        ):
            parser.error(
                "--apply requires --receipt, --global-lock, --expected-dry-run, "
                "and --expected-dry-run-sha256"
            )
        result = apply_correction(
            data_root=args.data_root,
            state_path=args.state,
            episode_id=args.episode_id,
            receipt_path=args.receipt,
            global_lock=args.global_lock,
            expected_dry_run=args.expected_dry_run,
            expected_dry_run_sha256=args.expected_dry_run_sha256,
        )
    else:
        if (
            args.receipt is not None
            or args.global_lock is not None
            or args.expected_dry_run is not None
            or args.expected_dry_run_sha256 is not None
        ):
            parser.error(
                "dry-run accepts no write, lock, or expected-evidence options; "
                "redirect stdout for a local record"
            )
        result = build_correction(
            data_root=args.data_root,
            state_path=args.state,
            episode_id=args.episode_id,
        )
        result = {**result, "phase": "dry_run", "applied": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
