"""Persistent Jev NetHack runner that stops only on a verified ascension.

This runtime intentionally separates three lifetimes:

* a NetHack episode continues until NLE ends it or the explicit engine limit;
* Jev accounting rotates in bounded chunks without resetting that episode; and
* a process fragment ends on a clean stop, fatal preservation error, or host
  interruption. A later process reconstructs that exact seeded game from the
  durable action journal and continues only after every replay check passes.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from nle import nethack
from nle.env.tasks import NetHackScore

import run as game
from broadcast import (
    DEFAULT_INGEST_URL,
    DEFAULT_TOKEN_FILE,
    BroadcastRecorder,
    IngestClient,
    read_secret,
    sha256_file,
    utc_now,
)
from jev_client import (
    MAX_REQUEST_BYTES,
    JevBudgetExceededError,
    JevClient,
    JevClientError,
)
from menu_labels import contextual_choices
from recovery import ReplayPlanError, load_legacy_plan, load_native_plan
from transition_pack import (
    SCHEMA as TRANSITION_SCHEMA,
    TransitionPackError,
    TransitionPackWriter,
    scan_pack,
)


RUNTIME_SCHEMA = "jev-nethack-until-win/v1"
DEFAULT_DATA_ROOT = Path.home() / "Library" / "Application Support" / "JevNetHack" / "recordings" / "until-win"
DEFAULT_GLOBAL_LOCK = Path.home() / "Library" / "Application Support" / "JevNetHack" / "gameplay.lock"
DEFAULT_ENGINE_EPISODE_LIMIT = 1_000_000
DEFAULT_JEV_CHUNK_CALLS = 1_000
DEFAULT_JEV_CHUNK_BYTES = 24 * 1024 * 1024
DEFAULT_BACKOFF_SECONDS = 5.0
DEFAULT_MAX_BACKOFF_SECONDS = 300.0


class UntilWinError(Exception):
    """Base runtime error."""


class AlreadyRunningError(UntilWinError):
    """Another process owns the runtime lock."""


class ConflictingGameplayError(AlreadyRunningError):
    """A legacy/bounded Jev NetHack process is still active."""


class PreservationError(UntilWinError):
    """Gameplay must halt because a transition cannot be preserved."""


class RecoveryBlockedError(PreservationError):
    """Exact deterministic reconstruction could not be proved."""


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    data = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def observation_digest(observation: Mapping[str, Any]) -> str:
    """Hash all observation arrays with names, dtypes, shapes, and byte order."""
    digest = hashlib.sha256()
    digest.update(b"jev-nethack-observation/v1\0")
    for key in sorted(observation):
        if not isinstance(key, str) or not key:
            raise PreservationError("observation keys must be non-empty strings")
        array = np.ascontiguousarray(np.asarray(observation[key]))
        if array.dtype.hasobject:
            raise PreservationError(f"observation {key!r} has object dtype")
        header = canonical_json(
            {"key": key, "dtype": array.dtype.str, "shape": list(array.shape)}
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def write_initial_observation(path: Path, observation: Mapping[str, Any], digest: str) -> None:
    """Atomically preserve the complete reset observation used for replay."""
    arrays = {f"obs__{key}": np.array(value, copy=True) for key, value in observation.items()}
    arrays["metadata_json"] = np.frombuffer(
        canonical_json({"schema": RUNTIME_SCHEMA, "observationDigest": digest}).encode("utf-8"),
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableJsonl:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8")

    def append(self, value: Mapping[str, Any]) -> None:
        self._stream.write(canonical_json(value) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


class SingleInstanceLock:
    """Kernel-backed single-instance lock; stale PID text is never trusted."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._stream.close()
            self._stream = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise AlreadyRunningError("until-win runtime lock is already held") from None
            raise
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(canonical_json({"pid": os.getpid(), "acquiredAt": utc_now(), "schema": RUNTIME_SCHEMA}) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def release(self) -> None:
        if self._stream is None:
            return
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.release()


def lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


class RotatingJevPolicy:
    """Bounded Jev clients that rotate without changing the game episode."""

    def __init__(
        self,
        *,
        api_key: str,
        ledger_root: Path,
        max_calls_per_chunk: int,
        max_bytes_per_chunk: int,
        client_factory: Callable[..., Any] = JevClient,
    ) -> None:
        if max_calls_per_chunk <= 0 or max_bytes_per_chunk < MAX_REQUEST_BYTES:
            raise UntilWinError("Jev chunk bounds are invalid")
        self.api_key = api_key
        self.ledger_root = ledger_root
        self.ledger_root.mkdir(parents=True, exist_ok=False)
        self.max_calls_per_chunk = max_calls_per_chunk
        self.max_bytes_per_chunk = max_bytes_per_chunk
        self.client_factory = client_factory
        self.chunk = 0
        self.client: Any = None
        self.receipt_path: Path | None = None
        self._new_chunk()

    def _receipt(self, status: str, error_type: str | None = None) -> None:
        assert self.client is not None and self.receipt_path is not None
        receipt = {
            "schema": RUNTIME_SCHEMA,
            "chunk": self.chunk,
            "status": status,
            "updatedAt": utc_now(),
            "maxCalls": self.max_calls_per_chunk,
            "maxInputBytes": self.max_bytes_per_chunk,
            "attemptedCalls": int(self.client.calls_used),
            "reservedRequestBytes": int(self.client.input_bytes_used),
            "errorType": error_type,
        }
        write_json_atomic(self.receipt_path, receipt)

    def _new_chunk(self) -> None:
        if self.client is not None:
            self._receipt("rotated")
        self.chunk += 1
        self.client = self.client_factory(
            max_calls=self.max_calls_per_chunk,
            max_input_bytes=self.max_bytes_per_chunk,
            api_key=self.api_key,
        )
        self.receipt_path = self.ledger_root / f"jev-chunk-{self.chunk:06d}.json"
        self._receipt("open")

    def choose(self, state: Mapping[str, Any], criteria: Mapping[str, str], instructions: str) -> dict[str, Any]:
        assert self.client is not None
        if (
            self.client.calls_used >= self.max_calls_per_chunk
            or self.max_bytes_per_chunk - self.client.input_bytes_used < MAX_REQUEST_BYTES
        ):
            self._new_chunk()
        try:
            decision = self.client.choose(state, criteria, instructions)
        except JevBudgetExceededError:
            self._receipt("budget_exhausted", "JevBudgetExceededError")
            self._new_chunk()
            decision = self.client.choose(state, criteria, instructions)
        except JevClientError as exc:
            self._receipt("provider_error", type(exc).__name__)
            raise
        self._receipt("open")
        enriched = dict(decision)
        enriched["accountingChunk"] = self.chunk
        return enriched

    def close(self, status: str = "closed") -> None:
        if self.client is not None:
            self._receipt(status)


class ErrorBackoff:
    def __init__(self, initial: float, maximum: float, sleep: Callable[[float], None] = time.sleep) -> None:
        if initial <= 0 or maximum < initial:
            raise UntilWinError("backoff bounds are invalid")
        self.initial = initial
        self.maximum = maximum
        self.sleep = sleep
        self.failures = 0

    def reset(self) -> None:
        self.failures = 0

    def next_delay(self) -> float:
        return min(self.maximum, self.initial * (2 ** self.failures))

    def wait(self, should_stop: Callable[[], bool]) -> float:
        delay = self.next_delay()
        self.failures += 1
        remaining = delay
        while remaining > 0 and not should_stop():
            interval = min(1.0, remaining)
            self.sleep(interval)
            remaining -= interval
        return delay


def copy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def compact_step_ranges(steps: list[int] | set[int]) -> list[dict[str, int]]:
    ordered = sorted(set(steps))
    if not ordered:
        return []
    ranges: list[dict[str, int]] = []
    start = previous = ordered[0]
    for step in ordered[1:]:
        if step != previous + 1:
            ranges.append({"first": start, "last": previous, "count": previous - start + 1})
            start = step
        previous = step
    ranges.append({"first": start, "last": previous, "count": previous - start + 1})
    return ranges


def end_status_name(info: Mapping[str, Any]) -> str | None:
    value = info.get("end_status")
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def make_env(character: str, engine_episode_limit: int, episode_dir: Path) -> Any:
    ttyrec_dir = episode_dir / "nle-ttyrec"
    ttyrec_dir.mkdir(parents=True, exist_ok=False)
    # Construct the NLE task directly. Gymnasium consumes max_episode_steps as
    # a TimeLimit override in gym.make(), leaving NLE's inner 5,000-step abort
    # unchanged. Direct construction applies the intended full-game limit to
    # the engine itself and avoids a second, ambiguous outer limit.
    return NetHackScore(
        actions=nethack.ACTIONS,
        character=character,
        max_episode_steps=engine_episode_limit,
        allow_all_modes=True,
        allow_all_yn_questions=True,
        penalty_step=0.0,
        penalty_time=0.0,
        fix_moon_phase=True,
        save_ttyrec_every=1,
        savedir=str(ttyrec_dir),
    )


def conflicting_gameplay_processes(rows: str, *, own_pid: int) -> list[dict[str, Any]]:
    """Identify known Jev NetHack runners that predate the shared lock."""
    markers = (
        "/JevNetHack/runner/broadcast.py",
        "/outputs/jev-nethack/broadcast.py",
        "/JevNetHack/launch_hour.py",
    )
    conflicts: list[dict[str, Any]] = []
    for raw_line in rows.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        pieces = stripped.split(None, 1)
        if len(pieces) != 2 or not pieces[0].isdigit():
            continue
        pid = int(pieces[0])
        command = pieces[1]
        if pid != own_pid and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "commandSha256": hashlib.sha256(command.encode()).hexdigest()})
    return conflicts


def inspect_gameplay_conflicts(*, own_pid: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return conflicting_gameplay_processes(completed.stdout, own_pid=own_pid)


def deterministic_contract(source_root: Path, *, character: str, seed: int, engine_episode_limit: int) -> dict[str, Any]:
    action_bytes = canonical_json({"keycodes": [int(action) for action in nethack.ACTIONS]}).encode()
    source_names = ("run.py", "menu_labels.py")
    return {
        "schema": "jev-nethack-deterministic-contract/v1",
        "character": character,
        "seed": seed,
        "engineEpisodeLimit": engine_episode_limit,
        "rng": "core=seed;disp=seed+100000;lgen=seed+200000;reseed=false;fix_moon_phase=true",
        "environment": {
            "task": "nle.env.tasks.NetHackScore",
            "allowAllModes": True,
            "allowAllYnQuestions": True,
            "penaltyStep": 0.0,
            "penaltyTime": 0.0,
            "saveTtyrecEvery": 1,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("nle", "gymnasium", "numpy")
        },
        "actionCount": len(nethack.ACTIONS),
        "actionMapSha256": hashlib.sha256(action_bytes).hexdigest(),
        "sourceSha256": {
            name: sha256_file(source_root / name)
            for name in source_names
            if (source_root / name).exists()
        },
    }


def prepare_legacy_resume_state(
    *,
    data_root: Path,
    source: Path,
    episode: int,
    engine_episode_limit: int,
) -> dict[str, Any]:
    """Persist a legacy plan without starting NLE or contacting Jev."""
    state_path = data_root / "state.json"
    if state_path.exists():
        raise RecoveryBlockedError("legacy import requires an unused data root")
    try:
        plan = load_legacy_plan(source.resolve(), episode)
    except ReplayPlanError as exc:
        raise RecoveryBlockedError(str(exc)) from exc
    data_root.mkdir(parents=True, exist_ok=True)
    plan_path = data_root / "recovery" / "legacy-replay-plan.json"
    write_json_atomic(plan_path, plan)
    candidate = {
        "episodeId": plan["episodeId"],
        "seed": plan["seed"],
        "character": plan["character"],
        "fragments": [],
        "legacyPlan": str(plan_path.relative_to(data_root)),
        "origin": plan["origin"],
        "startedAt": utc_now(),
        "totalActionsAtStart": 0,
    }
    state = {
        "schema": RUNTIME_SCHEMA,
        "status": "recovery_pending",
        "createdAt": utc_now(),
        "updatedAt": utc_now(),
        "nextEpisodeId": max(episode + 1, 1),
        "nextSeed": plan["seed"] + 1,
        "totalActions": 0,
        "completedEpisodes": 0,
        "interruptedEpisodes": 0,
        "ascensions": 0,
        "recoveries": 0,
        "activeEpisode": None,
        "activeFragment": None,
        "resumeCandidate": candidate,
        "winningTransition": None,
        "legacyImport": {
            "preparedAt": utc_now(),
            "plan": candidate["legacyPlan"],
            "engineEpisodeLimit": engine_episode_limit,
            "sourceEvidence": plan["sources"],
            "providerCalls": 0,
        },
    }
    write_json_atomic(state_path, state)
    return state


class UntilWinSupervisor:
    def __init__(
        self,
        *,
        data_root: Path,
        character: str,
        first_seed: int,
        engine_episode_limit: int,
        policy: Any,
        stop_event: threading.Event,
        stop_file: Path,
        env_factory: Callable[[str, int, Path], Any] = make_env,
        recorder: Any = None,
        backoff: ErrorBackoff | None = None,
        source_root: Path | None = None,
    ) -> None:
        if first_seed < 0 or engine_episode_limit <= 0:
            raise UntilWinError("seed must be non-negative and engine episode limit positive")
        self.data_root = data_root.expanduser().resolve()
        self.character = character
        self.first_seed = first_seed
        self.engine_episode_limit = engine_episode_limit
        self.policy = policy
        self.stop_event = stop_event
        self.stop_file = stop_file
        self.env_factory = env_factory
        self.recorder = recorder
        self.backoff = backoff or ErrorBackoff(DEFAULT_BACKOFF_SECONDS, DEFAULT_MAX_BACKOFF_SECONDS)
        self.source_root = source_root or Path(__file__).parent
        self.state_path = self.data_root / "state.json"
        self.recovery_log = DurableJsonl(self.data_root / "recovery.jsonl")
        self.state = self._load_state()
        self.fragment_dir: Path | None = None
        self.events: DurableJsonl | None = None
        self.training: TransitionPackWriter | None = None

    def should_stop(self) -> bool:
        return self.stop_event.is_set() or self.stop_file.exists()

    def _load_state(self) -> dict[str, Any]:
        self.data_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text())
            if not isinstance(loaded, dict) or loaded.get("schema") != RUNTIME_SCHEMA:
                raise PreservationError("existing until-win state has an unsupported schema")
            state = loaded
        else:
            state = {
                "schema": RUNTIME_SCHEMA,
                "status": "ready",
                "createdAt": utc_now(),
                "updatedAt": utc_now(),
                "nextEpisodeId": 1,
                "nextSeed": self.first_seed,
                "totalActions": 0,
                "completedEpisodes": 0,
                "interruptedEpisodes": 0,
                "ascensions": 0,
                "bestScore": 0,
                "maxDepth": 0,
                "episodeResults": [],
                "recoveries": 0,
                "activeEpisode": None,
                "activeFragment": None,
                "winningTransition": None,
            }
            write_json_atomic(self.state_path, state)
        active = state.get("activeEpisode")
        if active and state.get("status") == "running":
            resume_candidate = dict(active)
            if not isinstance(resume_candidate.get("totalActionsAtStart"), int):
                committed = int(resume_candidate.get("stepsCommitted", 0))
                resume_candidate["totalActionsAtStart"] = max(
                    0,
                    int(state.get("totalActions", 0)) - committed,
                )
            fragments = list(resume_candidate.get("fragments") or [])
            active_fragment = state.get("activeFragment")
            if active_fragment and active_fragment not in fragments:
                fragments.append(active_fragment)
            resume_candidate["fragments"] = fragments
            recovery = {
                "schema": RUNTIME_SCHEMA,
                "event": "deterministic_replay_pending",
                "detectedAt": utc_now(),
                "activeEpisode": resume_candidate,
                "activeFragment": active_fragment,
                "reason": "previous process ended without a clean state transition; exact seed/action replay must verify every raw observation before continuing",
            }
            self.recovery_log.append(recovery)
            state["resumeCandidate"] = resume_candidate
            state["interruptedEpisodes"] = int(state.get("interruptedEpisodes", 0)) + 1
            state["lastRecoveryAttempt"] = recovery
            state["activeEpisode"] = None
            state["activeFragment"] = None
            state["status"] = "recovery_pending"
            state["updatedAt"] = utc_now()
            write_json_atomic(self.state_path, state)
        return state

    def _persist_state(self) -> None:
        self.state["updatedAt"] = utc_now()
        try:
            write_json_atomic(self.state_path, self.state)
        except (OSError, TypeError, ValueError) as exc:
            raise PreservationError("cannot durably persist supervisor state") from exc

    def _source_hashes(self) -> dict[str, str]:
        names = (
            "until_win.py",
            "recovery.py",
            "transition_pack.py",
            "run.py",
            "jev_client.py",
            "menu_labels.py",
            "broadcast.py",
        )
        return {
            name: sha256_file(self.source_root / name)
            for name in names
            if (self.source_root / name).exists()
        }

    def _public_metrics(self) -> dict[str, Any]:
        return {
            "scope": "continuous_run",
            "totalActions": int(self.state.get("totalActions", 0)),
            "completedEpisodes": int(self.state.get("completedEpisodes", 0)),
            "ascensions": int(self.state.get("ascensions", 0)),
            "interruptedEpisodes": int(self.state.get("interruptedEpisodes", 0)),
            "bestScore": int(self.state.get("bestScore", 0)),
            "maxDepth": int(self.state.get("maxDepth", 0)),
            "recoveries": int(self.state.get("recoveries", 0)),
            "episodeResults": list(self.state.get("episodeResults") or [])[-100:],
        }

    def _set_recorder_telemetry(self, phase: str) -> None:
        if self.recorder is not None:
            self.recorder.set_telemetry(
                metrics=self._public_metrics(),
                runtime_status={"phase": phase},
            )

    def _report_recorder_status(
        self,
        phase: str,
        *,
        error_type: str | None = None,
        retry_at: str | None = None,
    ) -> None:
        if self.recorder is not None:
            self.recorder.set_telemetry(metrics=self._public_metrics())
            self.recorder.report_status(phase, error_type=error_type, retry_at=retry_at)

    def _begin_fragment(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        fragment_id = f"fragment-{stamp}-pid{os.getpid()}-{secrets.token_hex(4)}"
        self.fragment_dir = self.data_root / "fragments" / fragment_id
        self.fragment_dir.mkdir(parents=True, exist_ok=False)
        self.events = DurableJsonl(self.fragment_dir / "transitions.jsonl")
        self.training = TransitionPackWriter(self.fragment_dir / "training")
        config = {
            "schema": RUNTIME_SCHEMA,
            "fragmentId": fragment_id,
            "startedAt": utc_now(),
            "totalActionsAtStart": int(self.state.get("totalActions", 0)),
            "pid": os.getpid(),
            "character": self.character,
            "engineEpisodeLimit": self.engine_episode_limit,
            "firstAllocatedSeed": self.state["nextSeed"],
            "transitionSchema": TRANSITION_SCHEMA,
            "sourceSha256": self._source_hashes(),
            "winCondition": "terminated == true and info.is_ascended == true; NLE derives is_ascended from how_done() == ASCENDED",
        }
        write_json_atomic(self.fragment_dir / "config.json", config)
        self.state["status"] = "running"
        self.state["activeFragment"] = str(self.fragment_dir.relative_to(self.data_root))
        self._persist_state()

    def _attempt_directory(self, episode_id: int, seed: int, label: str) -> Path:
        assert self.fragment_dir is not None
        attempt = self.fragment_dir / "episodes" / f"episode-{episode_id:08d}-seed-{seed}-{label}"
        attempt.mkdir(parents=True, exist_ok=False)
        return attempt

    def _allocate_episode(self) -> tuple[int, int, Path, Path, None]:
        assert self.fragment_dir is not None
        episode_id = int(self.state["nextEpisodeId"])
        seed = int(self.state["nextSeed"])
        episode_dir = self._attempt_directory(episode_id, seed, "initial")
        global_dir = self.data_root / "episodes" / f"episode-{episode_id:08d}-seed-{seed}"
        global_dir.mkdir(parents=True, exist_ok=False)
        self.state["nextEpisodeId"] = episode_id + 1
        self.state["nextSeed"] = seed + 1
        self.state["activeEpisode"] = {
            "episodeId": episode_id,
            "seed": seed,
            "directory": str(episode_dir.relative_to(self.data_root)),
            "globalDirectory": str(global_dir.relative_to(self.data_root)),
            "fragments": [self.state["activeFragment"]],
            "startedAt": utc_now(),
            "totalActionsAtStart": int(self.state.get("totalActions", 0)),
        }
        self._persist_state()
        return episode_id, seed, episode_dir, global_dir, None

    def _load_candidate_plan(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        legacy_relative = candidate.get("legacyPlan")
        native_plan: dict[str, Any] | None = None
        if candidate.get("fragments"):
            try:
                native_plan = load_native_plan(self.data_root, candidate)
            except ReplayPlanError as exc:
                raise RecoveryBlockedError(str(exc)) from exc
        if not legacy_relative:
            if native_plan is None:
                raise RecoveryBlockedError("resume candidate has no replay evidence")
            if native_plan.get("firstStep", 0) != 0:
                raise RecoveryBlockedError("native replay evidence does not start at step zero")
            return native_plan
        if not isinstance(legacy_relative, str):
            raise RecoveryBlockedError("legacy replay plan path is invalid")
        legacy_path = (self.data_root / legacy_relative).resolve()
        try:
            legacy_path.relative_to(self.data_root.resolve())
        except ValueError as exc:
            raise RecoveryBlockedError("legacy replay plan escapes the data root") from exc
        legacy_plan = json.loads(legacy_path.read_text())
        if not isinstance(legacy_plan, dict):
            raise RecoveryBlockedError("legacy replay plan is invalid")
        if native_plan is None:
            return legacy_plan
        native_records = native_plan["records"]
        legacy_records = legacy_plan["records"]
        native_by_step = {int(record["step"]): record for record in native_records}
        legacy_by_step = {int(record["step"]): record for record in legacy_records}
        for index in sorted(set(native_by_step) & set(legacy_by_step)):
            native = native_by_step[index]
            legacy = legacy_by_step[index]
            for key in ("step", "actionIndex", "keycode", "reward", "terminated", "truncated"):
                if native.get(key) != legacy.get(key):
                    raise RecoveryBlockedError(f"native/legacy replay conflict at step {index}")
        combined_by_step = dict(legacy_by_step)
        combined_by_step.update(native_by_step)
        combined_steps = sorted(combined_by_step)
        if combined_steps != list(range(len(combined_steps))):
            raise RecoveryBlockedError("legacy/native replay evidence has a gap")
        combined = [combined_by_step[index] for index in combined_steps]
        pending = native_plan.get("pendingIntent")
        if pending is not None and pending.get("step") != len(combined):
            raise RecoveryBlockedError("native pending intent does not follow merged replay evidence")
        result = dict(legacy_plan)
        result["origin"] = "legacy_then_native_full_observation"
        result["records"] = combined
        result["nativeSteps"] = sorted(native_by_step)
        result["legacyRecordCount"] = len(legacy_records)
        result["pendingIntent"] = pending
        result["sources"] = list(legacy_plan.get("sources") or []) + list(native_plan.get("sources") or [])
        if any(step >= len(legacy_records) for step in native_by_step):
            # The native transition at the legacy boundary already verifies
            # that compact state. Reapplying expectedFinalState after replaying
            # later native actions would compare it at the wrong step.
            result.pop("expectedFinalState", None)
        return result

    def migrate_recovery_evidence(self) -> dict[str, Any]:
        """Validate and receipt a paused mixed legacy/native recovery plan."""
        if not self.stop_file.exists():
            raise RecoveryBlockedError("recovery migration requires the STOP latch")
        if self.state.get("status") not in ("paused", "recovery_pending", "recovery_blocked", "preservation_halted"):
            raise RecoveryBlockedError("recovery migration requires a paused or blocked state")
        if self.state.get("activeEpisode") is not None:
            raise RecoveryBlockedError("recovery migration refuses an active episode")
        candidate = self.state.get("resumeCandidate")
        if not isinstance(candidate, Mapping):
            raise RecoveryBlockedError("recovery migration has no resume candidate")
        plan = self._load_candidate_plan(candidate)
        records = list(plan.get("records") or [])
        steps = [record.get("step") for record in records]
        if steps != list(range(len(records))):
            raise RecoveryBlockedError("recovery migration did not produce a contiguous full history")
        native_steps = {int(step) for step in plan.get("nativeSteps") or []}
        legacy_count = int(plan.get("legacyRecordCount", 0))
        regenerated_steps = set(range(legacy_count)) - native_steps
        fingerprint_rows = [
            {
                key: record.get(key)
                for key in (
                    "eventId",
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
                    "verifiedAscension",
                )
            }
            for record in records
        ]
        history_digest = hashlib.sha256(
            canonical_json({"records": fingerprint_rows}).encode("utf-8")
        ).hexdigest()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        receipt = {
            "schema": "jev-nethack-recovery-migration/v1",
            "completed": True,
            "createdAt": utc_now(),
            "migration": "legacy_native_suffix_to_contiguous_replay",
            "episodeId": candidate.get("episodeId"),
            "seed": candidate.get("seed"),
            "actionsPreserved": len(records),
            "legacyRecordCount": legacy_count,
            "nativeEvidenceStepRanges": compact_step_ranges(native_steps),
            "legacyRawRegenerationStepRanges": compact_step_ranges(regenerated_steps),
            "pendingIntentEventId": (plan.get("pendingIntent") or {}).get("eventId"),
            "historyFingerprintSha256": history_digest,
            "sources": list(plan.get("sources") or []),
            "migrationSourceSha256": self._source_hashes(),
            "providerCalls": 0,
            "freshSeedAllocated": False,
        }
        receipt_path = self.data_root / "recovery" / f"migration-{stamp}.json"
        write_json_atomic(receipt_path, receipt)
        self.state["lastRecoveryMigration"] = {
            "path": str(receipt_path.relative_to(self.data_root)),
            "sha256": sha256_file(receipt_path),
            "completed": True,
            "actionsPreserved": len(records),
            "providerCalls": 0,
        }
        self._persist_state()
        return receipt

    def _allocate_resume(self, candidate: Mapping[str, Any]) -> tuple[int, int, Path, Path, dict[str, Any]]:
        episode_id = candidate.get("episodeId")
        seed = candidate.get("seed")
        if not isinstance(episode_id, int) or not isinstance(seed, int):
            raise RecoveryBlockedError("resume candidate has invalid episode or seed")
        if candidate.get("character") not in (None, self.character):
            raise RecoveryBlockedError("resume candidate character does not match runtime character")
        plan = self._load_candidate_plan(candidate)
        attempt_dir = self._attempt_directory(episode_id, seed, "replay")
        global_relative = candidate.get("globalDirectory")
        if isinstance(global_relative, str):
            global_dir = (self.data_root / global_relative).resolve()
            try:
                global_dir.relative_to(self.data_root.resolve())
            except ValueError as exc:
                raise RecoveryBlockedError("episode directory escapes data root") from exc
            global_dir.mkdir(parents=True, exist_ok=True)
        else:
            global_dir = self.data_root / "episodes" / f"episode-{episode_id:08d}-seed-{seed}"
            global_dir.mkdir(parents=True, exist_ok=True)
        fragments = list(candidate.get("fragments") or [])
        if self.state["activeFragment"] not in fragments:
            fragments.append(self.state["activeFragment"])
        active = dict(candidate)
        if not isinstance(active.get("totalActionsAtStart"), int):
            committed = int(active.get("stepsCommitted", 0))
            active["totalActionsAtStart"] = max(
                0,
                int(self.state.get("totalActions", 0)) - committed,
            )
        active.update(
            {
                "directory": str(attempt_dir.relative_to(self.data_root)),
                "globalDirectory": str(global_dir.relative_to(self.data_root)),
                "fragments": fragments,
                "replayStartedAt": utc_now(),
            }
        )
        self.state["activeEpisode"] = active
        self.state["resumeCandidate"] = None
        self._persist_state()
        return episode_id, seed, attempt_dir, global_dir, plan

    def _append(self, value: Mapping[str, Any]) -> None:
        if self.events is None:
            raise PreservationError("transition event log is unavailable")
        try:
            self.events.append(value)
        except (OSError, TypeError, ValueError) as exc:
            raise PreservationError("cannot durably append transition event") from exc

    def _record_transition(
        self,
        *,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.training is None:
            raise PreservationError("transition pack writer is unavailable")
        try:
            return self.training.write(
                observation=observation,
                next_observation=next_observation,
                metadata=metadata,
            )
        except (OSError, TypeError, ValueError, TransitionPackError) as exc:
            raise PreservationError("cannot durably preserve raw transition") from exc

    def _choose_with_backoff(
        self,
        *,
        state: Mapping[str, Any],
        criteria: Mapping[str, str],
        episode_id: int,
        step: int,
    ) -> dict[str, Any] | None:
        while not self.should_stop():
            try:
                decision = self.policy.choose(state, criteria, game.INSTRUCTIONS)
                self.backoff.reset()
                self._set_recorder_telemetry("running")
                return decision
            except JevClientError as exc:
                delay = self.backoff.next_delay()
                retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
                failure = {
                    "schema": RUNTIME_SCHEMA,
                    "eventType": "jev_error",
                    "at": utc_now(),
                    "episodeId": episode_id,
                    "step": step,
                    "errorType": type(exc).__name__,
                    "backoffSeconds": delay,
                    "retryAt": retry_at,
                }
                self._append(failure)
                self.state["lastJevError"] = failure
                self._persist_state()
                self._report_recorder_status(
                    "provider_backoff",
                    error_type=type(exc).__name__,
                    retry_at=retry_at,
                )
                self.backoff.wait(self.should_stop)
        return None

    def _prepare_initial_snapshot(
        self,
        *,
        global_dir: Path,
        observation: Mapping[str, Any],
        seed: int,
        origin: str,
    ) -> str:
        digest = observation_digest(observation)
        contract = deterministic_contract(
            self.source_root,
            character=self.character,
            seed=seed,
            engine_episode_limit=self.engine_episode_limit,
        )
        receipt_path = global_dir / "episode-contract.json"
        snapshot_path = global_dir / "initial-observation.npz"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("deterministicContract") != contract:
                raise RecoveryBlockedError("dependency, source, configuration, or action-map mismatch blocks exact replay")
            if receipt.get("initialObservationDigest") != digest:
                raise RecoveryBlockedError("initial raw observation hash mismatch blocks exact replay")
            artifact = receipt.get("initialObservationArtifact")
            if (
                not snapshot_path.is_file()
                or not isinstance(artifact, Mapping)
                or artifact.get("bytes") != snapshot_path.stat().st_size
                or artifact.get("sha256") != sha256_file(snapshot_path)
            ):
                raise RecoveryBlockedError("initial observation artifact is missing or corrupt")
        else:
            write_initial_observation(snapshot_path, observation, digest)
            receipt = {
                "schema": RUNTIME_SCHEMA,
                "createdAt": utc_now(),
                "origin": origin,
                "initialObservationDigest": digest,
                "initialObservationArtifact": {
                    "filename": snapshot_path.name,
                    "bytes": snapshot_path.stat().st_size,
                    "sha256": sha256_file(snapshot_path),
                },
                "deterministicContract": contract,
            }
            write_json_atomic(receipt_path, receipt)
        return digest

    @staticmethod
    def _require_replay_equal(label: str, actual: Any, expected: Any, step: int) -> None:
        if actual != expected:
            raise RecoveryBlockedError(f"replay mismatch for {label} at step {step}")

    def _play_episode(
        self,
        episode_id: int,
        seed: int,
        episode_dir: Path,
        global_dir: Path,
        replay_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        env = self.env_factory(self.character, self.engine_episode_limit, episode_dir)
        started = time.monotonic()
        visits: Counter[Any] = Counter()
        history: deque[Any] = deque(maxlen=8)
        step = 0
        total_reward = 0.0
        max_depth = 0
        terminated = False
        truncated = False
        verified_ascension = False
        final_end_status: str | None = None
        status = "engine_episode_limit"
        obs: Mapping[str, Any] | None = None
        pending_intent: Mapping[str, Any] | None = None
        expected_final_state: Mapping[str, Any] | None = None
        origin = replay_plan.get("origin") if replay_plan else "native_full_observation"
        try:
            env.unwrapped.seed(core=seed, disp=seed + 100000, lgen=seed + 200000, reseed=False)
            obs, _ = env.reset()
            self._prepare_initial_snapshot(
                global_dir=global_dir,
                observation=copy_observation(obs),
                seed=seed,
                origin=str(origin),
            )

            records = list(replay_plan.get("records") or []) if replay_plan else []
            if replay_plan and "nativeSteps" in replay_plan:
                native_steps = set(replay_plan.get("nativeSteps") or [])
            elif replay_plan and replay_plan.get("origin") == "deterministically_reconstructed_legacy":
                native_steps = set()
            else:
                native_steps = set(range(len(records))) if replay_plan else set()
            pending_intent = replay_plan.get("pendingIntent") if replay_plan else None
            expected_final_state = replay_plan.get("expectedFinalState") if replay_plan else None
            if replay_plan:
                self._report_recorder_status("replaying")
            for record_index, record in enumerate(records):
                if self.should_stop():
                    status = "stop_requested"
                    break
                if record.get("step") != step:
                    raise RecoveryBlockedError(f"noncontiguous replay step {record.get('step')} expected {step}")
                raw_before = copy_observation(obs)
                before_digest = observation_digest(raw_before)
                expected_before_digest = record.get("observationDigest")
                if expected_before_digest is not None:
                    self._require_replay_equal("raw observation hash", before_digest, expected_before_digest, step)
                before_stats = game.stats(obs)
                visits[(before_stats["dungeon"], before_stats["level"], before_stats["x"], before_stats["y"])] += 1
                state = game.make_state(obs, visits, history)
                if isinstance(record.get("state"), Mapping):
                    self._require_replay_equal("compact before state", state, record["state"], step)
                action_index = record.get("actionIndex")
                keycode = record.get("keycode")
                if not isinstance(action_index, int) or not 0 <= action_index < len(env.unwrapped.actions):
                    raise RecoveryBlockedError(f"invalid replay action index at step {step}")
                self._require_replay_equal("action keycode", int(env.unwrapped.actions[action_index]), keycode, step)
                raw_action_label = record.get("actionLabel")
                next_obs, reward, terminated, truncated, info = env.step(action_index)
                raw_after = copy_observation(next_obs)
                after_digest = observation_digest(raw_after)
                expected_after_digest = record.get("nextObservationDigest")
                if expected_after_digest is not None:
                    self._require_replay_equal("next raw observation hash", after_digest, expected_after_digest, step)
                self._require_replay_equal("reward", float(reward), record.get("reward"), step)
                self._require_replay_equal("terminated", bool(terminated), bool(record.get("terminated")), step)
                self._require_replay_equal("truncated", bool(truncated), bool(record.get("truncated")), step)
                native_ascension = bool(info.get("is_ascended", False))
                if record.get("origin") != "deterministically_reconstructed_legacy":
                    self._require_replay_equal("ascension flag", native_ascension, bool(record.get("isAscended")), step)
                after_stats = game.stats(next_obs)
                max_depth = max(max_depth, int(before_stats["depth"]), int(after_stats["depth"]))
                total_reward += float(reward)
                final_end_status = end_status_name(info)
                after_state_before_history = game.make_state(next_obs, visits, history)
                if record.get("provenance", {}).get("origin") == "deterministically_reconstructed_legacy":
                    if isinstance(record.get("nextState"), Mapping):
                        self._require_replay_equal("legacy compact after state", after_state_before_history, record["nextState"], step)
                action_label = raw_action_label or game.action_choices(env.unwrapped.actions).get(f"a{action_index}")
                history.append(
                    {
                        "action": action_label,
                        "before": before_stats,
                        "after": after_stats,
                        "message_after": game.text_line(next_obs["message"]),
                    }
                )
                next_state = game.make_state(next_obs, visits, history)
                if record.get("provenance", {}).get("origin") != "deterministically_reconstructed_legacy" and isinstance(record.get("nextState"), Mapping):
                    self._require_replay_equal("compact next state", next_state, record["nextState"], step)

                # Legacy history did not contain raw NLE arrays. Regenerate
                # them now, but keep provenance explicit and never describe
                # them as historical captures.
                if step not in native_steps and record.get("provenance", {}).get("origin") == "deterministically_reconstructed_legacy":
                    base_criteria = game.action_choices(env.unwrapped.actions)
                    criteria = contextual_choices(state, env.unwrapped.actions, base_criteria)
                    metadata = {
                        **record,
                        "recordedAt": utc_now(),
                        "state": state,
                        # Legacy nextState was captured before the runner
                        # appended this action to recent_actions.  Preserve
                        # that historical compact-state boundary so the raw
                        # regeneration remains replayable on a second restart.
                        "nextState": after_state_before_history,
                        "criteria": criteria,
                        "instructions": game.INSTRUCTIONS,
                        "observationDigest": before_digest,
                        "nextObservationDigest": after_digest,
                        "isAscended": native_ascension,
                        "verifiedAscension": bool(terminated) and native_ascension,
                        "endStatus": final_end_status,
                        "provenance": {
                            **dict(record.get("provenance") or {}),
                            "rawArrays": "regenerated_by_verified_seed_action_replay",
                            "regeneratedAt": utc_now(),
                        },
                    }
                    pack_receipt = self._record_transition(
                        observation=raw_before,
                        next_observation=raw_after,
                        metadata=metadata,
                    )
                    self._append(
                        {
                            "schema": RUNTIME_SCHEMA,
                            "eventType": "transition_committed",
                            **metadata,
                            "rawTransition": pack_receipt,
                        }
                    )
                step += 1
                baseline = int(self.state["activeEpisode"].get("totalActionsAtStart", 0))
                self.state["totalActions"] = max(
                    int(self.state.get("totalActions", 0)),
                    baseline + step,
                )
                self.state["activeEpisode"]["stepsCommitted"] = step
                self.state["activeEpisode"]["lastEventId"] = record.get("eventId")
                self._persist_state()
                self._set_recorder_telemetry("replaying")
                obs = next_obs
                verified_ascension = bool(terminated) and native_ascension
                if terminated or truncated:
                    if record_index != len(records) - 1:
                        raise RecoveryBlockedError("saved replay continues after a terminal environment state")
                    if replay_plan:
                        self.state["recoveries"] = int(self.state.get("recoveries", 0)) + 1
                        self.state["lastRecovery"] = {
                            "episodeId": episode_id,
                            "seed": seed,
                            "verifiedAt": utc_now(),
                            "actionsReplayed": len(records),
                            "providerCalls": 0,
                            "origin": origin,
                        }
                        self.state["fatalLatch"] = None
                        self._persist_state()
                    status = "verified_ascension" if verified_ascension else "engine_truncation" if truncated or final_end_status == "ABORTED" else "game_end"
                    break
            else:
                if replay_plan:
                    self.state["recoveries"] = int(self.state.get("recoveries", 0)) + 1
                    self.state["lastRecovery"] = {
                        "episodeId": episode_id,
                        "seed": seed,
                        "verifiedAt": utc_now(),
                        "actionsReplayed": len(records),
                        "providerCalls": 0,
                        "origin": origin,
                    }
                    self.state["fatalLatch"] = None
                    self._persist_state()
                    self._set_recorder_telemetry("running")

            while not (terminated or truncated) and step < self.engine_episode_limit and not self.should_stop():
                raw_before = copy_observation(obs)
                before_digest = observation_digest(raw_before)
                before_stats = game.stats(obs)
                visits[(before_stats["dungeon"], before_stats["level"], before_stats["x"], before_stats["y"])] += 1
                state = game.make_state(obs, visits, history)
                if expected_final_state is not None:
                    self._require_replay_equal("legacy final compact state", state, expected_final_state, step)
                    expected_final_state = None
                base_criteria = game.action_choices(env.unwrapped.actions)
                criteria = contextual_choices(state, env.unwrapped.actions, base_criteria)
                if self.recorder is not None:
                    self.recorder.observed_frame(state=state, phase="before_action", episode=episode_id, step=step)

                recovered_intent = pending_intent is not None
                if recovered_intent:
                    intent = dict(pending_intent)
                    self._require_replay_equal("pending intent step", intent.get("step"), step, step)
                    if isinstance(intent.get("state"), Mapping):
                        self._require_replay_equal("pending intent state", state, intent["state"], step)
                    if isinstance(intent.get("criteria"), Mapping):
                        self._require_replay_equal("pending intent criteria", criteria, intent["criteria"], step)
                    self._require_replay_equal(
                        "pending intent raw observation hash",
                        before_digest,
                        intent.get("observationDigest"),
                        step,
                    )
                    decision = intent.get("decision")
                    if not isinstance(decision, Mapping):
                        raise RecoveryBlockedError("pending intent has no validated decision")
                    event_id = intent.get("eventId")
                    self._append(
                        {
                            "schema": RUNTIME_SCHEMA,
                            "eventType": "pending_intent_replayed_once",
                            "eventId": event_id,
                            "episodeId": episode_id,
                            "seed": seed,
                            "step": step,
                            "at": utc_now(),
                            "providerCalls": 0,
                        }
                    )
                    pending_intent = None
                else:
                    decision = self._choose_with_backoff(
                        state=state,
                        criteria=criteria,
                        episode_id=episode_id,
                        step=step,
                    )
                    if decision is None:
                        status = "stop_requested"
                        break
                    event_id = f"episode-{episode_id:08d}-step-{step:09d}"
                choice = decision.get("choice")
                if (
                    not isinstance(choice, str)
                    or choice not in criteria
                    or not choice.startswith("a")
                    or not choice[1:].isdigit()
                ):
                    raise RecoveryBlockedError("saved or new Jev choice is outside the validated action criteria")
                action_index = int(choice[1:])
                keycode = int(env.unwrapped.actions[action_index])
                action_label = criteria[choice]
                if recovered_intent:
                    self._require_replay_equal("pending action index", action_index, intent.get("actionIndex"), step)
                    self._require_replay_equal("pending keycode", keycode, intent.get("keycode"), step)
                else:
                    intent = {
                        "schema": RUNTIME_SCHEMA,
                        "eventType": "action_intent",
                        "eventId": event_id,
                        "at": utc_now(),
                        "episodeId": episode_id,
                        "seed": seed,
                        "step": step,
                        "state": state,
                        "decision": decision,
                        "criteria": criteria,
                        "instructions": game.INSTRUCTIONS,
                        "actionIndex": action_index,
                        "keycode": keycode,
                        "actionLabel": action_label,
                        "observationDigest": before_digest,
                    }
                    self._append(intent)
                if self.recorder is not None:
                    self.recorder.decision(
                        episode=episode_id,
                        step=step,
                        decision=decision,
                        criteria=criteria,
                    )

                try:
                    next_obs, reward, terminated, truncated, info = env.step(action_index)
                    raw_after = copy_observation(next_obs)
                    after_digest = observation_digest(raw_after)
                    after_stats = game.stats(next_obs)
                    max_depth = max(max_depth, int(before_stats["depth"]), int(after_stats["depth"]))
                    total_reward += float(reward)
                    final_end_status = end_status_name(info)
                    native_ascension = bool(info.get("is_ascended", False))
                    verified_ascension = bool(terminated) and native_ascension
                    history.append(
                        {
                            "action": action_label,
                            "before": before_stats,
                            "after": after_stats,
                            "message_after": game.text_line(next_obs["message"]),
                        }
                    )
                    next_state = game.make_state(next_obs, visits, history)
                    metadata = {
                        "eventId": event_id,
                        "recordedAt": utc_now(),
                        "episodeId": episode_id,
                        "seed": seed,
                        "step": step,
                        "state": state,
                        "nextState": next_state,
                        "decision": dict(decision),
                        "criteria": criteria,
                        "instructions": game.INSTRUCTIONS,
                        "actionIndex": action_index,
                        "keycode": keycode,
                        "actionLabel": action_label,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "isAscended": native_ascension,
                        "verifiedAscension": verified_ascension,
                        "endStatus": final_end_status,
                        "observationDigest": before_digest,
                        "nextObservationDigest": after_digest,
                        "recoveredPendingIntent": recovered_intent,
                        "provenance": {"origin": "native_full_observation"},
                    }
                    pack_receipt = self._record_transition(
                        observation=raw_before,
                        next_observation=raw_after,
                        metadata=metadata,
                    )
                    committed = {
                        "schema": RUNTIME_SCHEMA,
                        "eventType": "transition_committed",
                        **metadata,
                        "rawTransition": pack_receipt,
                    }
                    self._append(committed)
                    step += 1
                    self.state["totalActions"] = int(self.state.get("totalActions", 0)) + 1
                    self.state["activeEpisode"]["stepsCommitted"] = step
                    self.state["activeEpisode"]["lastEventId"] = event_id
                    self._persist_state()
                    self._set_recorder_telemetry("running")
                    if self.recorder is not None:
                        action_record = {"index": action_index, "keycode": keycode, "label": action_label}
                        self.recorder.observed_frame(
                            state=next_state,
                            phase="after_action",
                            episode=episode_id,
                            step=step,
                            action=action_record,
                            reward=float(reward),
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            is_ascended=verified_ascension,
                        )
                        self.recorder.action_finished()
                        if self.recorder.should_finalize_segment():
                            self.recorder.finalize_segment()
                    obs = next_obs
                except RecoveryBlockedError:
                    raise
                except Exception as exc:
                    incomplete = {
                        "schema": RUNTIME_SCHEMA,
                        "eventId": event_id,
                        "episodeId": episode_id,
                        "seed": seed,
                        "step": step,
                        "actionIndex": action_index,
                        "keycode": keycode,
                        "detectedAt": utc_now(),
                        "errorType": type(exc).__name__,
                        "recovery": "replay the persisted intent exactly once from the last verified transition",
                    }
                    self.state["lastIncompleteTransition"] = incomplete
                    try:
                        write_json_atomic(global_dir / f"incomplete-step-{step:09d}.json", incomplete)
                        self._persist_state()
                    except Exception:
                        pass
                    raise PreservationError("an action may have executed before its durable transition commit") from exc

                if verified_ascension:
                    status = "verified_ascension"
                    break
                if truncated or final_end_status == "ABORTED":
                    status = "engine_truncation"
                    break
                if terminated:
                    status = "game_end"
                    break
            if self.should_stop() and status == "engine_episode_limit":
                status = "stop_requested"
        finally:
            env.close()
        summary = {
            "schema": RUNTIME_SCHEMA,
            "episodeId": episode_id,
            "seed": seed,
            "endedAt": utc_now(),
            "status": status,
            "steps": step,
            "score": game.stats(obs)["score"] if obs is not None else None,
            "maxDepth": max_depth,
            "totalReward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "isAscended": verified_ascension,
            "endStatus": final_end_status,
            "wallSeconds": time.monotonic() - started,
            "replayOrigin": origin if replay_plan else None,
        }
        write_json_atomic(episode_dir / "summary.json", summary)
        self._append({"eventType": "episode_summary", **summary})
        return summary

    def run(self, *, max_episodes: int | None = None) -> dict[str, Any]:
        if self.state.get("status") == "won" and self.state.get("winningTransition"):
            evidence = self.state["winningTransition"]
            candidate = evidence.get("replayCandidate") if isinstance(evidence, Mapping) else None
            try:
                if not isinstance(candidate, Mapping):
                    raise RecoveryBlockedError("winning state has no immutable replay evidence")
                plan = self._load_candidate_plan(candidate)
                records = plan.get("records") or []
                last = records[-1] if records else None
                if (
                    not isinstance(last, Mapping)
                    or last.get("eventId") != evidence.get("eventId")
                    or last.get("terminated") is not True
                    or last.get("verifiedAscension") is not True
                    or last.get("isAscended") is not True
                ):
                    raise RecoveryBlockedError("winning state does not match a committed ascension transition")
                self.state["winningTransition"]["evidenceRevalidatedAt"] = utc_now()
                self._persist_state()
                self.recovery_log.close()
                return self.state
            except RecoveryBlockedError:
                self.state["status"] = "recovery_blocked"
                self.state["fatalLatch"] = {"reason": "winning_evidence_invalid", "at": utc_now()}
                self._persist_state()
                self.recovery_log.close()
                raise
        if self.state.get("fatalLatch") and not isinstance(self.state.get("resumeCandidate"), Mapping):
            self.state["status"] = "recovery_blocked"
            self._persist_state()
            self.recovery_log.close()
            raise RecoveryBlockedError("fatal preservation latch has no verified resume candidate")
        self._begin_fragment()
        self._set_recorder_telemetry("starting")
        episodes_this_process = 0
        exit_reason = "stop_requested"
        fragment_complete = False
        resume_candidate = self.state.get("resumeCandidate")
        try:
            while not self.should_stop():
                if max_episodes is not None and episodes_this_process >= max_episodes:
                    exit_reason = "test_episode_limit"
                    break
                if isinstance(resume_candidate, Mapping):
                    allocation = self._allocate_resume(resume_candidate)
                    resume_candidate = None
                else:
                    allocation = self._allocate_episode()
                episode_id, seed, episode_dir, global_dir, replay_plan = allocation
                active_snapshot = dict(self.state["activeEpisode"])
                try:
                    summary = self._play_episode(
                        episode_id,
                        seed,
                        episode_dir,
                        global_dir,
                        replay_plan,
                    )
                except RecoveryBlockedError as exc:
                    exit_reason = "recovery_blocked"
                    candidate = dict(self.state.get("activeEpisode") or active_snapshot)
                    self.state["resumeCandidate"] = candidate
                    self.state["activeEpisode"] = None
                    self.state["status"] = "recovery_blocked"
                    self.state["fatalLatch"] = {
                        "reason": "deterministic_replay_not_verified",
                        "errorType": type(exc).__name__,
                        "detail": str(exc),
                        "at": utc_now(),
                    }
                    self._persist_state()
                    self._report_recorder_status("recovery_blocked", error_type=type(exc).__name__)
                    raise
                except PreservationError as exc:
                    exit_reason = "preservation_halted"
                    candidate = dict(self.state.get("activeEpisode") or active_snapshot)
                    self.state["resumeCandidate"] = candidate
                    self.state["activeEpisode"] = None
                    self.state["status"] = "preservation_halted"
                    self.state["fatalLatch"] = {
                        "reason": "training_data_commit_failed",
                        "errorType": type(exc).__name__,
                        "detail": str(exc),
                        "at": utc_now(),
                    }
                    self._persist_state()
                    self._report_recorder_status("preservation_halted", error_type=type(exc).__name__)
                    raise
                except Exception as exc:
                    exit_reason = "recovery_blocked"
                    candidate = dict(self.state.get("activeEpisode") or active_snapshot)
                    self.state["resumeCandidate"] = candidate
                    self.state["activeEpisode"] = None
                    self.state["status"] = "recovery_blocked"
                    self.state["fatalLatch"] = {
                        "reason": "unexpected_episode_failure",
                        "errorType": type(exc).__name__,
                        "detail": str(exc),
                        "at": utc_now(),
                    }
                    self._persist_state()
                    self._report_recorder_status("recovery_blocked", error_type=type(exc).__name__)
                    raise RecoveryBlockedError("unexpected episode failure blocks fresh-seed fallback") from exc
                episodes_this_process += 1
                if summary["status"] == "stop_requested":
                    self.state["resumeCandidate"] = dict(self.state["activeEpisode"])
                    self.state["lastExitReason"] = "clean_stop_replay_required"
                else:
                    self.state["completedEpisodes"] = int(self.state.get("completedEpisodes", 0)) + 1
                    self.state["resumeCandidate"] = None
                active_snapshot = dict(self.state["activeEpisode"])
                self.state["lastEpisodeCandidate"] = active_snapshot
                self.state["activeEpisode"] = None
                result = {
                    "episodeId": episode_id,
                    "seed": seed,
                    "score": summary.get("score"),
                    "maxDepth": summary.get("maxDepth"),
                    "steps": summary.get("steps"),
                    "status": summary.get("status"),
                    "isAscended": summary.get("isAscended"),
                    "terminated": summary.get("terminated"),
                    "truncated": summary.get("truncated"),
                }
                episode_results = list(self.state.get("episodeResults") or [])
                episode_results.append(result)
                self.state["episodeResults"] = episode_results[-100:]
                self.state["bestScore"] = max(int(self.state.get("bestScore", 0)), int(summary.get("score") or 0))
                self.state["maxDepth"] = max(int(self.state.get("maxDepth", 0)), int(summary.get("maxDepth") or 0))
                if summary["isAscended"]:
                    exit_reason = "verified_ascension"
                    self.state["status"] = "won"
                    self.state["ascensions"] = int(self.state.get("ascensions", 0)) + 1
                    self.state["winningTransition"] = {
                        "episodeId": episode_id,
                        "seed": seed,
                        "summary": str((episode_dir / "summary.json").relative_to(self.data_root)),
                        "verifiedAt": utc_now(),
                        "eventId": active_snapshot.get("lastEventId"),
                        "replayCandidate": active_snapshot,
                    }
                    self._persist_state()
                    write_json_atomic(self.data_root / "WIN.json", self.state["winningTransition"])
                    self._report_recorder_status("verified_ascension")
                    break
                self._persist_state()
            if self.state.get("status") != "won":
                self.state["status"] = "paused" if self.should_stop() else "ready"
                self.state["activeEpisode"] = None
                self.state["lastExitReason"] = exit_reason
                self._persist_state()
                self._report_recorder_status("paused" if self.should_stop() else "ready")
            fragment_complete = True
            return self.state
        finally:
            if self.training is not None:
                try:
                    self.training.close(completed=fragment_complete, reason=exit_reason)
                except (OSError, TransitionPackError) as exc:
                    candidate = self.state.get("resumeCandidate")
                    if not isinstance(candidate, Mapping):
                        winning = self.state.get("winningTransition")
                        candidate = winning.get("replayCandidate") if isinstance(winning, Mapping) else None
                    if not isinstance(candidate, Mapping):
                        candidate = self.state.get("lastEpisodeCandidate")
                    if isinstance(candidate, Mapping):
                        self.state["resumeCandidate"] = dict(candidate)
                    self.state["status"] = "recovery_blocked"
                    self.state["lastError"] = {"type": type(exc).__name__, "at": utc_now()}
                    self.state["fatalLatch"] = {"reason": "training_close_failed", "at": utc_now()}
                    try:
                        self._persist_state()
                    except PreservationError:
                        pass
            if self.events is not None:
                self.events.close()
            self.recovery_log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run sequential episodes until verified ascension")
    run_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run_parser.add_argument("--global-lock", type=Path, default=DEFAULT_GLOBAL_LOCK)
    run_parser.add_argument("--character", default="mon-hum-neu-mal")
    run_parser.add_argument("--first-seed", type=int, default=10_000_001)
    run_parser.add_argument("--engine-episode-limit", type=int, default=DEFAULT_ENGINE_EPISODE_LIMIT)
    run_parser.add_argument("--jev-chunk-calls", type=int, default=DEFAULT_JEV_CHUNK_CALLS)
    run_parser.add_argument("--jev-chunk-bytes", type=int, default=DEFAULT_JEV_CHUNK_BYTES)
    run_parser.add_argument("--jev-token-file", default=None)
    run_parser.add_argument("--ingest-url", default=DEFAULT_INGEST_URL)
    run_parser.add_argument("--ingest-token-file", default=DEFAULT_TOKEN_FILE)
    run_parser.add_argument("--no-ingest", action="store_true")
    run_parser.add_argument("--segment-seconds", type=float, default=60.0)
    run_parser.add_argument("--segment-actions", type=int, default=100)
    run_parser.add_argument("--ingest-timeout", type=float, default=10.0)
    run_parser.add_argument("--ingest-retries", type=int, default=2)
    run_parser.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    run_parser.add_argument("--max-backoff-seconds", type=float, default=DEFAULT_MAX_BACKOFF_SECONDS)
    status_parser = subparsers.add_parser("status", help="read persisted state and kernel-lock status")
    status_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    import_parser = subparsers.add_parser(
        "import-legacy",
        help="prepare an unused data root to replay and resume one preserved legacy game",
    )
    import_parser.add_argument("--data-root", required=True, type=Path)
    import_parser.add_argument("--source", required=True, type=Path)
    import_parser.add_argument("--episode", required=True, type=int)
    import_parser.add_argument("--engine-episode-limit", type=int, default=DEFAULT_ENGINE_EPISODE_LIMIT)
    import_parser.add_argument("--global-lock", type=Path, default=DEFAULT_GLOBAL_LOCK)
    migrate_parser = subparsers.add_parser(
        "migrate-recovery",
        help="validate and receipt a paused mixed legacy/native same-game recovery plan",
    )
    migrate_parser.add_argument("--data-root", required=True, type=Path)
    migrate_parser.add_argument("--character", default="mon-hum-neu-mal")
    migrate_parser.add_argument("--engine-episode-limit", type=int, default=DEFAULT_ENGINE_EPISODE_LIMIT)
    migrate_parser.add_argument("--global-lock", type=Path, default=DEFAULT_GLOBAL_LOCK)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if args.command == "status":
        state_path = data_root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"schema": RUNTIME_SCHEMA, "status": "not_started"}
        state["lockHeld"] = lock_is_held(data_root / "runtime.lock")
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    if args.command == "import-legacy":
        if args.episode < 0 or args.engine_episode_limit <= 0:
            raise SystemExit("episode must be non-negative and engine episode limit positive")
        source = args.source.expanduser().resolve()
        global_lock_path = args.global_lock.expanduser().resolve()
        try:
            with SingleInstanceLock(global_lock_path), SingleInstanceLock(data_root / "runtime.lock"):
                conflicts = inspect_gameplay_conflicts(own_pid=os.getpid())
                if conflicts:
                    raise ConflictingGameplayError(
                        f"refusing import while {len(conflicts)} Jev NetHack process(es) are active"
                    )
                state = prepare_legacy_resume_state(
                    data_root=data_root,
                    source=source,
                    episode=args.episode,
                    engine_episode_limit=args.engine_episode_limit,
                )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        except AlreadyRunningError as exc:
            print(str(exc), file=sys.stderr)
            return 73
        except PreservationError as exc:
            print(str(exc), file=sys.stderr)
            return 74

    if args.command == "migrate-recovery":
        if args.engine_episode_limit <= 0:
            raise SystemExit("engine episode limit must be positive")
        global_lock_path = args.global_lock.expanduser().resolve()
        try:
            with SingleInstanceLock(global_lock_path), SingleInstanceLock(data_root / "runtime.lock"):
                conflicts = inspect_gameplay_conflicts(own_pid=os.getpid())
                if conflicts:
                    raise ConflictingGameplayError(
                        f"refusing migration while {len(conflicts)} Jev NetHack process(es) are active"
                    )
                supervisor = UntilWinSupervisor(
                    data_root=data_root,
                    character=args.character,
                    first_seed=0,
                    engine_episode_limit=args.engine_episode_limit,
                    policy=None,
                    stop_event=threading.Event(),
                    stop_file=data_root / "control" / "STOP",
                )
                try:
                    receipt = supervisor.migrate_recovery_evidence()
                finally:
                    supervisor.recovery_log.close()
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        except AlreadyRunningError as exc:
            print(str(exc), file=sys.stderr)
            return 73
        except PreservationError as exc:
            print(str(exc), file=sys.stderr)
            return 74

    if min(
        args.engine_episode_limit,
        args.jev_chunk_calls,
        args.jev_chunk_bytes,
        args.segment_seconds,
        args.segment_actions,
        args.ingest_timeout,
        args.backoff_seconds,
        args.max_backoff_seconds,
    ) <= 0:
        raise SystemExit("all limits and durations must be positive")
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    lock_path = data_root / "runtime.lock"
    global_lock_path = args.global_lock.expanduser().resolve()
    try:
        with SingleInstanceLock(global_lock_path), SingleInstanceLock(lock_path):
            conflicts = inspect_gameplay_conflicts(own_pid=os.getpid())
            handoff = {
                "schema": RUNTIME_SCHEMA,
                "checkedAt": utc_now(),
                "globalLock": str(global_lock_path),
                "conflicts": conflicts,
                "accepted": not conflicts,
            }
            write_json_atomic(data_root / "handoff-receipt.json", handoff)
            if conflicts:
                raise ConflictingGameplayError(
                    f"refusing overlap with {len(conflicts)} existing Jev NetHack process(es)"
                )
            if (data_root / "state.json").exists():
                persisted = json.loads((data_root / "state.json").read_text())
                if persisted.get("status") == "won" and persisted.get("winningTransition"):
                    verifier = UntilWinSupervisor(
                        data_root=data_root,
                        character=args.character,
                        first_seed=args.first_seed,
                        engine_episode_limit=args.engine_episode_limit,
                        policy=None,
                        stop_event=stop_event,
                        stop_file=data_root / "control" / "STOP",
                    )
                    revalidated = verifier.run()
                    print(json.dumps(revalidated, indent=2, sort_keys=True))
                    return 0
            jev_key = read_secret(env_name="TYPESAFE_API_KEY", token_file=args.jev_token_file)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            ledger_root = data_root / "jev-accounting" / f"process-{stamp}-pid{os.getpid()}"
            policy = RotatingJevPolicy(
                api_key=jev_key,
                ledger_root=ledger_root,
                max_calls_per_chunk=args.jev_chunk_calls,
                max_bytes_per_chunk=args.jev_chunk_bytes,
            )
            ingest = None
            if not args.no_ingest:
                ingest_token = read_secret(env_name="INGEST_TOKEN", token_file=args.ingest_token_file)
                ingest = IngestClient(
                    args.ingest_url,
                    ingest_token,
                    timeout=args.ingest_timeout,
                    retries=args.ingest_retries,
                )
            stream_id = secrets.token_hex(16)
            broadcast_root = data_root / "broadcast-fragments" / f"broadcast-{stamp}-{stream_id}"
            recorder = BroadcastRecorder(
                broadcast_root,
                stream_id=stream_id,
                ingest=ingest,
                segment_seconds=args.segment_seconds,
                segment_actions=args.segment_actions,
            )
            supervisor = UntilWinSupervisor(
                data_root=data_root,
                character=args.character,
                first_seed=args.first_seed,
                engine_episode_limit=args.engine_episode_limit,
                policy=policy,
                stop_event=stop_event,
                stop_file=data_root / "control" / "STOP",
                recorder=recorder,
                backoff=ErrorBackoff(args.backoff_seconds, args.max_backoff_seconds),
            )
            try:
                state = supervisor.run()
            finally:
                reason = "verified_ascension" if supervisor.state.get("status") == "won" else "process_exit"
                try:
                    recorder.finalize(reason=reason)
                finally:
                    policy.close(reason)
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0 if state.get("status") in ("won", "paused") else 1
    except AlreadyRunningError as exc:
        print(str(exc), file=sys.stderr)
        return 73
    except PreservationError as exc:
        print(str(exc), file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
