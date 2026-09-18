"""Paired exact-prefix NLE branch experiment for cycle-aware-v1.

All outputs are development evidence under this candidate directory.  The
script reads the live recovery journal but never writes to it, and creates the
TypeSafe client only after deterministic replay has been verified.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from typing import Any, Mapping


CANDIDATE_ROOT = Path(__file__).resolve().parent
LIVE_SOURCE = Path("<PRIVATE_JEV_ROOT>/until-win")
LIVE_DATA = Path("<PRIVATE_JEV_ROOT>/recordings/until-win")
PREFIX_LAST_STEP = 3029
PREFIX_COUNT = PREFIX_LAST_STEP + 1
EPISODE_ID = 0
SEED = 103
CHARACTER = "mon-hum-neu-mal"
ENGINE_EPISODE_LIMIT = 1_000_000
FUTURE_ACTIONS = 64
SCHEMA = "jev-nethack-cycle-aware-paired-branch/v1"
PROMPT_RE = re.compile(
    r"(--More--|What do you want|In what direction|Pick up what|Really .*\?|\[[ynq]+\])",
    re.IGNORECASE,
)

if str(LIVE_SOURCE) not in sys.path:
    sys.path.insert(0, str(LIVE_SOURCE))

import run as game  # noqa: E402
from candidate import (  # noqa: E402
    BASELINE_INSTRUCTIONS,
    CANDIDATE_INSTRUCTIONS,
    augment_state,
    canonical_json,
    sha256_json,
)
from jev_client import JevClient  # noqa: E402
from menu_labels import contextual_choices  # noqa: E402
from recovery import ReplayPlanError, load_native_plan  # noqa: E402
from transition_pack import TransitionPackWriter, validate_shard  # noqa: E402
from until_win import (  # noqa: E402
    copy_observation,
    end_status_name,
    make_env,
    observation_digest,
)


class BranchError(RuntimeError):
    """The branch experiment could not preserve its comparison contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    data = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
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
        self.stream = path.open("x", encoding="utf-8")

    def append(self, value: Mapping[str, Any]) -> None:
        self.stream.write(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


def require_equal(name: str, actual: Any, expected: Any, step: int) -> None:
    if actual != expected:
        raise BranchError(f"prefix {name} mismatch at step {step}: {actual!r} != {expected!r}")


def source_candidate() -> dict[str, Any]:
    """Read only the identity/path fields needed to find native evidence."""
    state = json.loads((LIVE_DATA / "state.json").read_text())
    active = state.get("activeEpisode")
    if not isinstance(active, dict):
        raise BranchError("live state has no active episode")
    if active.get("episodeId") != EPISODE_ID or active.get("seed") != SEED:
        raise BranchError("live active episode identity changed")
    fragments = active.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise BranchError("live active episode has no fragments")
    return {"episodeId": EPISODE_ID, "seed": SEED, "fragments": list(fragments)}


def load_prefix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an immutable committed prefix, retrying only read/write races at the live tip."""
    last_error: Exception | None = None
    for _ in range(5):
        candidate = source_candidate()
        try:
            plan = load_native_plan(LIVE_DATA, candidate)
        except (ReplayPlanError, OSError, ValueError) as exc:
            last_error = exc
            time.sleep(0.2)
            continue
        by_step = {record.get("step"): record for record in plan.get("records", [])}
        if not all(step in by_step for step in range(PREFIX_COUNT)):
            last_error = BranchError("native evidence does not contain the complete frozen prefix")
            time.sleep(0.2)
            continue
        records = [dict(by_step[step]) for step in range(PREFIX_COUNT)]
        compact = []
        for record in records:
            compact.append(
                {
                    key: record.get(key)
                    for key in (
                        "eventId",
                        "episodeId",
                        "seed",
                        "step",
                        "actionIndex",
                        "keycode",
                        "actionLabel",
                        "observationDigest",
                        "nextObservationDigest",
                        "reward",
                        "terminated",
                        "truncated",
                        "isAscended",
                        "verifiedAscension",
                        "endStatus",
                    )
                }
            )
        provenance = {
            "schema": SCHEMA,
            "episodeId": EPISODE_ID,
            "seed": SEED,
            "prefixFirstStep": 0,
            "prefixLastStep": PREFIX_LAST_STEP,
            "prefixActions": PREFIX_COUNT,
            "prefixRecordsSha256": hashlib.sha256(canonical_json(compact)).hexdigest(),
            "loadedAt": utc_now(),
            "origin": plan.get("origin"),
            "sourceCandidate": candidate,
            "sourceEvidence": plan.get("sources"),
            "liveFilesReadOnly": True,
        }
        return records, provenance
    raise BranchError(f"could not read stable native prefix: {last_error}")


def position(stats: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (int(stats["dungeon"]), int(stats["level"]), int(stats["x"]), int(stats["y"]))


def replay_prefix(records: list[dict[str, Any]], episode_dir: Path) -> tuple[Any, Any, Counter[Any], deque[Any], dict[str, Any]]:
    """Reconstruct the exact branch state before any provider client exists."""
    env = make_env(CHARACTER, ENGINE_EPISODE_LIMIT, episode_dir)
    visits: Counter[Any] = Counter()
    history: deque[Any] = deque(maxlen=8)
    positions: set[tuple[int, int, int, int]] = set()
    started = time.monotonic()
    try:
        env.unwrapped.seed(core=SEED, disp=SEED + 100000, lgen=SEED + 200000, reseed=False)
        obs, _ = env.reset()
        initial_digest = observation_digest(copy_observation(obs))
        for expected_step, record in enumerate(records):
            require_equal("step", record.get("step"), expected_step, expected_step)
            raw_before = copy_observation(obs)
            before_digest = observation_digest(raw_before)
            require_equal("raw before digest", before_digest, record.get("observationDigest"), expected_step)
            before_stats = game.stats(obs)
            before_position = position(before_stats)
            positions.add(before_position)
            visits[before_position] += 1
            state = game.make_state(obs, visits, history)
            if isinstance(record.get("state"), Mapping):
                require_equal("compact before state", state, record["state"], expected_step)
            action_index = record.get("actionIndex")
            if not isinstance(action_index, int) or not 0 <= action_index < len(env.unwrapped.actions):
                raise BranchError(f"invalid prefix action index at step {expected_step}")
            require_equal(
                "action keycode",
                int(env.unwrapped.actions[action_index]),
                record.get("keycode"),
                expected_step,
            )
            next_obs, reward, terminated, truncated, info = env.step(action_index)
            raw_after = copy_observation(next_obs)
            after_digest = observation_digest(raw_after)
            require_equal("raw after digest", after_digest, record.get("nextObservationDigest"), expected_step)
            require_equal("reward", float(reward), record.get("reward"), expected_step)
            require_equal("terminated", bool(terminated), bool(record.get("terminated")), expected_step)
            require_equal("truncated", bool(truncated), bool(record.get("truncated")), expected_step)
            require_equal("ascension", bool(info.get("is_ascended", False)), bool(record.get("isAscended")), expected_step)
            if terminated or truncated:
                raise BranchError(f"frozen prefix reached a terminal state at step {expected_step}")
            after_stats = game.stats(next_obs)
            positions.add(position(after_stats))
            history.append(
                {
                    "action": record.get("actionLabel")
                    or game.action_choices(env.unwrapped.actions).get(f"a{action_index}"),
                    "before": before_stats,
                    "after": after_stats,
                    "message_after": game.text_line(next_obs["message"]),
                }
            )
            next_state = game.make_state(next_obs, visits, history)
            provenance = record.get("provenance") or {}
            if provenance.get("origin") != "deterministically_reconstructed_legacy" and isinstance(record.get("nextState"), Mapping):
                require_equal("compact next state", next_state, record["nextState"], expected_step)
            obs = next_obs
        final_raw = copy_observation(obs)
        final_state = game.make_state(obs, visits, history)
        receipt = {
            "schema": SCHEMA,
            "verified": True,
            "verifiedAt": utc_now(),
            "actionsReplayed": len(records),
            "providerClientConstructedDuringReplay": False,
            "providerCallsDuringReplay": 0,
            "initialObservationDigest": initial_digest,
            "branchPointObservationDigest": observation_digest(final_raw),
            "branchPointStateSha256": sha256_json(final_state),
            "historicalDistinctPositions": len(positions),
            "replayWallSeconds": time.monotonic() - started,
            "branchPointPlayer": final_state["player"],
        }
        return env, obs, visits, history, receipt
    except Exception:
        env.close()
        raise


def frozen_question(arm: str, state: Mapping[str, Any], criteria: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if arm == "baseline":
        return dict(state), BASELINE_INSTRUCTIONS
    if arm == "cycle-aware-v1":
        return augment_state(state), CANDIDATE_INSTRUCTIONS
    raise BranchError(f"unknown arm {arm}")


def prompt_visible(state: Mapping[str, Any]) -> bool:
    text = f"{state.get('message') or ''}\n{state.get('terminal') or ''}"
    return bool(PROMPT_RE.search(text))


def movement_target(state: Mapping[str, Any], action: Any) -> dict[str, Any] | None:
    if action.__class__.__name__ != "CompassDirection":
        return None
    name = getattr(action, "name", None)
    adjacent = state.get("adjacent_cells")
    if not isinstance(name, str) or not isinstance(adjacent, Mapping):
        return None
    cell = adjacent.get(name)
    if not isinstance(cell, Mapping):
        return None
    symbol = cell.get("symbol")
    description = str(cell.get("description") or "")
    blocked = symbol == " " or any(term in description.lower() for term in ("wall", "solid stone", "outside map"))
    return {"direction": name, "symbol": symbol, "description": description, "classifiedBlankOrWall": blocked}


def run_arm(
    *,
    arm: str,
    arm_dir: Path,
    records: list[dict[str, Any]],
    prefix_provenance: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    arm_dir.mkdir(parents=True, exist_ok=False)
    env, obs, visits, history, replay = replay_prefix(records, arm_dir / "replay")
    write_json_atomic(arm_dir / "replay-verification.json", replay)
    if replay["branchPointObservationDigest"] != freeze["branchPointObservationDigest"]:
        env.close()
        raise BranchError(f"{arm} branch point raw digest differs from freeze")
    if replay["branchPointStateSha256"] != freeze["branchPointStateSha256"]:
        env.close()
        raise BranchError(f"{arm} branch point compact state differs from freeze")

    # This is intentionally below the durable replay receipt and its assertions.
    client = JevClient(max_calls=FUTURE_ACTIONS, max_input_bytes=FUTURE_ACTIONS * 24 * 1024)
    events = DurableJsonl(arm_dir / "transitions.jsonl")
    writer = TransitionPackWriter(arm_dir / "training", max_records_per_shard=64)
    future_positions: list[tuple[int, int, int, int]] = [position(game.stats(obs))]
    prefix_positions = {key for key, count in visits.items() if count > 0}
    future_digests: list[str] = [observation_digest(copy_observation(obs))]
    blank_wall_attempts: list[dict[str, Any]] = []
    prompt_steps: list[dict[str, Any]] = []
    no_op_steps: list[int] = []
    unchanged_position_steps: list[int] = []
    start_stats = game.stats(obs)
    player_rows: list[dict[str, int]] = [dict(start_stats)]
    terminal: dict[str, Any] | None = None
    status = "action_limit"
    total_reward = 0.0
    failure: dict[str, Any] | None = None
    try:
        base_criteria = game.action_choices(env.unwrapped.actions)
        if len(base_criteria) != 121:
            raise BranchError(f"expected 121 actions, got {len(base_criteria)}")
        for branch_index in range(FUTURE_ACTIONS):
            global_step = PREFIX_COUNT + branch_index
            raw_before = copy_observation(obs)
            before_digest = observation_digest(raw_before)
            before_stats = game.stats(obs)
            before_position = position(before_stats)
            visits[before_position] += 1
            base_state = game.make_state(obs, visits, history)
            criteria = contextual_choices(base_state, env.unwrapped.actions, base_criteria)
            if list(criteria) != list(base_criteria) or len(criteria) != 121:
                raise BranchError("contextual labels changed the action identity set or order")
            question_state, instructions = frozen_question(arm, base_state, criteria)
            expected_question = freeze["questions"][arm]
            if hashlib.sha256(instructions.encode()).hexdigest() != expected_question["instructionsSha256"]:
                raise BranchError(f"{arm} instructions changed after freeze")
            if branch_index == 0:
                if sha256_json(criteria) != freeze["criteriaAtBranchPointSha256"]:
                    raise BranchError(f"{arm} branch-point criteria changed after freeze")
                if sha256_json(question_state) != expected_question["branchPointQuestionStateSha256"]:
                    raise BranchError(f"{arm} branch-point question state changed after freeze")
            decision = client.choose(question_state, criteria, instructions)
            choice = decision.get("choice")
            if not isinstance(choice, str) or choice not in criteria or not choice.startswith("a") or not choice[1:].isdigit():
                raise BranchError("Jev choice is outside the complete criteria mapping")
            action_index = int(choice[1:])
            keycode = int(env.unwrapped.actions[action_index])
            action_label = criteria[choice]
            event_id = f"paired-{arm}-step-{global_step:09d}"
            intent = {
                "schema": SCHEMA,
                "eventType": "action_intent",
                "eventId": event_id,
                "recordedAt": utc_now(),
                "arm": arm,
                "episodeId": EPISODE_ID,
                "seed": SEED,
                "step": global_step,
                "branchActionIndex": branch_index,
                "state": base_state,
                "questionState": question_state,
                "decision": decision,
                "criteria": criteria,
                "instructions": instructions,
                "actionIndex": action_index,
                "keycode": keycode,
                "actionLabel": action_label,
                "observationDigest": before_digest,
            }
            events.append(intent)
            next_obs, reward, terminated, truncated, info = env.step(action_index)
            raw_after = copy_observation(next_obs)
            after_digest = observation_digest(raw_after)
            after_stats = game.stats(next_obs)
            after_position = position(after_stats)
            native_ascension = bool(info.get("is_ascended", False))
            end_status = end_status_name(info)
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
                **intent,
                "eventType": "transition_committed",
                "nextState": next_state,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "isAscended": native_ascension,
                "verifiedAscension": bool(terminated) and native_ascension,
                "endStatus": end_status,
                "nextObservationDigest": after_digest,
                "provenance": {
                    "experiment": SCHEMA,
                    "prefixRecordsSha256": prefix_provenance["prefixRecordsSha256"],
                    "questionFreezeSha256": freeze["freezeSha256"],
                    "rawArrays": "captured_before_nle_buffer_reuse",
                },
            }
            pack_receipt = writer.write(observation=raw_before, next_observation=raw_after, metadata=metadata)
            events.append({**metadata, "rawTransition": pack_receipt})
            total_reward += float(reward)
            future_positions.append(after_position)
            future_digests.append(after_digest)
            player_rows.append(dict(after_stats))
            if before_digest == after_digest:
                no_op_steps.append(global_step)
            if before_position == after_position:
                unchanged_position_steps.append(global_step)
            target = movement_target(base_state, env.unwrapped.actions[action_index])
            if target and target["classifiedBlankOrWall"]:
                blank_wall_attempts.append(
                    {
                        "step": global_step,
                        "choice": choice,
                        "target": target,
                        "positionChanged": before_position != after_position,
                        "turnChanged": before_stats["turn"] != after_stats["turn"],
                        "messageAfter": game.text_line(next_obs["message"]),
                    }
                )
            if prompt_visible(base_state):
                prompt_steps.append(
                    {
                        "step": global_step,
                        "choice": choice,
                        "actionLabel": action_label,
                        "message": base_state.get("message"),
                        "handledByContextualLabel": bool(
                            isinstance(action_label, str)
                            and ("current prompt" in action_label or "continue past" in action_label or "current inventory prompt" in action_label)
                        ),
                    }
                )
            obs = next_obs
            if terminated or truncated:
                status = "terminal_stop"
                terminal = {
                    "step": global_step,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "isAscended": native_ascension,
                    "verifiedAscension": bool(terminated) and native_ascension,
                    "endStatus": end_status,
                }
                break
    except Exception as exc:
        status = "error"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        completed = status != "error"
        manifest = writer.close(completed=completed, reason=status)
        events.close()
        env.close()

    transitions = len(player_rows) - 1
    reversals = [
        PREFIX_COUNT + index - 1
        for index in range(2, len(future_positions))
        if future_positions[index] == future_positions[index - 2]
        and future_positions[index] != future_positions[index - 1]
    ]
    distinct_future = set(future_positions)
    new_cells = sorted([list(item) for item in distinct_future - prefix_positions])
    end_stats = player_rows[-1]
    summary = {
        "schema": SCHEMA,
        "arm": arm,
        "status": status,
        "developmentOnly": True,
        "notGeneralizationEvidence": True,
        "prefix": dict(prefix_provenance),
        "replay": replay,
        "futureActionsRequested": FUTURE_ACTIONS,
        "futureActionsExecuted": transitions,
        "providerCallsAfterReplay": client.calls_used,
        "providerInputBytes": client.input_bytes_used,
        "positions": [list(item) for item in future_positions],
        "observationDigests": future_digests,
        "distinctPositions": len(distinct_future),
        "newCellsVsPrefix": new_cells,
        "newCellCountVsPrefix": len(new_cells),
        "twoStepReversalSteps": reversals,
        "twoStepReversals": len(reversals),
        "exactRawNoOpSteps": no_op_steps,
        "exactRawNoOps": len(no_op_steps),
        "unchangedPositionSteps": unchanged_position_steps,
        "unchangedPositionActions": len(unchanged_position_steps),
        "blankOrWallAttempts": blank_wall_attempts,
        "blankOrWallAttemptCount": len(blank_wall_attempts),
        "promptSteps": prompt_steps,
        "promptStepCount": len(prompt_steps),
        "startPlayer": start_stats,
        "endPlayer": end_stats,
        "playerTrend": player_rows,
        "turnDelta": int(end_stats["turn"]) - int(start_stats["turn"]),
        "scoreDelta": int(end_stats["score"]) - int(start_stats["score"]),
        "depthDelta": int(end_stats["depth"]) - int(start_stats["depth"]),
        "experienceLevelDelta": int(end_stats["experience_level"]) - int(start_stats["experience_level"]),
        "hpDelta": int(end_stats["hp"]) - int(start_stats["hp"]),
        "hungerCodeStart": int(start_stats["hunger_code"]),
        "hungerCodeEnd": int(end_stats["hunger_code"]),
        "totalReward": total_reward,
        "terminal": terminal,
        "failure": failure,
        "trainingManifest": manifest,
        "completedAt": utc_now(),
    }
    write_json_atomic(arm_dir / "summary.json", summary)
    return summary


def freeze_contract(records: list[dict[str, Any]], provenance: Mapping[str, Any], output: Path) -> dict[str, Any]:
    probe_dir = output / "freeze-replay"
    env, obs, visits, history, replay = replay_prefix(records, probe_dir)
    try:
        before_stats = game.stats(obs)
        visits[position(before_stats)] += 1
        state = game.make_state(obs, visits, history)
        base = game.action_choices(env.unwrapped.actions)
        criteria = contextual_choices(state, env.unwrapped.actions, base)
        if len(criteria) != 121 or list(criteria) != list(base):
            raise BranchError("branch-point criteria is not the complete ordered 121-action mapping")
        contract = {
            "schema": SCHEMA,
            "frozenAt": utc_now(),
            "choiceReceipt": "04d838be-00d0-434f-913e-b9048112a772",
            "prefixRecordsSha256": provenance["prefixRecordsSha256"],
            "branchPointObservationDigest": replay["branchPointObservationDigest"],
            "branchPointStateSha256": replay["branchPointStateSha256"],
            "criteriaCount": len(criteria),
            "criteriaIdsSha256": sha256_json(list(criteria)),
            "criteriaAtBranchPointSha256": sha256_json(criteria),
            "candidatePySha256": sha256_file(CANDIDATE_ROOT / "candidate.py"),
            "runnerSha256": sha256_file(Path(__file__).resolve()),
            "questions": {
                "baseline": {
                    "instructions": BASELINE_INSTRUCTIONS,
                    "instructionsSha256": hashlib.sha256(BASELINE_INSTRUCTIONS.encode()).hexdigest(),
                    "stateTransform": "identity",
                    "branchPointQuestionStateSha256": sha256_json(state),
                },
                "cycle-aware-v1": {
                    "instructions": CANDIDATE_INSTRUCTIONS,
                    "instructionsSha256": hashlib.sha256(CANDIDATE_INSTRUCTIONS.encode()).hexdigest(),
                    "stateTransform": "candidate.augment_state",
                    "branchPointQuestionStateSha256": sha256_json(augment_state(state)),
                },
            },
        }
        contract["freezeSha256"] = sha256_json(contract)
        return contract
    finally:
        env.close()


def compare(run_dir: Path, baseline: Mapping[str, Any], candidate: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    if baseline["replay"]["branchPointObservationDigest"] != candidate["replay"]["branchPointObservationDigest"]:
        raise BranchError("arm raw branch points differ")
    if baseline["replay"]["branchPointStateSha256"] != candidate["replay"]["branchPointStateSha256"]:
        raise BranchError("arm compact branch points differ")
    metric_names = (
        "futureActionsExecuted",
        "distinctPositions",
        "newCellCountVsPrefix",
        "twoStepReversals",
        "exactRawNoOps",
        "unchangedPositionActions",
        "blankOrWallAttemptCount",
        "promptStepCount",
        "turnDelta",
        "scoreDelta",
        "depthDelta",
        "experienceLevelDelta",
        "hpDelta",
        "hungerCodeEnd",
        "totalReward",
    )
    deltas = {
        key: candidate[key] - baseline[key]
        for key in metric_names
        if isinstance(candidate.get(key), (int, float)) and isinstance(baseline.get(key), (int, float))
    }
    comparison = {
        "schema": SCHEMA,
        "completedAt": utc_now(),
        "developmentOnly": True,
        "notGeneralizationEvidence": True,
        "questionsFrozenBeforeBranches": True,
        "freezeSha256": freeze["freezeSha256"],
        "identicalPrefix": True,
        "identicalRawBranchPoint": True,
        "identicalCompactBranchPoint": True,
        "providerCallsDuringReplayEachArm": [
            baseline["replay"]["providerCallsDuringReplay"],
            candidate["replay"]["providerCallsDuringReplay"],
        ],
        "baseline": {key: baseline.get(key) for key in metric_names},
        "cycleAwareV1": {key: candidate.get(key) for key in metric_names},
        "candidateMinusBaseline": deltas,
        "terminal": {"baseline": baseline.get("terminal"), "cycleAwareV1": candidate.get("terminal")},
        "counterevidence": [],
        "interpretationLimits": [
            "This is one paired development branch from one saved state, not held-out or generalization evidence.",
            "Jev confidence and probability changes are not treated as gameplay accuracy.",
            "The experiment does not authorize or perform a live policy migration.",
        ],
    }
    if candidate["blankOrWallAttemptCount"]:
        comparison["counterevidence"].append("The candidate selected at least one movement aimed at a blank/wall-classified adjacent cell.")
    if candidate["twoStepReversals"]:
        comparison["counterevidence"].append("The candidate still produced executed two-step position reversals.")
    if candidate["newCellCountVsPrefix"] <= baseline["newCellCountVsPrefix"]:
        comparison["counterevidence"].append("The candidate did not reach more cells outside the historical prefix than baseline.")
    if candidate["scoreDelta"] <= baseline["scoreDelta"] and candidate["depthDelta"] <= baseline["depthDelta"]:
        comparison["counterevidence"].append("The candidate did not improve score or dungeon depth over baseline in this branch.")
    write_json_atomic(run_dir / "comparison.json", comparison)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(CANDIDATE_ROOT)
    except ValueError as exc:
        raise BranchError("output must remain inside the candidate directory") from exc
    output.mkdir(parents=True, exist_ok=False)
    records, provenance = load_prefix()
    write_json_atomic(output / "prefix-provenance.json", provenance)
    with (output / "branch-prefix.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            compact = {
                key: record.get(key)
                for key in (
                    "eventId", "episodeId", "seed", "step", "actionIndex", "keycode", "actionLabel",
                    "observationDigest", "nextObservationDigest", "reward", "terminated", "truncated",
                    "isAscended", "verifiedAscension", "endStatus",
                )
            }
            stream.write(json.dumps(compact, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    freeze = freeze_contract(records, provenance, output)
    write_json_atomic(output / "question-freeze.json", freeze)
    if args.replay_only:
        write_json_atomic(
            output / "replay-only-result.json",
            {"schema": SCHEMA, "verified": True, "providerCalls": 0, "freeze": freeze},
        )
        return 0
    baseline = run_arm(
        arm="baseline",
        arm_dir=output / "baseline",
        records=records,
        prefix_provenance=provenance,
        freeze=freeze,
    )
    candidate = run_arm(
        arm="cycle-aware-v1",
        arm_dir=output / "cycle-aware-v1",
        records=records,
        prefix_provenance=provenance,
        freeze=freeze,
    )
    comparison = compare(output, baseline, candidate, freeze)
    receipt = {
        "schema": SCHEMA,
        "completed": True,
        "completedAt": utc_now(),
        "runDirectory": str(output.relative_to(CANDIDATE_ROOT)),
        "prefixActions": PREFIX_COUNT,
        "futureActions": baseline["futureActionsExecuted"] + candidate["futureActionsExecuted"],
        "providerCalls": baseline["providerCallsAfterReplay"] + candidate["providerCallsAfterReplay"],
        "providerCallsDuringReplay": 0,
        "questionFreezeSha256": freeze["freezeSha256"],
        "comparisonSha256": sha256_file(output / "comparison.json"),
        "armSummarySha256": {
            "baseline": sha256_file(output / "baseline" / "summary.json"),
            "cycle-aware-v1": sha256_file(output / "cycle-aware-v1" / "summary.json"),
        },
        "developmentOnly": True,
        "notLiveMigration": True,
        "counterevidence": comparison["counterevidence"],
    }
    write_json_atomic(output / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
