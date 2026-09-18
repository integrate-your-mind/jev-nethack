"""Deterministic replay plans for native and legacy Jev NetHack sessions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from transition_pack import TransitionPackError, validate_shard


RECOVERY_SCHEMA = "jev-nethack-replay-plan/v1"


class ReplayPlanError(Exception):
    """Saved evidence is inconsistent, incomplete, or unsafe to replay."""


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except ValueError as exc:
            # A hard crash can leave only the final append incomplete. It is
            # excluded from replay and remains preserved in the original file.
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ReplayPlanError(f"invalid JSONL record in {path}") from exc
        if not isinstance(decoded, dict):
            raise ReplayPlanError(f"non-object JSONL record in {path}")
        records.append(decoded)
    return records


def _same_transition(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "eventId",
        "episodeId",
        "seed",
        "step",
        "actionIndex",
        "keycode",
        "actionLabel",
        "state",
        "nextState",
        "decision",
        "criteria",
        "instructions",
        "observationDigest",
        "nextObservationDigest",
        "reward",
        "terminated",
        "truncated",
        "isAscended",
        "verifiedAscension",
        "endStatus",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def load_native_plan(data_root: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    episode_id = candidate.get("episodeId")
    seed = candidate.get("seed")
    if not isinstance(episode_id, int) or not isinstance(seed, int):
        raise ReplayPlanError("resume candidate has invalid episode or seed")
    fragments = candidate.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ReplayPlanError("resume candidate has no preserved fragment list")

    by_step: dict[int, dict[str, Any]] = {}
    intents: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for relative in fragments:
        if not isinstance(relative, str):
            raise ReplayPlanError("fragment path is invalid")
        fragment = (data_root / relative).resolve()
        try:
            fragment.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ReplayPlanError("fragment path escapes the data root") from exc
        pack_count = 0
        pack_records = 0
        for pack in sorted((fragment / "training").glob("*.npzpack")):
            pack = pack.resolve()
            try:
                pack.relative_to(data_root.resolve())
            except ValueError as exc:
                raise ReplayPlanError("transition pack path escapes the data root") from exc
            pack_count += 1
            try:
                validation = validate_shard(pack)
                pack_records += int(validation["records"])
                for frame in validation["frames"]:
                    metadata = dict(frame.metadata)
                    if metadata.get("episodeId") != episode_id:
                        continue
                    if metadata.get("seed") != seed:
                        raise ReplayPlanError("saved transition seed does not match resume candidate")
                    step = metadata.get("step")
                    if not isinstance(step, int) or step < 0:
                        raise ReplayPlanError("saved transition has an invalid step")
                    existing = by_step.get(step)
                    if existing is not None and not _same_transition(existing, metadata):
                        raise ReplayPlanError(f"conflicting saved transitions at step {step}")
                    by_step[step] = metadata
            except TransitionPackError as exc:
                raise ReplayPlanError(f"corrupt transition pack {pack.name}") from exc
        event_path = (fragment / "transitions.jsonl").resolve()
        try:
            event_path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ReplayPlanError("transition event path escapes the data root") from exc
        for event in _json_lines(event_path):
            if event.get("episodeId") != episode_id or event.get("eventType") != "action_intent":
                continue
            event_id = event.get("eventId")
            if not isinstance(event_id, str) or not event_id:
                raise ReplayPlanError("saved action intent has no event ID")
            if event.get("seed") != seed:
                raise ReplayPlanError("saved action intent seed does not match resume candidate")
            previous = intents.get(event_id)
            if previous is not None and previous != event:
                raise ReplayPlanError(f"conflicting action intents for {event_id}")
            intents[event_id] = event
        sources.append(
            {
                "fragment": relative,
                "packs": pack_count,
                "records": pack_records,
                "eventsSha256": hashlib.sha256(event_path.read_bytes()).hexdigest() if event_path.exists() else None,
            }
        )

    ordered_steps = sorted(by_step)
    records = [by_step[index] for index in ordered_steps]
    committed_ids = {record.get("eventId") for record in records}
    if len(committed_ids) != len(records) or None in committed_ids:
        raise ReplayPlanError("committed replay event IDs are missing or duplicated")
    pending = [event for event_id, event in intents.items() if event_id not in committed_ids]
    if len(pending) > 1:
        raise ReplayPlanError("multiple uncommitted action intents make exact replay ambiguous")
    pending_intent = pending[0] if pending else None
    if ordered_steps:
        first_step = ordered_steps[0]
    elif pending_intent is not None and isinstance(pending_intent.get("step"), int):
        first_step = int(pending_intent["step"])
    else:
        first_step = 0
    if ordered_steps != list(range(first_step, first_step + len(records))):
        raise ReplayPlanError("committed replay steps are not one contiguous range")
    if pending_intent is not None:
        if pending_intent.get("step") != first_step + len(records):
            raise ReplayPlanError("pending action intent is not the next contiguous step")
        if pending_intent.get("episodeId") != episode_id or pending_intent.get("seed") != seed:
            raise ReplayPlanError("pending action intent identity does not match resume candidate")
        decision = pending_intent.get("decision")
        if not isinstance(decision, dict) or decision.get("choice") is None:
            raise ReplayPlanError("pending action intent has no validated Jev decision")
    for record in records:
        if not isinstance(record.get("observationDigest"), str) or not isinstance(record.get("nextObservationDigest"), str):
            raise ReplayPlanError("native replay transition lacks full observation hashes")
    return {
        "schema": RECOVERY_SCHEMA,
        "origin": "native_full_observation",
        "episodeId": episode_id,
        "seed": seed,
        "firstStep": first_step,
        "records": records,
        "pendingIntent": pending_intent,
        "sources": sources,
    }


def load_legacy_plan(source: Path, episode: int) -> dict[str, Any]:
    """Build a replay plan from the old MP4/terminal broadcaster evidence.

    Historical raw arrays and exact HTTP bodies were not captured. The plan
    therefore verifies every saved compact state while replay regenerates and
    clearly labels the raw arrays as deterministic reconstructions.
    """
    config_path = source / "config.json"
    events_path = source / "events.jsonl"
    if not config_path.is_file() or not events_path.is_file():
        raise ReplayPlanError("legacy source must contain config.json and events.jsonl")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ReplayPlanError("legacy config is invalid")
    settings = config.get("settings")
    if not isinstance(settings, dict):
        raise ReplayPlanError("legacy config settings are missing")
    base_seed = settings.get("seed")
    character = settings.get("character")
    if not isinstance(base_seed, int) or not isinstance(character, str):
        raise ReplayPlanError("legacy seed or character is invalid")
    events = [event for event in _json_lines(events_path) if event.get("episode") == episode]
    before: dict[int, dict[str, Any]] = {}
    after: dict[int, dict[str, Any]] = {}
    decisions: dict[int, dict[str, Any]] = {}
    for event in events:
        step = event.get("step")
        if not isinstance(step, int) or step < 0:
            continue
        if event.get("event_type") == "observed_frame" and event.get("phase") == "before_action":
            target = before
        elif event.get("event_type") == "observed_frame" and event.get("phase") == "after_action":
            target = after
        elif event.get("event_type") == "decision":
            target = decisions
        else:
            continue
        existing = target.get(step)
        if existing is not None and existing != event:
            raise ReplayPlanError(f"legacy step {step} has conflicting duplicate evidence")
        target[step] = event
    action_steps = sorted(after)
    if action_steps != list(range(len(action_steps))):
        raise ReplayPlanError("legacy action sequence is not contiguous from zero")
    records: list[dict[str, Any]] = []
    for step in action_steps:
        before_event = before.get(step)
        after_event = after[step]
        if before_event is None:
            raise ReplayPlanError(f"legacy step {step} lacks its before observation")
        action = after_event.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("index"), int) or not isinstance(action.get("keycode"), int):
            raise ReplayPlanError(f"legacy step {step} has invalid action evidence")
        decision_event = decisions.get(step)
        decision = decision_event.get("decision") if decision_event else None
        if not isinstance(decision, dict):
            raise ReplayPlanError(f"legacy step {step} lacks its saved Jev decision")
        records.append(
            {
                "eventId": f"legacy-{config.get('stream_id')}-episode-{episode}-step-{step:09d}",
                "episodeId": episode,
                "seed": base_seed + episode,
                "step": step,
                "state": before_event.get("state"),
                "nextState": after_event.get("state"),
                "decision": decision,
                "actionIndex": action["index"],
                "keycode": action["keycode"],
                "actionLabel": action.get("label"),
                "reward": after_event.get("reward"),
                "terminated": bool(after_event.get("terminated", False)),
                "truncated": bool(after_event.get("truncated", False)),
                "isAscended": False,
                "verifiedAscension": False,
                "endStatus": None,
                "provenance": {
                    "origin": "deterministically_reconstructed_legacy",
                    "historicalRawArraysAvailable": False,
                    "historicalRequestResponseBodiesAvailable": False,
                    "compactBeforeSequence": before_event.get("sequence"),
                    "compactAfterSequence": after_event.get("sequence"),
                },
            }
        )
    final_before = before.get(len(records))
    if final_before is None or not isinstance(final_before.get("state"), dict):
        raise ReplayPlanError("legacy source lacks the final compact state to resume")
    return {
        "schema": RECOVERY_SCHEMA,
        "origin": "deterministically_reconstructed_legacy",
        "episodeId": episode,
        "seed": base_seed + episode,
        "character": character,
        "records": records,
        "pendingIntent": None,
        "expectedFinalState": final_before["state"],
        "sources": [
            {
                "source": str(source.resolve()),
                "streamId": config.get("stream_id"),
                "configSha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "eventsSha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                "actions": len(records),
                "compactObservations": len(records) * 2 + 1,
            }
        ],
    }
