"""Observation-only persistent frontier context and hierarchical Jev questions.

Code computes graph facts from observations. Jev chooses the bounded goal and
every actual action. Nothing here executes, filters, forces, or reweights an
action candidate.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA = "jev-nethack-frontier-v2/v1"
GOAL_HORIZON_ACTIONS = 6
DIRECTIONS = {
    "N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
    "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1),
}
PROMPT_RE = re.compile(
    r"(--More--|What do you want|In what direction|Pick up what|Really .*\?|\[[ynq]+\])",
    re.IGNORECASE,
)
COMBAT_RE = re.compile(
    r"(bites?|hits?|miss(?:es)?|attacks?|thrusts?|kicks?|stings?|grabs?|engulfs?|you kill)",
    re.IGNORECASE,
)
BLOCKED_RE = re.compile(r"(wall|solid stone|outside map)", re.IGNORECASE)

GOAL_CRITERIA = {
    "resolve_visible_prompt": "Complete the currently visible menu, question, direction, inventory, or --More-- interaction.",
    "protect_immediate_survival": "Address low HP, dangerous conditions, entrapment, or another immediate survival threat.",
    "handle_immediate_combat": "Fight, retreat, reposition, or otherwise handle a currently observed combat threat.",
    "address_urgent_hunger": "Safely address weak, fainting, fainted, or starving hunger state.",
    "follow_observed_frontier_route": "Use the observation-derived known route facts to approach a plausibly traversable observed but unvisited frontier.",
    "investigate_observed_local_feature": "Use an appropriate action to examine or interact with a visible feature, item, door, stair, or corridor clue.",
    "consolidate_or_replan": "Take a bounded information-gathering or repositioning step when no current frontier route is suitable.",
}

GOAL_INSTRUCTIONS = (
    "Choose one short-term NetHack goal for at most six actual actions. Use only the current observation and "
    "observation_memory facts; never infer unseen map cells. A visible prompt or menu must be resolved first. "
    "Immediate survival, active combat, and urgent hunger take priority over exploration when their observed "
    "signals apply. Otherwise prefer a concrete observed frontier or local feature over oscillation. Route facts "
    "are advisory observations, not commands, and may be abandoned when the state changes."
)

ACTION_INSTRUCTIONS = (
    "Choose exactly one next keyboard action to play NetHack 3.6.7. Jev selected the bounded active_goal, but "
    "must still judge this action from the current observation. Resolve visible prompts before normal play; "
    "prioritize immediate survival, combat, and urgent hunger when observed. For exploration, use only the "
    "observation-derived frontier and route facts and avoid returning to an oscillating edge without a concrete "
    "reason. A route hint is advisory and can be reversed or abandoned for safety, combat, prompts, changing "
    "terrain, or a better observed opportunity. Every one of the complete 121 action candidates remains legal."
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def empty_memory() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "observations": 0,
        "nodes": {},
        "transitions": {},
        "recentPositions": [],
        "lastPosition": None,
    }


def _position(state: Mapping[str, Any]) -> tuple[int, int, int, int]:
    player = state.get("player")
    if not isinstance(player, Mapping):
        raise ValueError("state.player is required")
    values = tuple(player.get(key) for key in ("dungeon", "level", "x", "y"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError("state.player position is invalid")
    return values  # type: ignore[return-value]


def _key(position: tuple[int, int, int, int]) -> str:
    return ":".join(str(part) for part in position)


def _parse_key(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split(":"))
    if len(parts) != 4:
        raise ValueError("position key is invalid")
    return parts  # type: ignore[return-value]


def _target(position: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
    dx, dy = DIRECTIONS[direction]
    return position[0], position[1], position[2] + dx, position[3] + dy


def _direction(source: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> str | None:
    if source[:2] != target[:2]:
        return None
    delta = target[2] - source[2], target[3] - source[3]
    return next((name for name, pair in DIRECTIONS.items() if pair == delta), None)


def _plausibly_traversable(cell: Mapping[str, Any]) -> bool:
    symbol = cell.get("symbol")
    description = str(cell.get("description") or "")
    return isinstance(symbol, str) and len(symbol) == 1 and symbol != " " and not BLOCKED_RE.search(description)


def observe(
    memory: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    raw_observation_digest: str | None = None,
) -> dict[str, Any]:
    """Update memory from a compact view derived from one captured raw observation.

    The trial integration supplies ``run.make_state(raw_observation, ...)`` and
    the already computed full-array digest. The digest is provenance only; this
    function never loads a save file, seed-dependent oracle, or unseen map data.
    """
    result = deepcopy(dict(memory)) if memory else empty_memory()
    if result.get("schema") != SCHEMA:
        raise ValueError("memory schema is unsupported")
    result.setdefault("nodes", {})
    result.setdefault("transitions", {})
    step = int(result.get("observations", 0))
    current = _position(state)
    current_key = _key(current)
    nodes = result["nodes"]
    node = nodes.setdefault(
        current_key,
        {"position": list(current), "visits": 0, "firstObservedAt": step, "adjacent": {}},
    )
    node["visits"] = int(node.get("visits", 0)) + 1
    node["lastObservedAt"] = step
    if raw_observation_digest is not None:
        if not isinstance(raw_observation_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_observation_digest):
            raise ValueError("raw observation digest must be a SHA-256 hex string")
        node["lastRawObservationDigest"] = raw_observation_digest
    adjacent = state.get("adjacent_cells")
    if isinstance(adjacent, Mapping):
        for direction in DIRECTIONS:
            cell = adjacent.get(direction)
            if not isinstance(cell, Mapping):
                continue
            target = _target(current, direction)
            node["adjacent"][direction] = {
                "target": list(target),
                "symbol": cell.get("symbol"),
                "description": cell.get("description"),
                "plausiblyTraversable": _plausibly_traversable(cell),
                "observedAt": step,
            }
    previous_key = result.get("lastPosition")
    if isinstance(previous_key, str) and previous_key != current_key:
        previous = _parse_key(previous_key)
        direction = _direction(previous, current)
        if direction is not None:
            edge_key = f"{previous_key}>{current_key}"
            edge = result["transitions"].setdefault(
                edge_key,
                {"from": list(previous), "to": list(current), "direction": direction, "traversals": 0},
            )
            edge["traversals"] = int(edge.get("traversals", 0)) + 1
            edge["lastTraversedAt"] = step
    recent = list(result.get("recentPositions") or [])
    recent.append(current_key)
    result["recentPositions"] = recent[-24:]
    result["lastPosition"] = current_key
    result["observations"] = step + 1
    return result


def _neighbors(memory: Mapping[str, Any], node_key: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for edge in (memory.get("transitions") or {}).values():
        if not isinstance(edge, Mapping):
            continue
        source, target = _key(tuple(edge["from"])), _key(tuple(edge["to"]))
        if source == node_key:
            result.append((target, str(edge["direction"])))
        elif target == node_key:
            reverse = _direction(_parse_key(node_key), _parse_key(source))
            if reverse:
                result.append((source, reverse))
    return sorted(set(result))


def _paths(memory: Mapping[str, Any], start: str) -> dict[str, tuple[list[str], list[str]]]:
    found = {start: ([start], [])}
    queue = deque([start])
    while queue:
        source = queue.popleft()
        positions, directions = found[source]
        for target, direction in _neighbors(memory, source):
            if target not in found:
                found[target] = (positions + [target], directions + [direction])
                queue.append(target)
    return found


def frontier_context(memory: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    current = _key(_position(state))
    nodes = memory.get("nodes") or {}
    paths = _paths(memory, current)
    candidates: list[dict[str, Any]] = []
    for source_key, node in nodes.items():
        if source_key not in paths or not isinstance(node, Mapping):
            continue
        for direction, cell in (node.get("adjacent") or {}).items():
            if not isinstance(cell, Mapping) or not cell.get("plausiblyTraversable"):
                continue
            target_key = _key(tuple(cell["target"]))
            if target_key in nodes:
                continue
            positions, route_directions = paths[source_key]
            candidates.append(
                {
                    "target": cell["target"],
                    "frontierDirection": direction,
                    "frontierSymbol": cell.get("symbol"),
                    "frontierDescription": cell.get("description"),
                    "routePositions": [list(_parse_key(item)) for item in positions],
                    "routeDirections": route_directions,
                    "knownRouteLength": len(route_directions),
                    "nextKnownDirection": route_directions[0] if route_directions else direction,
                    "targetPreviouslyVisited": False,
                }
            )
    candidates.sort(key=lambda item: (item["knownRouteLength"], item["target"], item["frontierDirection"]))
    recent = list(memory.get("recentPositions") or [])
    reversals = sum(
        recent[index] == recent[index - 2] and recent[index] != recent[index - 1]
        for index in range(2, len(recent))
    )
    traversals = sorted(
        (
            {"from": edge["from"], "to": edge["to"], "direction": edge["direction"], "traversals": edge["traversals"]}
            for edge in (memory.get("transitions") or {}).values()
            if isinstance(edge, Mapping)
        ),
        key=lambda edge: (-edge["traversals"], edge["from"], edge["to"]),
    )
    return {
        "schema": SCHEMA,
        "basis": "only previously supplied player observations and observed transitions",
        "currentPosition": list(_position(state)),
        "observationsRetained": int(memory.get("observations", 0)),
        "observedNodeCount": len(nodes),
        "observedTransitionCount": len(memory.get("transitions") or {}),
        "currentNodeVisits": int((nodes.get(current) or {}).get("visits", 0)),
        "recentPositionPath": [list(_parse_key(item)) for item in recent],
        "recentTwoStepReversals": reversals,
        "mostTraversedObservedEdges": traversals[:8],
        "frontierCandidates": candidates[:8],
        "limits": [
            "Unobserved cells are absent rather than guessed.",
            "Traversability and routes are historical observations, not guarantees of current safety.",
            "Route facts never execute or restrict an action.",
        ],
    }


def signals(state: Mapping[str, Any]) -> dict[str, bool]:
    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    hp, maximum = player.get("hp"), player.get("max_hp")
    visible = f"{state.get('message') or ''}\n{state.get('terminal') or ''}"
    return {
        "visiblePrompt": bool(PROMPT_RE.search(visible)),
        "recentCombat": bool(COMBAT_RE.search(str(state.get("message") or ""))),
        "lowHp": isinstance(hp, int) and isinstance(maximum, int) and maximum > 0 and hp * 3 <= maximum,
        "urgentHunger": isinstance(player.get("hunger_code"), int) and player["hunger_code"] >= 3,
        "conditionImpairment": isinstance(player.get("condition_bits"), int) and player["condition_bits"] != 0,
    }


def prepare_goal_question(state: Mapping[str, Any], memory: Mapping[str, Any], active_goal: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str], str]:
    augmented = deepcopy(dict(state))
    augmented["observation_memory"] = frontier_context(memory, state)
    augmented["priority_signals"] = signals(state)
    augmented["previous_goal"] = deepcopy(dict(active_goal)) if active_goal else None
    return augmented, dict(GOAL_CRITERIA), GOAL_INSTRUCTIONS


def new_goal_contract(decision: Mapping[str, Any], state: Mapping[str, Any], memory: Mapping[str, Any], step: int) -> dict[str, Any]:
    choice = decision.get("choice")
    if choice not in GOAL_CRITERIA:
        raise ValueError("goal decision is outside the goal criteria")
    return {
        "schema": SCHEMA,
        "choice": choice,
        "description": GOAL_CRITERIA[choice],
        "startedAtAction": step,
        "expiresBeforeAction": step + GOAL_HORIZON_ACTIONS,
        "originObservationDigest": sha256_json(state),
        "originMemoryDigest": sha256_json(memory),
        "rawGoalDecision": deepcopy(dict(decision)),
    }


def goal_refresh_reasons(state: Mapping[str, Any], memory: Mapping[str, Any], active_goal: Mapping[str, Any] | None, step: int) -> list[str]:
    if not active_goal:
        return ["no_active_goal"]
    reasons: list[str] = []
    observed = signals(state)
    selected = active_goal.get("choice")
    if step >= int(active_goal.get("expiresBeforeAction", step)):
        reasons.append("goal_horizon_exhausted")
    priorities = (
        ("visiblePrompt", "resolve_visible_prompt", "visible_prompt_changed_priority"),
        ("lowHp", "protect_immediate_survival", "low_hp_changed_priority"),
        ("recentCombat", "handle_immediate_combat", "combat_changed_priority"),
        ("urgentHunger", "address_urgent_hunger", "urgent_hunger_changed_priority"),
    )
    for signal, required, reason in priorities:
        if observed[signal] and selected != required:
            reasons.append(reason)
    if selected == "follow_observed_frontier_route" and not frontier_context(memory, state)["frontierCandidates"]:
        reasons.append("observed_frontier_exhausted")
    return reasons


def prepare_action_question(state: Mapping[str, Any], memory: Mapping[str, Any], active_goal: Mapping[str, Any], criteria: Mapping[str, str | None]) -> tuple[dict[str, Any], dict[str, str | None], str]:
    if not isinstance(criteria, Mapping) or not criteria:
        raise TypeError("criteria must be a non-empty mapping")
    augmented = deepcopy(dict(state))
    augmented["observation_memory"] = frontier_context(memory, state)
    augmented["priority_signals"] = signals(state)
    augmented["active_goal"] = deepcopy(dict(active_goal))
    return augmented, dict(criteria), ACTION_INSTRUCTIONS


def retention_record(goal_decision: Mapping[str, Any], action_decision: Mapping[str, Any]) -> dict[str, Any]:
    """Make the future evidence obligation explicit without interpreting confidence."""
    return {
        "schema": SCHEMA,
        "goalDecision": deepcopy(dict(goal_decision)),
        "actionDecision": deepcopy(dict(action_decision)),
        "confidenceSemantics": "distribution concentration only; not accuracy or permission",
    }
