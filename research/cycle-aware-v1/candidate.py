"""Pure candidate transformation for cycle-aware Jev action selection.

This module never executes an action and never removes or reweights criteria.
It only adds compact observed cycle evidence to state and clarifies the Choice
instructions.  Jev remains the action chooser over the original full mapping.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA = "jev-nethack-cycle-aware-question/v1"

BASELINE_INSTRUCTIONS = (
    "Choose exactly one next keyboard action to play NetHack 3.6.7. "
    "Your objective is to survive, explore the dungeon, obtain the Amulet of Yendor, "
    "and ascend. The state contains only player observations and past actions. "
    "Use the named adjacent cells for immediate movement; directions are relative to the player. "
    "A movement into a monster attacks it. Avoid repeatedly walking into a wall or repeating "
    "an ineffective action. Handle any visible menu or question before normal movement. "
    "In an inventory prompt, the same keyboard letter selects that inventory item. "
    "Use the visible terminal text to distinguish movement from menu selection. "
    "Inspect HP, hunger, pets, and dangerous terrain when deciding."
)

CYCLE_GUIDANCE = (
    " The state also contains cycle_context, a deterministic summary of recent observed "
    "positions, reversals, and visit counts. When cycle_context.detected is true and no "
    "immediate exception applies, prefer an action that breaks the demonstrated cycle and "
    "advances survival, exploration, or another concrete game objective. Before choosing a "
    "cycle-breaking movement, inspect adjacent_cells: prefer a plausibly traversable, lower-visit "
    "cell or a relevant non-movement action, and do not choose a blank, wall, solid-stone, or "
    "outside-map direction merely to be different. Do not mechanically "
    "forbid reversal or repetition: it remains valid when needed for combat, immediate survival, "
    "retreat from danger, a visible menu/question/--More-- prompt, or when it is the only safe "
    "or legal move. Choose exactly one action from the complete criteria mapping."
)

CANDIDATE_INSTRUCTIONS = BASELINE_INSTRUCTIONS + CYCLE_GUIDANCE

PROMPT_PATTERN = re.compile(
    r"(--More--|What do you want|In what direction|Pick up what|Really .*\?|\[[ynq]+\])",
    re.IGNORECASE,
)
COMBAT_PATTERN = re.compile(
    r"(bites?|hits?|miss(?:es)?|attacks?|thrusts?|kicks?|stings?|grabs?|engulfs?|you kill)",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _position(stats: Mapping[str, Any] | None) -> list[int] | None:
    if not isinstance(stats, Mapping):
        return None
    keys = ("dungeon", "level", "x", "y")
    if not all(isinstance(stats.get(key), int) for key in keys):
        return None
    return [int(stats[key]) for key in keys]


def build_cycle_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize only facts already present in a compact NetHack state."""
    recent = state.get("recent_actions")
    recent_actions = list(recent) if isinstance(recent, list) else []
    positions: list[list[int]] = []
    labels: list[str] = []
    for action in recent_actions:
        if not isinstance(action, Mapping):
            continue
        before = _position(action.get("before"))
        after = _position(action.get("after"))
        if before is not None and not positions:
            positions.append(before)
        if after is not None:
            positions.append(after)
        if isinstance(action.get("action"), str):
            labels.append(action["action"])

    reversals = sum(
        positions[index] == positions[index - 2]
        and positions[index] != positions[index - 1]
        for index in range(2, len(positions))
    )
    immediate_reversal = (
        len(positions) >= 3
        and positions[-1] == positions[-3]
        and positions[-1] != positions[-2]
    )

    adjacent = state.get("adjacent_cells")
    visits: dict[str, int] = {}
    if isinstance(adjacent, Mapping):
        for direction, cell in adjacent.items():
            if isinstance(direction, str) and isinstance(cell, Mapping):
                value = cell.get("visits")
                if isinstance(value, int) and value >= 0:
                    visits[direction] = value

    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    hp = player.get("hp")
    max_hp = player.get("max_hp")
    hunger = player.get("hunger_code")
    condition_bits = player.get("condition_bits")
    low_hp = (
        isinstance(hp, int)
        and isinstance(max_hp, int)
        and max_hp > 0
        and hp * 3 <= max_hp
    )
    urgent_hunger = isinstance(hunger, int) and hunger >= 3
    impaired = isinstance(condition_bits, int) and condition_bits != 0
    visible_text = f"{state.get('message') or ''}\n{state.get('terminal') or ''}"
    prompt_visible = bool(PROMPT_PATTERN.search(visible_text))
    recent_combat = bool(COMBAT_PATTERN.search(str(state.get("message") or "")))

    return {
        "schema": SCHEMA,
        "detected": reversals >= 2,
        "recentPositionPath": positions,
        "recentUniquePositions": len({tuple(position) for position in positions}),
        "twoStepPositionReversals": reversals,
        "immediateReversal": immediate_reversal,
        "recentActionLabels": labels,
        "adjacentVisitCounts": visits,
        "exceptionSignals": {
            "visiblePromptOrMenu": prompt_visible,
            "recentCombat": recent_combat,
            "lowHp": low_hp,
            "urgentHunger": urgent_hunger,
            "conditionImpairment": impaired,
        },
        "interpretation": (
            "Cycle evidence is advisory context for Jev. It never filters actions or forces a move."
        ),
    }


def augment_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    result = deepcopy(dict(state))
    result["cycle_context"] = build_cycle_context(state)
    return result


def prepare_candidate(
    state: Mapping[str, Any], criteria: Mapping[str, str | None]
) -> tuple[dict[str, Any], dict[str, str | None], str]:
    """Return candidate inputs without changing criteria identity or order."""
    if not isinstance(criteria, Mapping) or not criteria:
        raise TypeError("criteria must be a non-empty mapping")
    unchanged_criteria = dict(criteria)
    return augment_state(state), unchanged_criteria, CANDIDATE_INSTRUCTIONS
