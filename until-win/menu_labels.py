"""Contextual, lossless labels for the complete NetHack action menu.

The action space remains unchanged.  This adapter only rewrites descriptions
for controls that are visibly meaningful in the current prompt, so callers can
pass the returned mapping directly to Jev without masking any action IDs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_PRESS_KEY_RE = re.compile(r"Press\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
_INVENTORY_PROMPT_RE = re.compile(
    r"\[\s*([A-Za-z](?:\s*-\s*[A-Za-z])?|[A-Za-z]+)\s+or\s+\?\*\s*\]",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_READ_PROMPT_RE = re.compile(r"\bread\b", re.IGNORECASE)


def contextual_choices(
    state: Mapping[str, Any],
    actions: Sequence[Any] | Mapping[Any, Any],
    base_criteria: Mapping[str, str | None],
) -> dict[str, str | None]:
    """Return the same complete action mapping with visible prompt labels.

    ``state`` is the observation mapping produced by ``run.make_state``;
    ``actions`` is the corresponding NLE action sequence (or an index mapping),
    and ``base_criteria`` is the normal 121-action Jev criteria mapping.  The
    function never masks, adds, or removes actions and never mutates its input.
    """

    result = dict(base_criteria)
    if not isinstance(state, Mapping) or not isinstance(base_criteria, Mapping):
        return result

    visible = _visible_text(state)
    if not visible:
        return result

    keys = _action_keys(actions, result)
    inventory = _inventory_items(state.get("inventory"))
    prompt_letters = _inventory_prompt_letters(visible)
    yes_no = _yes_no_prompt(visible)
    more = "--more--" in visible.lower()
    prompt_context = bool(prompt_letters or yes_no or more)
    inventory_context = "current read prompt" if _READ_PROMPT_RE.search(visible) else "current inventory prompt"

    for action_id, description in result.items():
        key = keys.get(action_id)
        if key is None:
            continue

        if prompt_letters and not yes_no and key in prompt_letters and key in inventory:
            result[action_id] = (
                f"Press {key} to select [{inventory[key]}] for {inventory_context}"
            )
            continue

        if yes_no and key in {"y", "n"}:
            answer = "yes" if key == "y" else "no"
            result[action_id] = f"Press {key} to answer {answer} to the current prompt"
            continue

        if prompt_context and key in {"ESC", "Escape", "ASCII 27"}:
            result[action_id] = "Press Escape to cancel the current prompt"
            continue

        if more and key in {"ENTER", "Return", "ASCII 10", "ASCII 13"}:
            result[action_id] = "Press Enter to continue past --More--"
            continue

        if more and key in {" ", "SPACE"}:
            result[action_id] = "Press Space to continue past --More--"

    return result


def _visible_text(state: Mapping[str, Any]) -> str:
    """Select the visible field containing an actual prompt marker."""

    message = state.get("message")
    terminal = state.get("terminal")
    message_text = message if isinstance(message, str) else ""
    terminal_text = terminal if isinstance(terminal, str) else ""
    if _has_prompt_marker(message_text):
        return message_text
    if _has_prompt_marker(terminal_text):
        return terminal_text
    return message_text or terminal_text


def _has_prompt_marker(text: str) -> bool:
    return (
        bool(_INVENTORY_PROMPT_RE.search(text))
        or _yes_no_prompt(text)
        or "--more--" in text.lower()
    )


def _inventory_items(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    items: dict[str, str] = {}
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        letter = entry.get("letter")
        item = entry.get("item")
        if isinstance(letter, str) and len(letter) == 1 and isinstance(item, str) and item.strip():
            items[letter] = item.strip()
    return items


def _inventory_prompt_letters(text: str) -> set[str]:
    match = _INVENTORY_PROMPT_RE.search(text)
    if not match:
        return set()
    body = re.sub(r"\s+", "", match.group(1))
    if len(body) == 3 and body[1] == "-":
        start, end = body[0], body[2]
        if start <= end:
            return {chr(code) for code in range(ord(start), ord(end) + 1)}
        return set()
    return set(body)


def _yes_no_prompt(text: str) -> bool:
    """Recognize compact explicit forms such as ``[yn]`` or ``[y/n]``."""

    for match in _BRACKET_RE.finditer(text):
        body = re.sub(r"\s+", "", match.group(1)).lower()
        if "y" not in body or "n" not in body:
            continue
        if body.endswith("or?*"):
            body = body[: -len("or?*")]
        if re.fullmatch(r"[ynq?!*/|,]+", body) or re.fullmatch(
            r"y(?:es)?(?:/|or)n(?:o)?", body
        ):
            return True
    return False


def _action_keys(
    actions: Sequence[Any] | Mapping[Any, Any],
    criteria: Mapping[str, str | None],
) -> dict[str, str]:
    values: list[Any]
    if isinstance(actions, Mapping):
        values = [actions.get(action_id) for action_id in criteria]
    elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
        values = list(actions)
    else:
        values = []

    keys: dict[str, str] = {}
    for index, (action_id, description) in enumerate(criteria.items()):
        key = _key_from_action(values[index]) if index < len(values) else None
        if key is None and isinstance(description, str):
            match = _PRESS_KEY_RE.search(description)
            if match:
                key = match.group(1)
        if key is not None:
            keys[action_id] = key
    return keys


def _key_from_action(action: Any) -> str | None:
    try:
        code = int(action)
    except (TypeError, ValueError):
        return None
    if code == 27:
        return "ESC"
    if code in {10, 13}:
        return "ASCII " + str(code)
    if code == 32:
        return " "
    if 32 <= code <= 126:
        return chr(code)
    return "ASCII " + str(code)


__all__ = ["contextual_choices"]
