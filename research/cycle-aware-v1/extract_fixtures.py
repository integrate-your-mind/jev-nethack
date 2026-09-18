"""Extract a bounded immutable fixture set from durable live-run events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from candidate import build_cycle_context, sha256_json


SELECTIONS = {
    "stagnant_cycle": [2911, 2930, 2948, 2960, 2980, 3029],
    "normal_play": [0, 550, 1650, 1700],
    "menu_control": [534, 1037, 2403],
    "survival_control": [240, 532, 533],
}


def sanitized_decision(value: Any) -> dict[str, Any]:
    decision = value if isinstance(value, dict) else {}
    return {
        key: decision.get(key)
        for key in (
            "choice",
            "confidence",
            "probabilities",
            "probability_sum",
            "model",
            "usage",
            "latency_seconds",
            "request_digest",
            "response_digest",
        )
        if key in decision
    }


def load_events(data_root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    by_step: dict[int, dict[str, Any]] = {}
    sources: dict[int, str] = {}
    for path in sorted((data_root / "fragments").glob("*/transitions.jsonl")):
        with path.open("rb") as stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                step = event.get("step")
                if event.get("eventType") == "transition_committed" and isinstance(step, int):
                    by_step[step] = event
                    sources[step] = str(path.relative_to(data_root))
    return by_step, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("fixtures.jsonl"), type=Path)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    events, sources = load_events(data_root)
    created_at = datetime.now(timezone.utc).isoformat()
    fixtures: list[dict[str, Any]] = []
    for category, steps in SELECTIONS.items():
        for step in steps:
            event = events.get(step)
            if event is None:
                raise SystemExit(f"missing committed transition at step {step}")
            state = event.get("state")
            criteria = event.get("criteria")
            if not isinstance(state, dict) or not isinstance(criteria, dict):
                raise SystemExit(f"step {step} lacks state or criteria")
            if len(criteria) != 121:
                raise SystemExit(f"step {step} has {len(criteria)} criteria, expected 121")
            fixture = {
                "schema": "jev-nethack-cycle-aware-fixture/v1",
                "fixtureId": f"{category}-step-{step}",
                "category": category,
                "episodeId": event.get("episodeId"),
                "seed": event.get("seed"),
                "step": step,
                "eventId": event.get("eventId"),
                "recordedAt": event.get("recordedAt"),
                "source": sources[step],
                "state": state,
                "criteria": criteria,
                "observedCycleContext": build_cycle_context(state),
                "historicalDecision": sanitized_decision(event.get("decision")),
            }
            fixture["stateSha256"] = sha256_json(state)
            fixture["criteriaSha256"] = sha256_json(criteria)
            fixture["eventEvidenceSha256"] = sha256_json(
                {
                    "eventId": event.get("eventId"),
                    "step": step,
                    "state": state,
                    "criteria": criteria,
                    "decision": fixture["historicalDecision"],
                }
            )
            fixtures.append(fixture)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for fixture in fixtures:
            stream.write(json.dumps(fixture, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "schema": "jev-nethack-cycle-aware-fixture-receipt/v1",
        "completed": True,
        "createdAt": created_at,
        "dataRoot": str(data_root),
        "fixtureFile": args.output.name,
        "fixtureCount": len(fixtures),
        "categories": {key: len(value) for key, value in SELECTIONS.items()},
        "fixtureIds": [fixture["fixtureId"] for fixture in fixtures],
        "fixtureFileSha256": __import__("hashlib").sha256(args.output.read_bytes()).hexdigest(),
        "note": "Compact saved states and full 121-option criteria; no credentials or raw provider bodies.",
    }
    receipt_path = args.output.with_name("fixture-receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
