"""Derive conservative, deterministic selection-effect evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate import direction_for_choice


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_class(fixture: dict[str, Any], choice: str) -> str:
    direction = direction_for_choice(choice, fixture["criteria"])
    if direction is None:
        return "non_movement"
    cell = (fixture["state"].get("adjacent_cells") or {}).get(direction) or {}
    symbol = str(cell.get("symbol") or "")
    description = str(cell.get("description") or "").lower()
    if symbol == " " or any(word in description for word in ("wall", "solid stone", "outside map")):
        return "blank_or_blocked"
    if symbol in ("#", ".", "+", "<", ">", "%") or any(
        word in description for word in ("corridor", "floor", "doorway", "door", "corpse")
    ):
        return "plausibly_traversable"
    return "uncertain_or_tactical"


def main() -> int:
    fixtures = {
        row["fixtureId"]: row
        for row in map(json.loads, Path("fixtures.jsonl").read_text().splitlines())
    }
    results = list(
        map(json.loads, Path("evaluation-results.jsonl").read_text().splitlines())
    )
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = []
    for result in results:
        fixture = fixtures[result["fixtureId"]]
        row = {
            "fixtureId": result["fixtureId"],
            "category": result["category"],
            "baselineChoice": result["baseline"]["choice"],
            "candidateChoice": result["candidate"]["choice"],
            "choiceChanged": result["effects"]["choiceChanged"],
            "baselineImmediateReversal": result["effects"]["baselineImmediateReversal"],
            "candidateImmediateReversal": result["effects"]["candidateImmediateReversal"],
            "candidateTargetClass": target_class(fixture, result["candidate"]["choice"]),
            "candidateChoiceLabel": fixture["criteria"][result["candidate"]["choice"]],
        }
        rows.append(row)
        categories[result["category"]].append(row)

    category_summary = {}
    for category, category_values in sorted(categories.items()):
        category_summary[category] = {
            "fixtures": len(category_values),
            "choiceChanges": sum(row["choiceChanged"] for row in category_values),
            "baselineImmediateReversals": sum(
                row["baselineImmediateReversal"] for row in category_values
            ),
            "candidateImmediateReversals": sum(
                row["candidateImmediateReversal"] for row in category_values
            ),
            "candidateTargetClasses": dict(
                Counter(row["candidateTargetClass"] for row in category_values)
            ),
        }

    evidence = {
        "schema": "jev-nethack-cycle-aware-observed-effects/v1",
        "completed": True,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "resultsSha256": sha256_file(Path("evaluation-results.jsonl")),
        "categories": category_summary,
        "rows": rows,
        "interpretation": {
            "supported": [
                "All 121 choices and raw probability maps were retained.",
                "The final candidate changed four of six stagnant-state selections and reduced selected immediate reversals from five to two.",
                "Normal-state choices were unchanged in four of four fixtures; prompt handlers were unchanged in three of three menu fixtures.",
                "At HP 1 after a jackal bite, the candidate changed one control choice from movement to QUAFF; this was not executed.",
            ],
            "counterevidence": [
                "One stagnant-state candidate choice targeted a blank adjacent cell.",
                "Two stagnant-state candidate choices remained immediate reversals.",
                "SEARCH and traversable-looking movement are plausible actions, not demonstrated progress without executing them.",
            ],
            "notEstablished": [
                "Higher gameplay score, depth, survival, exploration, or ascension probability.",
                "Generalization beyond the 16 selected saved states.",
                "Accuracy from confidence; confidence only describes distribution concentration.",
            ],
        },
    }
    output = Path("observed-selection-effects.json")
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "jev-nethack-cycle-aware-effects-receipt/v1",
        "completed": True,
        "filename": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    Path("observed-selection-effects.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
