"""Run bounded paired old/new Jev questions on immutable saved states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from candidate import (
    BASELINE_INSTRUCTIONS,
    CANDIDATE_INSTRUCTIONS,
    augment_state,
    sha256_json,
)


DIRECTION_DELTAS = {
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Retain typed outputs and digests, excluding base64 provider bodies."""
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
            "attempts_total",
            "calls_used",
            "input_bytes_used",
        )
        if key in decision
    }


def direction_for_choice(choice: str, criteria: Mapping[str, Any]) -> str | None:
    label = str(criteria.get(choice) or "")
    marker = "move/attack one step "
    if marker not in label:
        return None
    value = label.split(marker, 1)[1].split(" ", 1)[0]
    return value if value in DIRECTION_DELTAS else None


def last_position_pair(state: Mapping[str, Any]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    recent = state.get("recent_actions")
    if not isinstance(recent, list) or not recent:
        return None
    last = recent[-1]
    if not isinstance(last, Mapping):
        return None
    before = last.get("before")
    after = last.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    values = before.get("x"), before.get("y"), after.get("x"), after.get("y")
    if not all(isinstance(value, int) for value in values):
        return None
    return (int(before["x"]), int(before["y"])), (int(after["x"]), int(after["y"]))


def immediate_reversal_choice(
    choice: str, state: Mapping[str, Any], criteria: Mapping[str, Any]
) -> bool:
    pair = last_position_pair(state)
    direction = direction_for_choice(choice, criteria)
    if pair is None or direction is None:
        return False
    before, after = pair
    dx, dy = DIRECTION_DELTAS[direction]
    return (after[0] + dx, after[1] + dy) == before


def reversal_probability_mass(
    probabilities: Mapping[str, Any], state: Mapping[str, Any], criteria: Mapping[str, Any]
) -> float:
    return math.fsum(
        float(probability)
        for choice, probability in probabilities.items()
        if immediate_reversal_choice(choice, state, criteria)
    )


def prompt_handler(choice: str, criteria: Mapping[str, Any]) -> bool:
    label = str(criteria.get(choice) or "").lower()
    return "current inventory prompt" in label or "continue past --more--" in label


def total_variation(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return 0.5 * math.fsum(abs(float(left[key]) - float(right[key])) for key in left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default=Path("fixtures.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("evaluation-results.jsonl"), type=Path)
    parser.add_argument("--summary", default=Path("evaluation-summary.json"), type=Path)
    parser.add_argument("--until-win-source", required=True, type=Path)
    parser.add_argument("--max-calls", default=64, type=int)
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 64:
        raise SystemExit("max-calls must be between 1 and 64")
    fixtures = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
    needed_calls = 2 * len(fixtures)
    if needed_calls > args.max_calls:
        raise SystemExit(f"paired evaluation needs {needed_calls} calls, cap is {args.max_calls}")
    source_root = args.until_win_source.expanduser().resolve()
    sys.path.insert(0, str(source_root))
    from jev_client import JevClient  # type: ignore

    client = JevClient(
        max_calls=args.max_calls,
        max_input_bytes=args.max_calls * 24 * 1024,
        timeout=20,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    with args.output.open("x", encoding="utf-8") as stream:
        for fixture in fixtures:
            state = fixture["state"]
            criteria = fixture["criteria"]
            if len(criteria) != 121:
                raise SystemExit(f"{fixture['fixtureId']} does not have 121 actions")
            candidate_state = augment_state(state)
            baseline = sanitize_decision(
                client.choose(state, criteria, BASELINE_INSTRUCTIONS)
            )
            candidate = sanitize_decision(
                client.choose(candidate_state, criteria, CANDIDATE_INSTRUCTIONS)
            )
            baseline_probabilities = baseline["probabilities"]
            candidate_probabilities = candidate["probabilities"]
            result = {
                "schema": "jev-nethack-cycle-aware-evaluation/v1",
                "fixtureId": fixture["fixtureId"],
                "category": fixture["category"],
                "episodeId": fixture["episodeId"],
                "seed": fixture["seed"],
                "step": fixture["step"],
                "stateSha256": fixture["stateSha256"],
                "criteriaSha256": fixture["criteriaSha256"],
                "criteriaCount": len(criteria),
                "criteriaPreserved": criteria == fixture["criteria"],
                "baselineInstructionSha256": sha256_json(BASELINE_INSTRUCTIONS),
                "candidateInstructionSha256": sha256_json(CANDIDATE_INSTRUCTIONS),
                "candidateStateSha256": sha256_json(candidate_state),
                "cycleContext": candidate_state["cycle_context"],
                "baseline": baseline,
                "candidate": candidate,
                "effects": {
                    "choiceChanged": baseline["choice"] != candidate["choice"],
                    "baselineImmediateReversal": immediate_reversal_choice(
                        baseline["choice"], state, criteria
                    ),
                    "candidateImmediateReversal": immediate_reversal_choice(
                        candidate["choice"], state, criteria
                    ),
                    "baselineReversalProbabilityMass": reversal_probability_mass(
                        baseline_probabilities, state, criteria
                    ),
                    "candidateReversalProbabilityMass": reversal_probability_mass(
                        candidate_probabilities, state, criteria
                    ),
                    "baselinePromptHandler": prompt_handler(baseline["choice"], criteria),
                    "candidatePromptHandler": prompt_handler(candidate["choice"], criteria),
                    "probabilityTotalVariation": total_variation(
                        baseline_probabilities, candidate_probabilities
                    ),
                },
            }
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        category_rows[result["category"]].append(result)
    categories: dict[str, Any] = {}
    for category, rows in sorted(category_rows.items()):
        effects = [row["effects"] for row in rows]
        categories[category] = {
            "fixtures": len(rows),
            "choiceChanges": sum(effect["choiceChanged"] for effect in effects),
            "baselineImmediateReversalSelections": sum(
                effect["baselineImmediateReversal"] for effect in effects
            ),
            "candidateImmediateReversalSelections": sum(
                effect["candidateImmediateReversal"] for effect in effects
            ),
            "meanBaselineReversalProbabilityMass": math.fsum(
                effect["baselineReversalProbabilityMass"] for effect in effects
            ) / len(effects),
            "meanCandidateReversalProbabilityMass": math.fsum(
                effect["candidateReversalProbabilityMass"] for effect in effects
            ) / len(effects),
            "baselinePromptHandlerSelections": sum(
                effect["baselinePromptHandler"] for effect in effects
            ),
            "candidatePromptHandlerSelections": sum(
                effect["candidatePromptHandler"] for effect in effects
            ),
            "meanProbabilityTotalVariation": math.fsum(
                effect["probabilityTotalVariation"] for effect in effects
            ) / len(effects),
            "baselineChoices": dict(Counter(row["baseline"]["choice"] for row in rows)),
            "candidateChoices": dict(Counter(row["candidate"]["choice"] for row in rows)),
        }

    summary = {
        "schema": "jev-nethack-cycle-aware-evaluation-summary/v1",
        "completed": True,
        "startedAt": started_at,
        "endedAt": datetime.now(timezone.utc).isoformat(),
        "model": results[0]["candidate"]["model"] if results else None,
        "fixtureCount": len(fixtures),
        "attemptedApiCalls": client.calls_used,
        "maxApiCalls": args.max_calls,
        "inputBytesReserved": client.input_bytes_used,
        "criteriaCountPerFixture": 121,
        "allCriteriaPreserved": all(result["criteriaPreserved"] for result in results),
        "rawProbabilityMapsRetained": all(
            len(result["baseline"]["probabilities"]) == 121
            and len(result["candidate"]["probabilities"]) == 121
            for result in results
        ),
        "categories": categories,
        "limitations": [
            "Saved-state paired choices measure immediate selection/distribution effects, not gameplay outcomes.",
            "Confidence is distribution concentration and is not treated as accuracy.",
            "One baseline and one candidate call per fixture do not estimate variance.",
            "No candidate action was executed in NetHack.",
        ],
        "artifacts": {},
        "sourceSha256": {
            "candidate.py": sha256_file(Path(__file__).with_name("candidate.py")),
            "fixtures.jsonl": sha256_file(args.fixtures),
            "jev_client.py": sha256_file(source_root / "jev_client.py"),
        },
        "docsConsulted": [
            "https://docs.typesafe.ai/concepts/state.md",
            "https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md",
            "https://docs.typesafe.ai/primitives/advanced.md",
            "https://docs.typesafe.ai/primitives/choice.md",
            "https://docs.typesafe.ai/confidence.md",
        ],
    }
    summary["artifacts"][args.output.name] = {
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "records": len(results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"][args.summary.name] = {
        "bytes": args.summary.stat().st_size,
        "sha256BeforeArtifactSelfEntry": sha256_file(args.summary),
    }
    # Keep the summary non-self-referential; the receipt below hashes it.
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "jev-nethack-cycle-aware-evaluation-receipt/v1",
        "completed": True,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "results": {
            "filename": args.output.name,
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "summary": {
            "filename": args.summary.name,
            "bytes": args.summary.stat().st_size,
            "sha256": sha256_file(args.summary),
        },
        "attemptedApiCalls": client.calls_used,
        "providerBodiesPersisted": False,
        "credentialsPersisted": False,
    }
    Path("evaluation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
