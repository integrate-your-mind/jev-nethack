from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import branch_experiment as branch


class CompassDirection:
    name = "W"


def summary(*, digest: str = "same", reversals: int = 0, blank: int = 0, new: int = 1) -> dict:
    values = {
        "futureActionsExecuted": 64,
        "distinctPositions": 4,
        "newCellCountVsPrefix": new,
        "twoStepReversals": reversals,
        "exactRawNoOps": 0,
        "unchangedPositionActions": 2,
        "blankOrWallAttemptCount": blank,
        "promptStepCount": 0,
        "turnDelta": 20,
        "scoreDelta": 0,
        "depthDelta": 0,
        "experienceLevelDelta": 0,
        "hpDelta": 0,
        "hungerCodeEnd": 1,
        "totalReward": 0.0,
        "terminal": None,
        "replay": {
            "branchPointObservationDigest": digest,
            "branchPointStateSha256": digest,
            "providerCallsDuringReplay": 0,
        },
    }
    return values


class BranchMetricTests(unittest.TestCase):
    def test_blank_target_is_observed_without_forcing_action(self) -> None:
        state = {"adjacent_cells": {"W": {"symbol": " ", "description": "solid stone"}}}
        target = branch.movement_target(state, CompassDirection())
        self.assertEqual(target["direction"], "W")
        self.assertTrue(target["classifiedBlankOrWall"])

    def test_prompt_detection_uses_visible_nethack_text(self) -> None:
        self.assertTrue(branch.prompt_visible({"message": "Really attack the dog? [yn]", "terminal": ""}))
        self.assertFalse(branch.prompt_visible({"message": "You see here a rock.", "terminal": ""}))

    def test_comparison_requires_identical_raw_and_compact_branchpoint(self) -> None:
        with tempfile.TemporaryDirectory(dir=branch.CANDIDATE_ROOT) as temporary:
            with self.assertRaises(branch.BranchError):
                branch.compare(
                    Path(temporary),
                    summary(digest="left"),
                    summary(digest="right"),
                    {"freezeSha256": "freeze"},
                )

    def test_comparison_records_counterevidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=branch.CANDIDATE_ROOT) as temporary:
            output = Path(temporary)
            result = branch.compare(
                output,
                summary(reversals=0, blank=0, new=2),
                summary(reversals=3, blank=1, new=1),
                {"freezeSha256": "freeze"},
            )
            self.assertGreaterEqual(len(result["counterevidence"]), 4)
            persisted = json.loads((output / "comparison.json").read_text())
            self.assertEqual(persisted["counterevidence"], result["counterevidence"])


if __name__ == "__main__":
    unittest.main()
