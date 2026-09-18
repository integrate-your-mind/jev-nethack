from __future__ import annotations

import unittest

from candidate import (
    BASELINE_INSTRUCTIONS,
    CANDIDATE_INSTRUCTIONS,
    augment_state,
    build_cycle_context,
    prepare_candidate,
)


def action(before_x, after_x, label):
    return {
        "action": label,
        "before": {"dungeon": 0, "level": 1, "x": before_x, "y": 4},
        "after": {"dungeon": 0, "level": 1, "x": after_x, "y": 4},
        "message_after": "",
    }


class CycleAwareCandidateTests(unittest.TestCase):
    def state(self):
        return {
            "player": {
                "dungeon": 0,
                "level": 1,
                "x": 38,
                "y": 4,
                "hp": 17,
                "max_hp": 17,
                "hunger_code": 1,
                "condition_bits": 0,
            },
            "adjacent_cells": {
                "HERE": {"visits": 532},
                "W": {"visits": 666},
                "E": {"visits": 118},
            },
            "message": "",
            "terminal": "",
            "recent_actions": [
                action(38, 37, "west"),
                action(37, 38, "east"),
                action(38, 37, "west"),
                action(37, 38, "east"),
            ],
        }

    def test_detects_repeated_two_step_cycle_from_observed_positions(self):
        context = build_cycle_context(self.state())
        self.assertTrue(context["detected"])
        self.assertEqual(3, context["twoStepPositionReversals"])
        self.assertTrue(context["immediateReversal"])
        self.assertEqual(2, context["recentUniquePositions"])
        self.assertEqual(666, context["adjacentVisitCounts"]["W"])

    def test_preserves_all_121_criteria_exactly(self):
        criteria = {f"a{index}": f"action {index}" for index in range(121)}
        state = self.state()
        candidate_state, candidate_criteria, instructions = prepare_candidate(state, criteria)
        self.assertEqual(criteria, candidate_criteria)
        self.assertEqual(list(criteria), list(candidate_criteria))
        self.assertIsNot(criteria, candidate_criteria)
        self.assertNotIn("cycle_context", state)
        self.assertIn("cycle_context", candidate_state)
        self.assertEqual(CANDIDATE_INSTRUCTIONS, instructions)

    def test_does_not_filter_force_or_reweight_actions(self):
        criteria = {"a0": "north", "a1": "east", "a2": "retreat"}
        _, candidate_criteria, _ = prepare_candidate(self.state(), criteria)
        self.assertEqual(criteria, candidate_criteria)
        self.assertFalse(any(key in candidate_criteria for key in ("forced", "blocked", "weight")))

    def test_explicitly_allows_safety_combat_and_prompt_reversal(self):
        text = CANDIDATE_INSTRUCTIONS.lower()
        for phrase in ("combat", "immediate survival", "retreat", "menu", "--more--", "only safe"):
            self.assertIn(phrase, text)
        self.assertIn("do not mechanically forbid reversal", text)

    def test_cycle_breaking_requires_plausibly_traversable_direction(self):
        text = CANDIDATE_INSTRUCTIONS.lower()
        for phrase in ("adjacent_cells", "plausibly traversable", "blank", "wall", "solid-stone"):
            self.assertIn(phrase, text)

    def test_prompt_and_low_hp_are_observed_exception_signals(self):
        state = self.state()
        state["player"]["hp"] = 3
        state["message"] = "What do you want to drink? [e or ?*]"
        signals = build_cycle_context(state)["exceptionSignals"]
        self.assertTrue(signals["lowHp"])
        self.assertTrue(signals["visiblePromptOrMenu"])

    def test_recent_combat_message_is_an_observed_exception_signal(self):
        state = self.state()
        state["message"] = "The goblin thrusts his crude dagger.  The goblin hits!"
        self.assertTrue(build_cycle_context(state)["exceptionSignals"]["recentCombat"])

    def test_normal_noncycling_state_is_not_labeled_cycle(self):
        state = self.state()
        state["recent_actions"] = [
            action(30, 31, "east"),
            action(31, 32, "east"),
            action(32, 33, "east"),
        ]
        context = build_cycle_context(state)
        self.assertFalse(context["detected"])
        self.assertEqual(0, context["twoStepPositionReversals"])

    def test_baseline_text_is_preserved_as_prefix(self):
        self.assertTrue(CANDIDATE_INSTRUCTIONS.startswith(BASELINE_INSTRUCTIONS))


if __name__ == "__main__":
    unittest.main()
