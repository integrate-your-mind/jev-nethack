from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

import candidate


FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())


def built_memory() -> tuple[dict, dict]:
    memory = candidate.empty_memory()
    states = FIXTURES["longCycleWithRememberedFrontier"]["states"]
    for state in states:
        memory = candidate.observe(memory, state)
    return memory, states[-1]


class FrontierMemoryTests(unittest.TestCase):
    def test_persistent_memory_retains_frontier_beyond_eight_actions(self) -> None:
        memory, state = built_memory()
        context = candidate.frontier_context(memory, state)
        target = FIXTURES["longCycleWithRememberedFrontier"]["expectedTarget"]
        match = next(item for item in context["frontierCandidates"] if item["target"] == target)
        self.assertEqual(match["routeDirections"], ["E", "E"])
        self.assertEqual(match["frontierDirection"], "E")
        self.assertEqual(context["observationsRetained"], 11)
        self.assertEqual(context["observedNodeCount"], 3)

    def test_memory_contains_no_unobserved_node_or_guessed_edge(self) -> None:
        memory, _ = built_memory()
        self.assertEqual(set(memory["nodes"]), {"0:1:0:4", "0:1:1:4", "0:1:2:4"})
        for edge in memory["transitions"].values():
            self.assertIn(":".join(map(str, edge["from"])), memory["nodes"])
            self.assertIn(":".join(map(str, edge["to"])), memory["nodes"])

    def test_observe_is_pure_and_does_not_mutate_prior_memory_or_state(self) -> None:
        memory = candidate.empty_memory()
        state = FIXTURES["longCycleWithRememberedFrontier"]["states"][0]
        before_memory, before_state = deepcopy(memory), deepcopy(state)
        candidate.observe(memory, state)
        self.assertEqual(memory, before_memory)
        self.assertEqual(state, before_state)

    def test_raw_observation_digest_is_retained_as_provenance(self) -> None:
        state = FIXTURES["longCycleWithRememberedFrontier"]["states"][0]
        digest = "a" * 64
        memory = candidate.observe(candidate.empty_memory(), state, raw_observation_digest=digest)
        self.assertEqual(memory["nodes"]["0:1:0:4"]["lastRawObservationDigest"], digest)

    def test_raw_observation_digest_rejects_non_sha256_provenance(self) -> None:
        state = FIXTURES["longCycleWithRememberedFrontier"]["states"][0]
        with self.assertRaises(ValueError):
            candidate.observe(candidate.empty_memory(), state, raw_observation_digest="short")
        with self.assertRaises(ValueError):
            candidate.observe(candidate.empty_memory(), state, raw_observation_digest="z" * 64)

    def test_missing_symbol_is_not_invented_as_traversable_frontier(self) -> None:
        state = deepcopy(FIXTURES["longCycleWithRememberedFrontier"]["states"][0])
        state["adjacent_cells"]["W"] = {"description": "unknown"}
        memory = candidate.observe(candidate.empty_memory(), state)
        context = candidate.frontier_context(memory, state)
        self.assertFalse(any(item["frontierDirection"] == "W" for item in context["frontierCandidates"]))

    def test_frontier_route_is_advisory_data_without_action_execution(self) -> None:
        memory, state = built_memory()
        context = candidate.frontier_context(memory, state)
        serialized = json.dumps(context)
        self.assertNotIn("forcedAction", serialized)
        self.assertNotIn("executeAction", serialized)
        self.assertIn("never execute or restrict", context["limits"][-1])


class HierarchicalQuestionTests(unittest.TestCase):
    def test_goal_question_has_explicit_prompt_survival_combat_hunger_priority(self) -> None:
        memory, _ = built_memory()
        state = FIXTURES["combatSurvivalControl"]
        augmented, criteria, instructions = candidate.prepare_goal_question(state, memory)
        self.assertEqual(criteria, candidate.GOAL_CRITERIA)
        self.assertTrue(augmented["priority_signals"]["lowHp"])
        self.assertTrue(augmented["priority_signals"]["urgentHunger"])
        self.assertTrue(augmented["priority_signals"]["recentCombat"])
        for text in ("prompt", "survival", "combat", "hunger"):
            self.assertIn(text, instructions.lower())

    def test_goal_contract_retains_complete_raw_judgment_and_expires(self) -> None:
        memory, state = built_memory()
        raw = {
            "choice": "follow_observed_frontier_route",
            "confidence": 0.31,
            "probabilities": {key: 1 / len(candidate.GOAL_CRITERIA) for key in candidate.GOAL_CRITERIA},
            "request_body_base64": "request",
            "response_body_base64": "response",
        }
        contract = candidate.new_goal_contract(raw, state, memory, 40)
        self.assertEqual(contract["rawGoalDecision"], raw)
        self.assertEqual(contract["expiresBeforeAction"], 46)
        self.assertIn("goal_horizon_exhausted", candidate.goal_refresh_reasons(state, memory, contract, 46))

    def test_changed_prompt_forces_goal_refresh_not_an_action(self) -> None:
        memory, _ = built_memory()
        prompt = FIXTURES["promptControl"]
        contract = {"choice": "follow_observed_frontier_route", "expiresBeforeAction": 99}
        reasons = candidate.goal_refresh_reasons(prompt, memory, contract, 12)
        self.assertIn("visible_prompt_changed_priority", reasons)
        self.assertFalse(any(reason.startswith("action_") for reason in reasons))

    def test_action_question_preserves_all_121_candidates_and_order(self) -> None:
        memory, state = built_memory()
        raw_goal = {"choice": "follow_observed_frontier_route", "confidence": 0.4, "probabilities": {}}
        contract = candidate.new_goal_contract(raw_goal, state, memory, 10)
        criteria = {f"a{index}": f"action {index}" for index in range(121)}
        original = deepcopy(criteria)
        augmented, result, instructions = candidate.prepare_action_question(state, memory, contract, criteria)
        self.assertEqual(result, original)
        self.assertEqual(list(result), list(original))
        self.assertEqual(len(result), 121)
        self.assertEqual(criteria, original)
        self.assertIn("complete 121 action candidates remains legal", instructions)
        self.assertEqual(augmented["active_goal"]["choice"], raw_goal["choice"])

    def test_retention_record_keeps_both_raw_distributions(self) -> None:
        goal = {"choice": "consolidate_or_replan", "probabilities": {"a": 0.2, "b": 0.8}}
        action = {"choice": "a1", "probabilities": {f"a{i}": 1 / 121 for i in range(121)}}
        record = candidate.retention_record(goal, action)
        self.assertEqual(record["goalDecision"], goal)
        self.assertEqual(record["actionDecision"], action)
        self.assertIn("not accuracy", record["confidenceSemantics"])

    def test_frozen_trial_is_bounded_unrun_and_has_no_replacement_points(self) -> None:
        plan = json.loads((Path(__file__).parent / "frozen-trial-plan.json").read_text())
        self.assertEqual(plan["status"], "frozen_not_run")
        self.assertEqual(plan["maximumProviderCalls"], 192)
        self.assertEqual(plan["source"]["branchPointsAfterCommittedStep"], [3359, 3999])
        self.assertIn("do not replace a branch point, tune questions, or extend an arm after results", plan["stopRules"])


if __name__ == "__main__":
    unittest.main()
