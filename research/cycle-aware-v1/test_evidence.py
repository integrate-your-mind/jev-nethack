from __future__ import annotations

import json
from pathlib import Path
import unittest


class EvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {
            row["fixtureId"]: row
            for row in map(json.loads, Path("fixtures.jsonl").read_text().splitlines())
        }
        cls.results = list(
            map(json.loads, Path("evaluation-results.jsonl").read_text().splitlines())
        )
        cls.summary = json.loads(Path("evaluation-summary.json").read_text())

    def test_all_fixture_categories_are_present(self):
        self.assertEqual(16, len(self.fixtures))
        self.assertEqual(
            {"stagnant_cycle": 6, "normal_play": 4, "menu_control": 3, "survival_control": 3},
            {category: sum(row["category"] == category for row in self.fixtures.values()) for category in {row["category"] for row in self.fixtures.values()}},
        )

    def test_every_result_retains_exact_full_probability_support(self):
        self.assertEqual(16, len(self.results))
        for result in self.results:
            fixture = self.fixtures[result["fixtureId"]]
            self.assertEqual(121, result["criteriaCount"])
            self.assertEqual(set(fixture["criteria"]), set(result["baseline"]["probabilities"]))
            self.assertEqual(set(fixture["criteria"]), set(result["candidate"]["probabilities"]))
            self.assertTrue(result["criteriaPreserved"])

    def test_no_credentials_or_raw_provider_bodies_are_persisted(self):
        for path in Path(".").glob("*.json*"):
            text = path.read_text()
            self.assertNotIn("TYPESAFE_API_KEY", text)
            self.assertNotIn("Authorization", text)
            self.assertNotIn("request_body_base64", text)
            self.assertNotIn("response_body_base64", text)

    def test_summary_reports_bounded_calls_and_limits(self):
        self.assertEqual(32, self.summary["attemptedApiCalls"])
        self.assertTrue(self.summary["allCriteriaPreserved"])
        self.assertTrue(self.summary["rawProbabilityMapsRetained"])
        self.assertIn("Confidence is distribution concentration", " ".join(self.summary["limitations"]))


if __name__ == "__main__":
    unittest.main()
