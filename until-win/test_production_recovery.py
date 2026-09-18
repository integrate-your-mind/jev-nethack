"""Opt-in real-NLE recovery proof against a copied production data root.

Set JEV_NETHACK_PRODUCTION_RECOVERY_FIXTURE to a stopped runtime root.  The
fixture is copied before any write.  This test never constructs a Jev client;
its local policy commits one deterministic action after replay on each of two
successive process starts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from until_win import DEFAULT_ENGINE_EPISODE_LIMIT, UntilWinSupervisor


FIXTURE_ENV = "JEV_NETHACK_PRODUCTION_RECOVERY_FIXTURE"


class OneLocalActionThenStop:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.calls = 0

    def choose(self, _state, criteria, _instructions):
        self.calls += 1
        self.stop_event.set()
        choice = "a18" if "a18" in criteria else sorted(criteria)[0]
        return {
            "choice": choice,
            "confidence": 1.0,
            "probabilities": {key: 1.0 if key == choice else 0.0 for key in criteria},
            "model": "offline-production-recovery-test",
            "request_digest": "no-provider-call",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


@unittest.skipUnless(os.environ.get(FIXTURE_ENV), f"set {FIXTURE_ENV} to run")
class ProductionRecoveryIntegrationTests(unittest.TestCase):
    def test_imported_prefix_native_suffix_continues_across_two_real_nle_restarts(self):
        source = Path(os.environ[FIXTURE_ENV]).expanduser().resolve()
        self.assertTrue((source / "control" / "STOP").is_file(), "fixture must be stopped")
        original = json.loads((source / "state.json").read_text())
        candidate = original["resumeCandidate"]
        self.assertEqual("paused", original["status"])
        self.assertEqual(0, candidate["episodeId"])
        self.assertEqual(103, candidate["seed"])
        self.assertEqual(104, original["nextSeed"])
        self.assertEqual(1980, original["totalActions"])

        # NLE passes save paths through native buffers.  Keep the copied-root
        # path short so the test exercises replay rather than path truncation.
        with tempfile.TemporaryDirectory(prefix="jev-nh-recovery-", dir="/tmp") as temporary:
            root = Path(temporary) / "until-win"
            shutil.copytree(source, root)

            migration = UntilWinSupervisor(
                data_root=root,
                character="mon-hum-neu-mal",
                first_seed=0,
                engine_episode_limit=DEFAULT_ENGINE_EPISODE_LIMIT,
                policy=None,
                stop_event=threading.Event(),
                stop_file=root / "control" / "STOP",
            )
            receipt = migration.migrate_recovery_evidence()
            migration.recovery_log.close()
            self.assertEqual(1980, receipt["actionsPreserved"])
            self.assertEqual([{"first": 1967, "last": 1979, "count": 13}], receipt["nativeEvidenceStepRanges"])
            self.assertEqual([{"first": 0, "last": 1966, "count": 1967}], receipt["legacyRawRegenerationStepRanges"])
            self.assertEqual(0, receipt["providerCalls"])
            self.assertFalse(receipt["freshSeedAllocated"])
            self.assertIn("broadcast.py", receipt["migrationSourceSha256"])
            self.assertIn("recovery.py", receipt["migrationSourceSha256"])

            (root / "control" / "STOP").unlink()
            expected_total = 1980
            for restart in (1, 2):
                stop_event = threading.Event()
                policy = OneLocalActionThenStop(stop_event)
                supervisor = UntilWinSupervisor(
                    data_root=root,
                    character="mon-hum-neu-mal",
                    first_seed=0,
                    engine_episode_limit=DEFAULT_ENGINE_EPISODE_LIMIT,
                    policy=policy,
                    stop_event=stop_event,
                    stop_file=root / "control" / "STOP",
                )
                state = supervisor.run()
                expected_total += 1
                self.assertEqual(1, policy.calls)
                self.assertEqual("paused", state["status"])
                self.assertEqual(expected_total, state["totalActions"])
                self.assertEqual(104, state["nextSeed"])
                resumed = state["resumeCandidate"]
                self.assertEqual(0, resumed["episodeId"])
                self.assertEqual(103, resumed["seed"])
                self.assertEqual(expected_total, resumed["stepsCommitted"])

                verifier = UntilWinSupervisor(
                    data_root=root,
                    character="mon-hum-neu-mal",
                    first_seed=0,
                    engine_episode_limit=DEFAULT_ENGINE_EPISODE_LIMIT,
                    policy=None,
                    stop_event=threading.Event(),
                    stop_file=root / "control" / "STOP",
                )
                plan = verifier._load_candidate_plan(resumed)
                verifier.recovery_log.close()
                self.assertEqual(list(range(expected_total)), [record["step"] for record in plan["records"]])
                self.assertEqual(list(range(expected_total)), plan["nativeSteps"])


if __name__ == "__main__":
    unittest.main()
