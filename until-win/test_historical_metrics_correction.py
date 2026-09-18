import copy
from contextlib import redirect_stderr
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import historical_metrics_correction as correction_module
from historical_metrics_correction import (
    RUNTIME_SCHEMA,
    apply_correction,
    build_correction,
    main,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def transition(step: int, before: int, after: int, reward: float, *, terminal: bool = False) -> dict:
    return {
        "schema": RUNTIME_SCHEMA,
        "eventType": "transition_committed",
        "eventId": f"episode-{0:08d}-step-{step:09d}",
        "episodeId": 0,
        "seed": 103,
        "step": step,
        "actionIndex": 0,
        "keycode": 1,
        "observationDigest": f"obs-{step}",
        "nextObservationDigest": f"obs-{step + 1}",
        "reward": reward,
        "terminated": terminal,
        "truncated": False,
        "isAscended": False,
        "verifiedAscension": False,
        "endStatus": "DEATH" if terminal else "RUNNING",
        "state": {"player": {"score": before}},
        "nextState": {"player": {"score": after}},
        "provenance": {"origin": "native_full_observation"},
    }


def write_fragment(data: Path, name: str, attempt_label: str, events: list[dict], summary: dict | None = None) -> Path:
    fragment = data / "fragments" / name
    attempt = fragment / "episodes" / f"episode-00000000-seed-103-{attempt_label}"
    attempt.mkdir(parents=True)
    write_json(
        fragment / "config.json",
        {
            "schema": RUNTIME_SCHEMA,
            "fragmentId": name,
            "pid": 999991,
        },
    )
    rows = list(events)
    if summary is not None:
        write_json(attempt / "summary.json", summary)
        rows.append({"eventType": "episode_summary", **summary})
    (fragment / "transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return attempt


def fixture(base: str) -> tuple[Path, Path, dict, dict]:
    data = Path(base) / "data"
    global_lock = data / "global.lock"
    stopped_row = {
        "episodeId": 0,
        "seed": 103,
        "score": 2,
        "maxDepth": 1,
        "steps": 1,
        "status": "stop_requested",
        "isAscended": False,
        "terminated": False,
        "truncated": False,
    }
    terminal_row = {
        "episodeId": 0,
        "seed": 103,
        "score": 0,
        "maxDepth": 1,
        "steps": 3,
        "status": "game_end",
        "isAscended": False,
        "terminated": True,
        "truncated": False,
    }
    stopped_summary = {
        "schema": RUNTIME_SCHEMA,
        **stopped_row,
        "endedAt": "2026-09-18T14:00:00+00:00",
        "endStatus": "RUNNING",
        "totalReward": 1.0,
        "wallSeconds": 1.0,
        "replayOrigin": "native_full_observation",
    }
    terminal_summary = {
        "schema": RUNTIME_SCHEMA,
        **terminal_row,
        "endedAt": "2026-09-18T15:00:00+00:00",
        "endStatus": "DEATH",
        "totalReward": 6.0,
        "wallSeconds": 2.0,
        "replayOrigin": "native_full_observation",
    }
    write_fragment(
        data,
        "fragment-a-pid999991",
        "replay",
        [transition(0, 0, 2, 1.0)],
        stopped_summary,
    )
    write_fragment(
        data,
        "fragment-b-pid999991",
        "replay",
        [transition(1, 2, 5, 2.0)],
    )
    terminal_attempt = write_fragment(
        data,
        "fragment-c-pid999991",
        "replay",
        [transition(2, 5, 0, 3.0, terminal=True)],
        terminal_summary,
    )
    ttyrec = terminal_attempt / "nle-ttyrec"
    ttyrec.mkdir()
    (ttyrec / "nle.4242.0.ttyrec3.bz2").write_bytes(b"retained native terminal recording")
    (ttyrec / "nle.4242.xlogfile").write_text(
        "version=3.6.7\tpoints=7\tdeath=died of starvation\twhile=fainted"
        "\tturns=12\tttyrecname=nle.4242.0.ttyrec3.bz2\n"
    )
    training = terminal_attempt.parent.parent / "training"
    training.mkdir()
    (training / "transitions-000000.npzpack").write_bytes(b"sentinel raw transition pack")
    (training / "transitions-000000.index.jsonl").write_bytes(b'{"sentinel":true}\n')
    (training / "transitions-000000.complete.json").write_bytes(b'{"completed":true}\n')
    state = {
        "schema": RUNTIME_SCHEMA,
        "status": "paused",
        "activeEpisode": None,
        "activeFragment": None,
        "bestScore": 2,
        "episodeResults": [copy.deepcopy(stopped_row), copy.deepcopy(terminal_row)],
    }
    state_path = data / "state.json"
    write_json(state_path, state)
    (data / "control").mkdir()
    (data / "recovery").mkdir()
    (data / "control" / "STOP").write_text("")
    write_json(
        data / "handoff-receipt.json",
        {
            "schema": RUNTIME_SCHEMA,
            "accepted": True,
            "conflicts": [],
            "globalLock": str(global_lock),
            "checkedAt": "before-restart",
        },
    )
    return data, state_path, stopped_row, terminal_row


def reviewed_dry_run(data: Path, state_path: Path, *, episode_id: int = 0) -> tuple[Path, str]:
    value = {
        **build_correction(data_root=data, state_path=state_path, episode_id=episode_id),
        "phase": "dry_run",
        "applied": False,
    }
    path = data / "review" / f"episode-{episode_id:08d}.dry-run.json"
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def apply_args(
    data: Path,
    state_path: Path,
    *,
    receipt: Path | None = None,
    global_lock: Path | None = None,
) -> dict:
    expected, expected_sha256 = reviewed_dry_run(data, state_path)
    return {
        "data_root": data,
        "state_path": state_path,
        "episode_id": 0,
        "receipt_path": receipt or data / "recovery" / "correction.json",
        "global_lock": global_lock or data / "global.lock",
        "expected_dry_run": expected,
        "expected_dry_run_sha256": expected_sha256,
    }


def raw_hashes(data: Path) -> dict[str, str]:
    result = {}
    for path in sorted((data / "fragments").rglob("*")):
        if path.is_file():
            result[path.relative_to(data).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class HistoricalMetricCorrectionTests(unittest.TestCase):
    def test_dry_run_cli_rejects_write_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "unexpected.json"
            cases = (
                ("--receipt", str(receipt)),
                ("--global-lock", str(data / "global.lock")),
                ("--expected-dry-run", str(data / "review.json")),
                ("--expected-dry-run-sha256", "0" * 64),
            )
            for option, value in cases:
                argv = [
                    "historical_metrics_correction.py",
                    "--data-root",
                    str(data),
                    "--state",
                    str(state_path),
                    "--episode-id",
                    "0",
                    option,
                    value,
                ]
                with self.subTest(option=option), patch("sys.argv", argv), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main()
            self.assertFalse(receipt.exists())

    def test_duplicate_interrupted_row_is_preserved_and_terminal_bundle_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, stopped_row, _ = fixture(temporary)
            state_before = state_path.read_bytes()
            raw_before = raw_hashes(data)
            result = build_correction(data_root=data, state_path=state_path, episode_id=0)
            self.assertEqual(7, result["officialFinalScore"])
            self.assertEqual(5, result["lastObservedScore"])
            self.assertEqual(5, result["maxObservedScore"])
            self.assertEqual(6.0, result["totalReward"])
            self.assertEqual("DEATH", result["endStatus"])
            self.assertEqual("died of starvation", result["deathCause"])
            self.assertEqual("fainted", result["deathWhile"])
            self.assertEqual(1, result["target"]["episodeResultIndex"])
            self.assertEqual(3, result["evidence"]["transitionCount"])
            self.assertEqual("episode-00000000-step-000000002", result["evidence"]["terminalTransition"]["eventId"])
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertEqual(raw_before, raw_hashes(data))
            self.assertEqual(stopped_row, json.loads(state_path.read_text())["episodeResults"][0])

    def test_duplicate_terminal_row_and_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, terminal_row = fixture(temporary)
            state = json.loads(state_path.read_text())
            state["episodeResults"].append(copy.deepcopy(terminal_row))
            write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            summary_path = next(
                path
                for path in data.rglob("summary.json")
                if json.loads(path.read_text()).get("status") == "game_end"
            )
            summary = json.loads(summary_path.read_text())
            duplicate = write_fragment(data, "fragment-d-pid999991", "replay", [], summary)
            ttyrec = duplicate / "nle-ttyrec"
            ttyrec.mkdir()
            (ttyrec / "nle.4242.0.ttyrec3.bz2").write_bytes(b"copy")
            (ttyrec / "nle.4242.xlogfile").write_text(
                "version=3.6.7\tpoints=7\tttyrecname=nle.4242.0.ttyrec3.bz2\n"
            )
            with self.assertRaisesRegex(ValueError, "summary is missing or ambiguous"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

    def test_summary_event_or_transition_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            journal = data / "fragments" / "fragment-c-pid999991" / "transitions.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            rows[-1]["endStatus"] = "RUNNING"
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "summary event"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            journal = data / "fragments" / "fragment-b-pid999991" / "transitions.jsonl"
            conflict = transition(0, 999, 999, 1.0)
            with journal.open("a") as stream:
                stream.write(json.dumps(conflict) + "\n")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

    def test_state_and_matching_event_schema_must_match_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            state = json.loads(state_path.read_text())
            state["schema"] = "wrong-schema"
            write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, "state.json schema"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            journal = data / "fragments" / "fragment-b-pid999991" / "transitions.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            rows[0]["schema"] = "wrong-schema"
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "episode evidence schema"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

    def test_transition_digest_and_score_chains_must_be_continuous(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            journal = data / "fragments" / "fragment-b-pid999991" / "transitions.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            rows[0]["observationDigest"] = "disconnected-observation"
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "observation chain"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            journal = data / "fragments" / "fragment-b-pid999991" / "transitions.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            rows[0]["state"]["player"]["score"] = 999
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "score chain"):
                build_correction(data_root=data, state_path=state_path, episode_id=0)

    def test_apply_requires_stop_paused_inactive_state_and_free_kernel_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            lock_path = data / "global.lock"

            state = json.loads(state_path.read_text())
            state["status"] = "ready"
            write_json(state_path, state)
            with self.assertRaisesRegex(RuntimeError, "paused"):
                apply_correction(**apply_args(data, state_path, receipt=receipt, global_lock=lock_path))

            state["status"] = "paused"
            state["activeEpisode"] = {"episodeId": 1}
            write_json(state_path, state)
            with self.assertRaisesRegex(RuntimeError, "active"):
                apply_correction(**apply_args(data, state_path, receipt=receipt, global_lock=lock_path))

            state["activeEpisode"] = None
            write_json(state_path, state)
            (data / "control" / "STOP").unlink()
            with self.assertRaisesRegex(RuntimeError, "STOP"):
                apply_correction(**apply_args(data, state_path, receipt=receipt, global_lock=lock_path))

            (data / "control" / "STOP").write_text("")
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "held"):
                    apply_correction(**apply_args(data, state_path, receipt=receipt, global_lock=lock_path))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def test_apply_rejects_wrong_global_lock_and_held_runtime_lock_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            state_before = state_path.read_bytes()
            wrong_lock = data / "wrong-global.lock"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                apply_correction(**apply_args(data, state_path, receipt=receipt, global_lock=wrong_lock))
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertFalse(receipt.exists())

            runtime_lock = data / "runtime.lock"
            fd = os.open(runtime_lock, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "held"):
                    apply_correction(**apply_args(data, state_path, receipt=receipt))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertFalse(receipt.exists())

    def test_apply_requires_preexisting_receipt_parent_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt_parent = data / "missing"
            receipt = receipt_parent / "receipt.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            state_before = state_path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "pre-existing regular directory"):
                apply_correction(**kwargs)

            self.assertEqual(state_before, state_path.read_bytes())
            self.assertFalse(receipt.exists())
            self.assertFalse(receipt_parent.exists())

    def test_apply_rejects_source_drift_after_review_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            state_before = state_path.read_bytes()
            xlog = next((data / "fragments").rglob("*.xlogfile"))
            xlog.write_text(xlog.read_text().replace("points=7", "points=8"))

            with self.assertRaisesRegex(RuntimeError, "does not match the reviewed dry-run"):
                apply_correction(**kwargs)
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertFalse(receipt.exists())

    def test_apply_rechecks_summary_fragment_producer_and_xlog_membership_before_state_write(self):
        def add_second_terminal_summary(data: Path) -> None:
            terminal_summary = next(
                path
                for path in data.rglob("summary.json")
                if json.loads(path.read_text()).get("status") == "game_end"
            )
            destination = (
                data
                / "fragments"
                / "fragment-b-pid999991"
                / "episodes"
                / "episode-00000000-seed-103-replay"
                / "summary.json"
            )
            destination.write_bytes(terminal_summary.read_bytes())

        def add_conflicting_fragment(data: Path) -> None:
            write_fragment(
                data,
                "fragment-injected-conflict-pid999991",
                "injected",
                [transition(0, 999, 999, 1.0)],
            )

        def add_live_producer_fragment(data: Path) -> None:
            fragment = data / "fragments" / f"fragment-injected-live-pid{os.getpid()}"
            attempt = fragment / "episodes" / "episode-00000000-seed-103-injected"
            attempt.mkdir(parents=True)
            write_json(
                fragment / "config.json",
                {"schema": RUNTIME_SCHEMA, "fragmentId": fragment.name, "pid": os.getpid()},
            )
            (fragment / "transitions.jsonl").write_text("")

        def add_second_xlog(data: Path) -> None:
            ttyrec = next(data.rglob("nle.4242.xlogfile")).parent
            (ttyrec / "nle.999.xlogfile").write_text(
                "version=3.6.7\tpoints=7\tttyrecname=nle.999.0.ttyrec3.bz2\n"
            )

        cases = (
            ("second_summary", add_second_terminal_summary, "summary is missing or ambiguous"),
            ("conflicting_fragment", add_conflicting_fragment, "transition step conflicts"),
            ("live_producer", add_live_producer_fragment, "active producer"),
            ("second_xlog", add_second_xlog, "ambiguous_xlogfile"),
        )
        for label, inject, expected_error in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                data, state_path, _, _ = fixture(temporary)
                receipt = data / "recovery" / "correction.json"
                kwargs = apply_args(data, state_path, receipt=receipt)
                state_before = state_path.read_bytes()
                original_verify = correction_module._verify_source_artifacts
                injected = False

                def inject_at_first_verify(data_root: Path, expected) -> None:
                    nonlocal injected
                    if not injected:
                        inject(data_root)
                        injected = True
                    original_verify(data_root, expected)

                with patch(
                    "historical_metrics_correction._verify_source_artifacts",
                    side_effect=inject_at_first_verify,
                ):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        apply_correction(**kwargs)
                self.assertTrue(injected)
                self.assertEqual(state_before, state_path.read_bytes())
                self.assertEqual("prepared", json.loads(receipt.read_text())["phase"])
                self.assertNotIn(
                    "sourceArtifactsVerifiedAfterStateWrite",
                    json.loads(receipt.read_text()),
                )

    def test_apply_rejects_tampered_review_file_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            state_before = state_path.read_bytes()
            kwargs["expected_dry_run"].write_bytes(
                kwargs["expected_dry_run"].read_bytes() + b" \n"
            )

            with self.assertRaisesRegex(RuntimeError, "file hash does not match"):
                apply_correction(**kwargs)
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertFalse(receipt.exists())

    def test_xlog_change_between_native_parse_and_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            original = correction_module._source_artifact
            changed = False

            def swap_before_hash(path: Path, *, data_root: Path):
                nonlocal changed
                if not changed and path.name.endswith(".xlogfile"):
                    path.write_text(path.read_text().replace("points=7", "points=8"))
                    changed = True
                return original(path, data_root=data_root)

            with patch("historical_metrics_correction._source_artifact", side_effect=swap_before_hash):
                with self.assertRaisesRegex(ValueError, "changed while the correction was built"):
                    build_correction(data_root=data, state_path=state_path, episode_id=0)
            self.assertTrue(changed)

    def test_apply_updates_only_terminal_row_preserves_raw_sources_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, stopped_row, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            raw_before = raw_hashes(data)
            kwargs = apply_args(data, state_path, receipt=receipt)
            atomic_writes: list[Path] = []
            original_atomic = correction_module._atomic_bytes

            def record_atomic_write(path: Path, raw: bytes) -> None:
                atomic_writes.append(path)
                original_atomic(path, raw)

            with patch("historical_metrics_correction._atomic_bytes", side_effect=record_atomic_write):
                result = apply_correction(**kwargs)
            state = json.loads(state_path.read_text())
            self.assertEqual(stopped_row, state["episodeResults"][0])
            corrected = state["episodeResults"][1]
            self.assertEqual(7, corrected["score"])
            self.assertEqual("official_final", corrected["scoreSemantics"])
            self.assertEqual(7, corrected["officialFinalScore"])
            self.assertEqual(5, corrected["lastObservedScore"])
            self.assertEqual(5, corrected["maxObservedScore"])
            self.assertEqual(6.0, corrected["totalReward"])
            self.assertEqual("DEATH", corrected["endStatus"])
            self.assertEqual(7, state["bestScore"])
            self.assertEqual("completed", result["phase"])
            self.assertTrue(result["sourceArtifactsVerifiedAfterStateWrite"])
            self.assertEqual({receipt, state_path}, set(atomic_writes))
            self.assertEqual(raw_before, raw_hashes(data))
            state_after = state_path.read_bytes()
            receipt_after = receipt.read_bytes()
            again = apply_correction(**kwargs)
            self.assertEqual(result, again)
            self.assertEqual(state_after, state_path.read_bytes())
            self.assertEqual(receipt_after, receipt.read_bytes())

            forged = json.loads(receipt.read_text())
            forged["officialFinalScore"] = 999_999
            write_json(receipt, forged)
            with self.assertRaisesRegex(RuntimeError, "conflicts at officialFinalScore"):
                apply_correction(**kwargs)
            self.assertEqual(state_after, state_path.read_bytes())
            self.assertEqual(999_999, json.loads(receipt.read_text())["officialFinalScore"])

    def test_prepared_receipt_survives_state_write_failure_and_retry_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            state_before = state_path.read_bytes()
            original_atomic = correction_module._atomic_bytes

            def fail_state_write(path: Path, raw: bytes) -> None:
                if path == state_path:
                    raise OSError("simulated state replacement failure")
                original_atomic(path, raw)

            with patch("historical_metrics_correction._atomic_bytes", side_effect=fail_state_write):
                with self.assertRaisesRegex(OSError, "simulated"):
                    apply_correction(**kwargs)
            self.assertEqual(state_before, state_path.read_bytes())
            self.assertEqual("prepared", json.loads(receipt.read_text())["phase"])
            completed = apply_correction(**kwargs)
            self.assertEqual("completed", completed["phase"])
            self.assertEqual(7, json.loads(state_path.read_text())["episodeResults"][1]["officialFinalScore"])

    def test_prepared_receipt_recovers_after_state_write_and_handoff_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            original_atomic_json = correction_module._atomic_json

            def fail_completed_receipt(path: Path, value) -> None:
                if path == receipt and value.get("phase") == "completed":
                    raise OSError("simulated completion receipt failure")
                original_atomic_json(path, value)

            with patch("historical_metrics_correction._atomic_json", side_effect=fail_completed_receipt):
                with self.assertRaisesRegex(OSError, "completion receipt"):
                    apply_correction(**kwargs)
            self.assertEqual(7, json.loads(state_path.read_text())["episodeResults"][1]["officialFinalScore"])
            prepared = json.loads(receipt.read_text())
            self.assertEqual("prepared", prepared["phase"])

            handoff_path = data / "handoff-receipt.json"
            handoff = json.loads(handoff_path.read_text())
            handoff["checkedAt"] = "after-restart"
            write_json(handoff_path, handoff)

            completed = apply_correction(**kwargs)
            self.assertEqual("completed", completed["phase"])
            self.assertTrue(completed["sourceArtifactsVerifiedAfterStateWrite"])
            self.assertEqual(prepared["preparedHandoffReceipt"], completed["preparedHandoffReceipt"])
            self.assertNotEqual(
                completed["preparedHandoffReceipt"]["sha256"],
                completed["completedHandoffReceipt"]["sha256"],
            )

    def test_prepared_receipt_rejects_non_target_state_and_after_hash_forgery(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, state_path, _, _ = fixture(temporary)
            receipt = data / "recovery" / "correction.json"
            kwargs = apply_args(data, state_path, receipt=receipt)
            original_atomic_json = correction_module._atomic_json

            def fail_completed_receipt(path: Path, value) -> None:
                if path == receipt and value.get("phase") == "completed":
                    raise OSError("simulated completion receipt failure")
                original_atomic_json(path, value)

            with patch("historical_metrics_correction._atomic_json", side_effect=fail_completed_receipt):
                with self.assertRaisesRegex(OSError, "completion receipt"):
                    apply_correction(**kwargs)

            tampered_state = json.loads(state_path.read_text())
            tampered_state["episodeResults"][0]["score"] = 999_999
            write_json(state_path, tampered_state)
            prepared = json.loads(receipt.read_text())
            prepared["stateSha256After"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            write_json(receipt, prepared)
            state_before_retry = state_path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "does not match reviewed evidence"):
                apply_correction(**kwargs)
            self.assertEqual(state_before_retry, state_path.read_bytes())
            self.assertEqual("prepared", json.loads(receipt.read_text())["phase"])


if __name__ == "__main__":
    unittest.main()
