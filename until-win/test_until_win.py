import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from nle import nethack

from broadcast import BroadcastRecorder
from jev_client import JevHTTPError, JevTransportError
from transition_pack import TransitionPackError, TransitionPackWriter, scan_pack
from until_win import (
    AlreadyRunningError,
    ErrorBackoff,
    PreservationError,
    RecoveryBlockedError,
    RUNTIME_SCHEMA,
    RotatingJevPolicy,
    SingleInstanceLock,
    UntilWinSupervisor,
    build_parser,
    make_env,
    observation_digest,
    prepare_legacy_resume_state,
    write_json_atomic,
)


def fake_observation(step=0):
    blstats = np.zeros(27, dtype=np.int64)
    blstats[nethack.NLE_BL_X] = 40
    blstats[nethack.NLE_BL_Y] = 10
    blstats[nethack.NLE_BL_HP] = 10
    blstats[nethack.NLE_BL_HPMAX] = 10
    blstats[nethack.NLE_BL_SCORE] = step
    blstats[nethack.NLE_BL_DEPTH] = 1
    blstats[nethack.NLE_BL_DNUM] = 0
    blstats[nethack.NLE_BL_DLEVEL] = 1
    blstats[nethack.NLE_BL_TIME] = step
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    chars[10, 40] = ord("@")
    tty = np.full((24, 80), ord(" "), dtype=np.uint8)
    message = np.zeros(256, dtype=np.uint8)
    encoded = f"step {step}".encode()
    message[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return {
        "chars": chars,
        "tty_chars": tty,
        "blstats": blstats,
        "message": message,
        "inv_letters": np.zeros(55, dtype=np.uint8),
        "inv_strs": np.zeros((55, 80), dtype=np.uint8),
    }


class FakeEnv:
    def __init__(self, outcomes):
        self.unwrapped = self
        self.actions = nethack.ACTIONS
        self.outcomes = list(outcomes)
        self.seed_values = None
        self.closed = False
        self.index = 0

    def seed(self, **values):
        self.seed_values = values

    def reset(self):
        return fake_observation(0), {}

    def step(self, action):
        outcome = self.outcomes[self.index]
        self.index += 1
        return (
            fake_observation(self.index),
            outcome.get("reward", 1.0),
            outcome.get("terminated", True),
            outcome.get("truncated", False),
            {
                "is_ascended": outcome.get("is_ascended", False),
                "end_status": SimpleNamespace(name=outcome.get("end_status", "DEATH")),
            },
        )

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def choose(self, state, criteria, instructions):
        self.calls += 1
        if self.calls <= self.failures:
            raise JevTransportError("offline test failure")
        return {
            "choice": "a0",
            "confidence": 1.0,
            "probabilities": {key: 1.0 if key == "a0" else 0.0 for key in criteria},
            "model": "fake-jev",
            "request_digest": "abc",
            "usage": {"input_tokens": 1},
        }


class FakeBoundedClient:
    def __init__(self, max_calls, max_input_bytes, api_key):
        self.max_calls = max_calls
        self.max_input_bytes = max_input_bytes
        self.api_key = api_key
        self.calls_used = 0
        self.input_bytes_used = 0

    def choose(self, state, criteria, instructions):
        if self.calls_used >= self.max_calls:
            raise AssertionError("wrapper should rotate before exhausted client is called")
        self.calls_used += 1
        self.input_bytes_used += 10
        return {"choice": "a0", "model": "fake", "usage": {"input_tokens": 1}}


class FakeRecorder:
    def __init__(self):
        self.frames = []
        self.decisions = []
        self.actions = 0
        self.telemetry = []
        self.statuses = []

    def wait_for_capacity(self, *, should_stop):
        return not should_stop()

    def observed_frame(self, **values):
        self.frames.append(values)

    def decision(self, **values):
        self.decisions.append(values)

    def set_telemetry(self, **values):
        self.telemetry.append(values)

    def report_status(self, phase, **values):
        self.statuses.append({"phase": phase, **values})

    def action_finished(self):
        self.actions += 1

    def should_finalize_segment(self):
        return False

    def finalize_segment(self):
        raise AssertionError("unexpected segment finalization")


class UntilWinTests(unittest.TestCase):
    def test_explicit_publisher_tombstone_root_is_optional(self):
        self.assertIsNone(build_parser().parse_args(["run"]).recording_tombstone_root)
        path = "/publisher-state/recording-tombstones"
        args = build_parser().parse_args(["run", "--recording-tombstone-root", path])
        self.assertEqual(args.recording_tombstone_root, Path(path))

    def make_supervisor(self, root, *, policy=None, outcomes=None, sleep=None):
        recorder = FakeRecorder()
        created = []

        def factory(_character, _limit, _episode_dir):
            env = FakeEnv(outcomes or [{"terminated": True, "is_ascended": True}])
            created.append(env)
            return env

        supervisor = UntilWinSupervisor(
            data_root=root,
            character="mon-hum-neu-mal",
            first_seed=101,
            engine_episode_limit=100,
            policy=policy or FakePolicy(),
            stop_event=threading.Event(),
            stop_file=root / "control" / "STOP",
            env_factory=factory,
            recorder=recorder,
            backoff=ErrorBackoff(2.0, 8.0, sleep=sleep or (lambda _seconds: None)),
        )
        return supervisor, recorder, created

    def test_terminal_metrics_bind_xlog_written_on_close_and_preserve_preterminal_score(self):
        class CloseFinalizedXlogEnv(FakeEnv):
            def __init__(self, episode_dir):
                super().__init__([{"terminated": True, "end_status": "DEATH"}])
                self.episode_dir = episode_dir

            def reset(self):
                return fake_observation(137), {}

            def step(self, action):
                self.index += 1
                return (
                    fake_observation(0),
                    11.5,
                    True,
                    False,
                    {"is_ascended": False, "end_status": SimpleNamespace(name="DEATH")},
                )

            def close(self):
                ttyrec = self.episode_dir / "nle-ttyrec"
                ttyrec.mkdir(parents=True, exist_ok=True)
                (ttyrec / "nle.4242.0.ttyrec3.bz2").write_bytes(b"retained terminal frames")
                (ttyrec / "nle.4242.xlogfile").write_text(
                    "version=3.6.7\tpoints=138\tdeath=died of starvation\twhile=fainted"
                    "\tturns=4732\tttyrecname=nle.4242.0.ttyrec3.bz2\n"
                )
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, _, created = self.make_supervisor(root)

            def factory(_character, _limit, episode_dir):
                env = CloseFinalizedXlogEnv(episode_dir)
                created.append(env)
                return env

            created.clear()
            supervisor.env_factory = factory
            state = supervisor.run(max_episodes=1)
            result = state["episodeResults"][-1]
            self.assertEqual("ready", state["status"])
            self.assertEqual(138, result["score"])
            self.assertEqual("official_final", result["scoreSemantics"])
            self.assertEqual(138, result["officialFinalScore"])
            self.assertEqual(137, result["lastObservedScore"])
            self.assertEqual(137, result["maxObservedScore"])
            self.assertEqual(11.5, result["totalReward"])
            self.assertEqual("DEATH", result["endStatus"])
            self.assertEqual("died of starvation", result["deathCause"])
            self.assertEqual("fainted", result["deathWhile"])
            self.assertTrue(created[0].closed)


    def test_replay_nonterminal_after_observation_updates_metrics_without_provider_call(self):
        class ScoreSeventeenStopEnv(FakeEnv):
            def __init__(self, stop_event):
                super().__init__([{"terminated": False, "reward": 5.0}])
                self.stop_event = stop_event

            def step(self, action):
                self.index += 1
                self.stop_event.set()
                return (
                    fake_observation(17),
                    5.0,
                    False,
                    False,
                    {"is_ascended": False, "end_status": SimpleNamespace(name="RUNNING")},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            first_policy = FakePolicy()
            first, _, _ = self.make_supervisor(root, policy=first_policy)
            first.env_factory = lambda *_args: ScoreSeventeenStopEnv(first.stop_event)
            first_state = first.run()
            self.assertEqual("paused", first_state["status"])
            self.assertEqual(1, first_policy.calls)
            self.assertEqual(17, first_state["episodeResults"][-1]["lastObservedScore"])

            second_policy = FakePolicy()
            resumed, _, _ = self.make_supervisor(root, policy=second_policy)
            resumed.env_factory = lambda *_args: ScoreSeventeenStopEnv(resumed.stop_event)
            resumed_state = resumed.run()
            result = resumed_state["episodeResults"][-1]
            self.assertEqual("paused", resumed_state["status"])
            self.assertEqual(0, second_policy.calls)
            self.assertEqual(17, result["lastObservedScore"])
            self.assertEqual(17, result["maxObservedScore"])
            self.assertEqual(5.0, result["totalReward"])


    def test_first_http_failure_after_replay_publishes_current_observation_without_advancing(self):
        class HTTP402Policy:
            def __init__(self):
                self.calls = 0

            def choose(self, state, criteria, instructions):
                self.calls += 1
                raise JevHTTPError(402)

        class CaptureLive:
            def __init__(self):
                self.payloads = []

            def submit(self, payload):
                self.payloads.append(dict(payload))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            first, _, _ = self.make_supervisor(
                root,
                outcomes=[{"terminated": False, "is_ascended": False}],
            )

            class StoppingEnv(FakeEnv):
                def step(inner_self, action):
                    result = super().step(action)
                    first.stop_event.set()
                    return result

            first.env_factory = lambda *_args: StoppingEnv(
                [{"terminated": False, "is_ascended": False}]
            )
            first.run()

            policy = HTTP402Policy()
            resumed, _, created = self.make_supervisor(
                root,
                policy=policy,
                outcomes=[{"terminated": False, "is_ascended": False}],
            )
            resumed.backoff = ErrorBackoff(2.0, 8.0, sleep=lambda _seconds: resumed.stop_event.set())
            recorder = BroadcastRecorder(
                Path(temporary) / "broadcast",
                stream_id="replay-http-402",
                ingest=None,
            )
            live = CaptureLive()
            recorder.ingest = object()
            recorder.live = live
            resumed.recorder = recorder
            try:
                with mock.patch(
                    "broadcast.render_terminal_png",
                    side_effect=lambda _terminal, path: path.write_bytes(b"png"),
                ):
                    state = resumed.run()
            finally:
                recorder.close()

            backoff = next(
                payload
                for payload in live.payloads
                if payload.get("runtime", {}).get("phase") == "provider_backoff"
            )
            observed = live.payloads[0]
            self.assertEqual("awaiting_decision", observed["phase"])
            self.assertEqual("awaiting_decision", observed["runtime"]["phase"])
            self.assertEqual(1, observed["step"])
            self.assertEqual(1, observed["state"]["player"]["score"])
            self.assertEqual(observed["capturedAt"], backoff["capturedAt"])
            self.assertEqual("JevHTTPError:402", backoff["runtime"]["errorType"])
            self.assertEqual(1, backoff["metrics"]["totalActions"])
            self.assertIsNone(backoff["decision"])
            self.assertIsNone(backoff["action"])
            self.assertEqual(1, policy.calls)
            self.assertEqual(1, created[0].index)
            self.assertEqual(1, state["totalActions"])
            self.assertEqual(0, recorder.action_count)
            log = "".join(path.read_text() for path in (root / "fragments").glob("*/transitions.jsonl"))
            self.assertIn('"errorType":"JevHTTPError:402"', log)


    def test_failed_training_close_retains_active_fragment_for_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, _, _ = self.make_supervisor(
                root,
                outcomes=[{"terminated": False, "is_ascended": False}],
            )

            class StoppingEnv(FakeEnv):
                def step(inner_self, action):
                    result = super().step(action)
                    supervisor.stop_event.set()
                    return result

            supervisor.env_factory = lambda *_args: StoppingEnv(
                [{"terminated": False, "is_ascended": False}]
            )
            original_close = TransitionPackWriter.close
            with mock.patch.object(
                TransitionPackWriter,
                "close",
                side_effect=TransitionPackError("simulated close failure"),
            ):
                state = supervisor.run()

            self.assertEqual("recovery_blocked", state["status"])
            self.assertIsNotNone(state["activeFragment"])
            self.assertEqual(
                state["activeFragment"],
                json.loads((root / "state.json").read_text())["activeFragment"],
            )
            original_close(supervisor.training, completed=False, reason="test_cleanup")


    def test_video_backlog_stop_spends_no_decision_and_executes_no_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            policy = FakePolicy()
            supervisor, recorder, created = self.make_supervisor(root, policy=policy)

            def full_backlog(*, should_stop):
                self.assertFalse(should_stop())
                self.assertEqual(policy.calls, 0)
                self.assertEqual(created[0].index, 0)
                supervisor.stop_event.set()
                return False

            recorder.wait_for_capacity = full_backlog
            state = supervisor.run()
            self.assertEqual(policy.calls, 0)
            self.assertEqual(created[0].index, 0)
            self.assertEqual(state["totalActions"], 0)

    def test_video_capacity_checked_before_each_new_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            policy = FakePolicy()
            supervisor, recorder, created = self.make_supervisor(
                root, policy=policy, outcomes=[{"terminated": False}, {"terminated": True, "is_ascended": True}])
            checks = []

            def room_available(*, should_stop):
                checks.append((policy.calls, created[0].index))
                return not should_stop()

            recorder.wait_for_capacity = room_available
            state = supervisor.run()
            self.assertEqual(checks, [(0, 0), (1, 1)])
            self.assertEqual(state["totalActions"], 2)

    def test_recording_failure_after_commit_does_not_invent_pending_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, recorder, created = self.make_supervisor(root, outcomes=[{"terminated": False}])
            observed = recorder.observed_frame

            def fail_after_commit(**values):
                if values["phase"] == "after_action":
                    raise OSError(28, "synthetic video storage exhaustion")
                return observed(**values)

            recorder.observed_frame = fail_after_commit
            with self.assertRaises(PreservationError): supervisor.run()
            saved = json.loads((root / "state.json").read_text())
            self.assertEqual(saved["fatalLatch"]["reason"], "post_commit_recording_failure")
            self.assertNotIn("lastIncompleteTransition", saved)
            self.assertEqual(saved["resumeCandidate"]["stepsCommitted"], 1)
            self.assertEqual(saved["totalActions"], 1)
            rows = [json.loads(line) for path in (root / "fragments").glob("*/transitions.jsonl") for line in path.read_text().splitlines()]
            commits = [row for row in rows if row["eventType"] == "transition_committed"]
            self.assertEqual([row["step"] for row in commits], [0])
            self.assertEqual(created[0].index, 1)
            # Model the misleading diagnostic left by the previous runtime.
            saved["lastIncompleteTransition"] = {
                key: commits[0][key] for key in ("eventId", "episodeId", "seed", "actionIndex", "keycode")}
            saved["lastIncompleteTransition"]["step"] = 1
            write_json_atomic(root / "state.json", saved)
            policy = FakePolicy()
            resumed, next_recorder, next_created = self.make_supervisor(root, policy=policy, outcomes=[{"terminated": False}])

            def stop_after_replay(*, should_stop):
                self.assertEqual(next_created[0].index, 1)
                self.assertEqual(policy.calls, 0)
                resumed.stop_event.set()
                return False

            next_recorder.wait_for_capacity = stop_after_replay
            recovered = resumed.run()
            self.assertEqual(recovered["totalActions"], 1)
            self.assertEqual(recovered["lastRecovery"]["actionsReplayed"], 1)
            self.assertEqual(recovered["lastRecovery"]["providerCalls"], 0)
            self.assertIsNone(recovered["fatalLatch"])
            self.assertNotIn("lastIncompleteTransition", recovered)
            self.assertEqual(recovered["lastResolvedIncompleteTransition"]["original"], saved["lastIncompleteTransition"])

    def test_verified_prefix_does_not_resolve_a_later_pending_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor, _, _ = self.make_supervisor(Path(temporary) / "data")
            marker = {"eventId": "later", "episodeId": 0, "seed": 101, "actionIndex": 0, "keycode": 107}
            supervisor.state = {"lastIncompleteTransition": marker.copy()}
            supervisor._resolve_replayed_incomplete([{**marker, "eventId": "earlier", "step": 0}])
            supervisor.recovery_log.close()
            self.assertEqual(supervisor.state["lastIncompleteTransition"], marker)
            self.assertNotIn("lastResolvedIncompleteTransition", supervisor.state)

    def test_verified_ascension_stops_and_persists_complete_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, recorder, created = self.make_supervisor(root)
            state = supervisor.run()
            self.assertEqual("won", state["status"])
            self.assertTrue((root / "WIN.json").exists())
            self.assertEqual(1, state["totalActions"])
            self.assertTrue(created[0].closed)
            self.assertEqual(1, recorder.actions)
            packs = list((root / "fragments").glob("*/training/*.npzpack"))
            self.assertEqual(1, len(packs))
            frames = list(scan_pack(packs[0]))
            self.assertEqual(1, len(frames))
            self.assertTrue(frames[0].metadata["verifiedAscension"])
            self.assertIn("obs__tty_chars", __import__("transition_pack").decode_transition(frames[0].payload)[0])

    def test_native_ascension_flag_without_termination_does_not_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            outcomes = [
                {"terminated": False, "is_ascended": True},
                {"terminated": True, "is_ascended": False},
            ]
            supervisor, _, _ = self.make_supervisor(root, outcomes=outcomes)
            state = supervisor.run(max_episodes=1)
            self.assertNotEqual("won", state["status"])
            self.assertFalse((root / "WIN.json").exists())
            self.assertEqual(2, state["totalActions"])

    def test_jev_error_uses_backoff_then_commits_one_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            sleeps = []
            recorder_ref = []

            def sleeping(seconds):
                self.assertEqual("provider_backoff", recorder_ref[0].statuses[-1]["phase"])
                self.assertIsNotNone(recorder_ref[0].statuses[-1]["retry_at"])
                sleeps.append(seconds)

            supervisor, _, _ = self.make_supervisor(
                root,
                policy=FakePolicy(failures=1),
                sleep=sleeping,
            )
            recorder_ref.append(supervisor.recorder)
            state = supervisor.run()
            self.assertEqual("won", state["status"])
            self.assertEqual(1, state["totalActions"])
            self.assertEqual(2.0, sum(sleeps))
            log = next((root / "fragments").glob("*/transitions.jsonl")).read_text()
            self.assertIn('"eventType":"jev_error"', log)
            self.assertIn('"backoffSeconds":2.0', log)

    def test_previous_running_episode_is_latched_for_same_game_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir(parents=True)
            write_json_atomic(
                root / "state.json",
                {
                    "schema": RUNTIME_SCHEMA,
                    "status": "running",
                    "createdAt": "earlier",
                    "updatedAt": "earlier",
                    "nextEpisodeId": 8,
                    "nextSeed": 108,
                    "totalActions": 4,
                    "completedEpisodes": 1,
                    "interruptedEpisodes": 0,
                    "activeEpisode": {"episodeId": 7, "seed": 107},
                    "activeFragment": "fragments/old",
                    "winningTransition": None,
                },
            )
            supervisor, _, _ = self.make_supervisor(root)
            self.assertEqual("recovery_pending", supervisor.state["status"])
            self.assertEqual(1, supervisor.state["interruptedEpisodes"])
            self.assertIsNone(supervisor.state["activeEpisode"])
            self.assertEqual(107, supervisor.state["resumeCandidate"]["seed"])
            supervisor.recovery_log.close()
            recovery = json.loads((root / "recovery.jsonl").read_text())
            self.assertEqual("deterministic_replay_pending", recovery["event"])

    def test_kernel_lock_rejects_second_instance_and_recovers_after_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.lock"
            first = SingleInstanceLock(path)
            second = SingleInstanceLock(path)
            first.acquire()
            with self.assertRaises(AlreadyRunningError):
                second.acquire()
            first.release()
            second.acquire()
            second.release()

    def test_jev_budget_rotation_does_not_require_an_episode_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = RotatingJevPolicy(
                api_key="offline-test-key",
                ledger_root=Path(temporary) / "ledger",
                max_calls_per_chunk=1,
                max_bytes_per_chunk=24 * 1024,
                client_factory=FakeBoundedClient,
            )
            first = policy.choose({}, {"a0": "wait"}, "choose")
            second = policy.choose({}, {"a0": "wait"}, "choose")
            policy.close()
            self.assertEqual(1, first["accountingChunk"])
            self.assertEqual(2, second["accountingChunk"])
            receipts = sorted((Path(temporary) / "ledger").glob("jev-chunk-*.json"))
            self.assertEqual(2, len(receipts))
            self.assertEqual("rotated", json.loads(receipts[0].read_text())["status"])

    def test_pending_intent_is_replayed_once_without_a_second_provider_call(self):
        class CrashAfterActionEnv(FakeEnv):
            def step(self, action):
                self.index += 1
                raise RuntimeError("simulated hard crash after env step")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            first_policy = FakePolicy()
            first, _, _ = self.make_supervisor(root, policy=first_policy)
            first.env_factory = lambda *_args: CrashAfterActionEnv([])
            with self.assertRaises(PreservationError):
                first.run()
            self.assertEqual(1, first_policy.calls)
            self.assertEqual("preservation_halted", json.loads((root / "state.json").read_text())["status"])

            resumed_policy = FakePolicy()
            resumed, _, created = self.make_supervisor(root, policy=resumed_policy)
            state = resumed.run()
            self.assertEqual("won", state["status"])
            self.assertEqual(0, resumed_policy.calls)
            self.assertEqual(1, created[0].index)
            self.assertEqual(1, state["totalActions"])
            events = "".join(path.read_text() for path in (root / "fragments").glob("*/transitions.jsonl"))
            self.assertEqual(1, events.count('"eventType":"pending_intent_replayed_once"'))

    def test_pack_before_commit_crash_recovers_committed_action_without_reasking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            first_policy = FakePolicy()
            first, _, _ = self.make_supervisor(root, policy=first_policy)
            append = first._append

            def crash_before_commit_event(value):
                if value.get("eventType") == "transition_committed":
                    raise OSError("simulated crash after pack fsync")
                append(value)

            first._append = crash_before_commit_event
            with self.assertRaises(PreservationError):
                first.run()
            resumed_policy = FakePolicy()
            resumed, _, _ = self.make_supervisor(root, policy=resumed_policy)
            state = resumed.run()
            self.assertEqual("won", state["status"])
            self.assertEqual(0, resumed_policy.calls)
            self.assertEqual(1, state["totalActions"])
            self.assertEqual(1, state["recoveries"])

    def test_clean_stop_resumes_same_episode_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            first, _, _ = self.make_supervisor(
                root,
                outcomes=[{"terminated": False, "is_ascended": False}],
            )

            class StoppingEnv(FakeEnv):
                def step(inner_self, action):
                    result = super().step(action)
                    first.stop_event.set()
                    return result

            first.env_factory = lambda *_args: StoppingEnv([{"terminated": False, "is_ascended": False}])
            paused = first.run()
            self.assertEqual("paused", paused["status"])
            self.assertEqual(101, paused["resumeCandidate"]["seed"])

            next_policy = FakePolicy()
            resumed, recorder, _ = self.make_supervisor(
                root,
                policy=next_policy,
                outcomes=[
                    {"terminated": False, "is_ascended": False},
                    {"terminated": True, "is_ascended": True},
                ],
            )
            won = resumed.run()
            self.assertEqual("won", won["status"])
            self.assertEqual(102, won["nextSeed"])
            self.assertEqual(2, won["totalActions"])
            self.assertEqual(1, next_policy.calls)
            self.assertTrue(any(item.get("metrics", {}).get("scope") == "continuous_run" for item in recorder.telemetry))
            self.assertTrue(all("criteria" in decision for decision in recorder.decisions))

    def test_real_nle_seed_replay_is_deterministic_and_uses_explicit_engine_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envs = [make_env("mon-hum-neu-mal", 12345, root / f"attempt-{index}") for index in range(2)]
            try:
                observations = []
                outcomes = []
                for env in envs:
                    env.unwrapped.seed(core=321, disp=100321, lgen=200321, reseed=False)
                    observation, _ = env.reset()
                    observations.append(observation_digest(observation))
                    next_observation, reward, terminated, truncated, info = env.step(0)
                    outcomes.append(
                        (
                            observation_digest(next_observation),
                            float(reward),
                            bool(terminated),
                            bool(truncated),
                            bool(info.get("is_ascended", False)),
                        )
                    )
                    self.assertEqual(12345, env.unwrapped._max_episode_steps)
                self.assertEqual(observations[0], observations[1])
                self.assertEqual(outcomes[0], outcomes[1])
            finally:
                for env in envs:
                    env.close()

    def test_legacy_import_preserves_trailing_before_state_and_prepares_same_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = {
                "stream_id": "verified-stream",
                "settings": {"seed": 103, "character": "mon-hum-neu-mal"},
            }
            (source / "config.json").write_text(json.dumps(config))
            before = {"player": {"score": 0}, "message": "before"}
            after = {"player": {"score": 1}, "message": "after"}
            events = [
                {"event_type": "observed_frame", "phase": "before_action", "episode": 0, "step": 0, "sequence": 1, "state": before},
                {"event_type": "decision", "episode": 0, "step": 0, "decision": {"choice": "a0", "model": "saved"}},
                {"event_type": "observed_frame", "phase": "after_action", "episode": 0, "step": 0, "sequence": 2, "state": after, "reward": 1.0, "action": {"index": 0, "keycode": int(nethack.ACTIONS[0]), "label": "saved label"}},
                {"event_type": "observed_frame", "phase": "before_action", "episode": 0, "step": 1, "sequence": 3, "state": after},
            ]
            (source / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
            state = prepare_legacy_resume_state(
                data_root=root / "until-win",
                source=source,
                episode=0,
                engine_episode_limit=1_000_000,
            )
            self.assertEqual("recovery_pending", state["status"])
            self.assertEqual(103, state["resumeCandidate"]["seed"])
            plan = json.loads((root / "until-win" / "recovery" / "legacy-replay-plan.json").read_text())
            self.assertEqual(after, plan["expectedFinalState"])
            self.assertEqual("saved label", plan["records"][0]["actionLabel"])
            self.assertFalse(plan["records"][0]["provenance"]["historicalRawArraysAvailable"])

    def test_source_contract_change_blocks_replay_before_any_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "run.py").write_text("version = 1\n")
            (source / "menu_labels.py").write_text("version = 1\n")
            supervisor, _, _ = self.make_supervisor(root / "data")
            supervisor.source_root = source
            episode_dir = root / "episode"
            episode_dir.mkdir()
            supervisor._prepare_initial_snapshot(
                global_dir=episode_dir,
                observation=fake_observation(0),
                seed=101,
                origin="native_full_observation",
            )
            (source / "run.py").write_text("version = 2\n")
            with self.assertRaisesRegex(RecoveryBlockedError, "source"):
                supervisor._prepare_initial_snapshot(
                    global_dir=episode_dir,
                    observation=fake_observation(0),
                    seed=101,
                    origin="native_full_observation",
                )
            supervisor.recovery_log.close()

    def test_winning_state_is_revalidated_from_immutable_pack_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, _, _ = self.make_supervisor(root)
            won = supervisor.run()
            self.assertEqual("won", won["status"])
            verifier, _, _ = self.make_supervisor(root)
            checked = verifier.run()
            self.assertEqual("won", checked["status"])
            self.assertIn("evidenceRevalidatedAt", checked["winningTransition"])

            marker_path = next((root / "fragments").glob("*/training/*.complete.json"))
            marker = json.loads(marker_path.read_text())
            marker["artifacts"][0]["sha256"] = "0" * 64
            marker_path.write_text(json.dumps(marker))
            blocked, _, _ = self.make_supervisor(root)
            with self.assertRaises(RecoveryBlockedError):
                blocked.run()

    def test_native_history_beyond_legacy_boundary_drops_stale_final_state_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, _, _ = self.make_supervisor(root)
            legacy_path = root / "recovery" / "legacy.json"
            legacy_path.parent.mkdir(parents=True)
            shared = {
                "step": 0,
                "actionIndex": 0,
                "keycode": int(nethack.ACTIONS[0]),
                "reward": 1.0,
                "terminated": False,
                "truncated": False,
            }
            legacy_path.write_text(
                json.dumps(
                    {
                        "schema": "jev-nethack-replay-plan/v1",
                        "origin": "deterministically_reconstructed_legacy",
                        "episodeId": 1,
                        "seed": 101,
                        "records": [shared],
                        "expectedFinalState": {"legacyBoundary": True},
                        "sources": [],
                    }
                )
            )
            training = root / "fragments" / "old" / "training"
            writer = TransitionPackWriter(training)
            for step in range(2):
                writer.write(
                    observation=fake_observation(step),
                    next_observation=fake_observation(step + 1),
                    metadata={
                        "eventId": f"e{step}",
                        "episodeId": 1,
                        "seed": 101,
                        "step": step,
                        "actionIndex": 0,
                        "keycode": int(nethack.ACTIONS[0]),
                        "reward": 1.0,
                        "terminated": False,
                        "truncated": False,
                        "isAscended": False,
                        "verifiedAscension": False,
                        "observationDigest": f"before-{step}",
                        "nextObservationDigest": f"after-{step}",
                    },
                )
            writer.close()
            plan = supervisor._load_candidate_plan(
                {
                    "episodeId": 1,
                    "seed": 101,
                    "character": "mon-hum-neu-mal",
                    "fragments": ["fragments/old"],
                    "legacyPlan": "recovery/legacy.json",
                }
            )
            self.assertEqual(2, len(plan["records"]))
            self.assertEqual([0, 1], plan["nativeSteps"])
            self.assertNotIn("expectedFinalState", plan)
            supervisor.recovery_log.close()

    def test_native_suffix_after_legacy_prefix_migrates_to_contiguous_same_game_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            supervisor, _, _ = self.make_supervisor(root)
            legacy_path = root / "recovery" / "legacy.json"
            legacy_path.parent.mkdir(parents=True)
            legacy_records = [
                {
                    "eventId": f"legacy-{step}",
                    "episodeId": 1,
                    "seed": 101,
                    "step": step,
                    "actionIndex": 0,
                    "keycode": int(nethack.ACTIONS[0]),
                    "reward": 1.0,
                    "terminated": False,
                    "truncated": False,
                    "provenance": {"origin": "deterministically_reconstructed_legacy"},
                }
                for step in range(2)
            ]
            legacy_path.write_text(
                json.dumps(
                    {
                        "schema": "jev-nethack-replay-plan/v1",
                        "origin": "deterministically_reconstructed_legacy",
                        "episodeId": 1,
                        "seed": 101,
                        "records": legacy_records,
                        "expectedFinalState": {"legacyBoundary": True},
                        "sources": [],
                    }
                )
            )
            training = root / "fragments" / "native-suffix" / "training"
            writer = TransitionPackWriter(training)
            for step in (2, 3):
                writer.write(
                    observation=fake_observation(step),
                    next_observation=fake_observation(step + 1),
                    metadata={
                        "eventId": f"native-{step}",
                        "episodeId": 1,
                        "seed": 101,
                        "step": step,
                        "actionIndex": 0,
                        "keycode": int(nethack.ACTIONS[0]),
                        "reward": 1.0,
                        "terminated": False,
                        "truncated": False,
                        "isAscended": False,
                        "verifiedAscension": False,
                        "observationDigest": f"before-{step}",
                        "nextObservationDigest": f"after-{step}",
                    },
                )
            writer.close()
            candidate = {
                "episodeId": 1,
                "seed": 101,
                "character": "mon-hum-neu-mal",
                "fragments": ["fragments/native-suffix"],
                "legacyPlan": "recovery/legacy.json",
            }
            plan = supervisor._load_candidate_plan(candidate)
            self.assertEqual([0, 1, 2, 3], [row["step"] for row in plan["records"]])
            self.assertEqual([2, 3], plan["nativeSteps"])
            self.assertNotIn("expectedFinalState", plan)
            supervisor.state["status"] = "paused"
            supervisor.state["resumeCandidate"] = candidate
            supervisor.stop_file.parent.mkdir(parents=True)
            supervisor.stop_file.touch()
            receipt = supervisor.migrate_recovery_evidence()
            self.assertEqual(4, receipt["actionsPreserved"])
            self.assertEqual([{"first": 2, "last": 3, "count": 2}], receipt["nativeEvidenceStepRanges"])
            self.assertEqual([{"first": 0, "last": 1, "count": 2}], receipt["legacyRawRegenerationStepRanges"])
            self.assertEqual(0, receipt["providerCalls"])
            self.assertFalse(receipt["freshSeedAllocated"])
            supervisor.recovery_log.close()


if __name__ == "__main__":
    unittest.main()
