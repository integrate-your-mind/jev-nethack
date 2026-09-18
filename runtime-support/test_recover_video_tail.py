"""Controlled local tests for crash-tail recording recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
UNTIL_WIN = HERE.parent / "until-win"
for path in (HERE, UNTIL_WIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from broadcast import probe_video, render_segment_ffmpeg, render_terminal_png
from recover_video_tail import (
    ActiveFragmentDeferred,
    ActivityEvidence,
    ExactProducerEvidence,
    NoRecoverableTail,
    TailRecoveryError,
    inspect_exact_producer,
    recover_video_tail,
)
import recover_video_tail as recovery_module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecoveryFixture:
    def __init__(
        self,
        root: Path,
        *,
        fragment: Path | None = None,
        stream_id: str = "abc123",
    ) -> None:
        self.root = root
        self.fragment = fragment or root / "recording"
        self.stream_id = stream_id
        self.frames = self.fragment / ".frames"
        self.frames.mkdir(parents=True)
        self.events = self.fragment / "segment-0001.jsonl"
        self.lock = root / "runtime.lock"
        self.lock.write_text("lock\n")
        dead = subprocess.Popen(["/usr/bin/true"])
        self.dead_pid = dead.pid
        dead.wait(timeout=5)
        self.pid_file = root / "writer.json"
        self.pid_file.write_text(json.dumps({"pid": self.dead_pid}) + "\n")
        self.writer = root / "current-writer.json"
        self.writer.write_text(json.dumps({"activeFragment": None}) + "\n")
        self.write_valid_source()

    @property
    def evidence(self) -> ActivityEvidence:
        return ActivityEvidence(
            pid_files=(self.pid_file,),
            lock_files=(self.lock,),
            current_writer_files=(self.writer,),
        )

    def write_valid_source(self) -> None:
        events = [
            {
                "event_type": "observed_frame",
                "phase": "before_action",
                "sequence": 40,
                "wall_epoch": 1789689600.0,
                "wall_time": "2026-09-18T00:00:00+00:00",
                "stream_id": self.stream_id,
                "action": None,
            },
            {
                "event_type": "decision",
                "wall_epoch": 1789689600.05,
                "wall_time": "2026-09-18T00:00:00.050000+00:00",
                "stream_id": self.stream_id,
            },
            {
                "event_type": "observed_frame",
                "phase": "after_action",
                "sequence": 41,
                "wall_epoch": 1789689600.2,
                "wall_time": "2026-09-18T00:00:00.200000+00:00",
                "stream_id": self.stream_id,
                "action": {"index": 3, "keycode": 104},
            },
        ]
        self.events.write_text("".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events))
        render_terminal_png("before", self.frames / "segment-0000-frame-00000040.png")
        render_terminal_png("after", self.frames / "segment-0000-frame-00000041.png")


def source_tree_snapshot(fragment: Path) -> dict[str, tuple[int, int, int, str]]:
    result: dict[str, tuple[int, int, int, str]] = {}
    for path in sorted(candidate for candidate in fragment.rglob("*") if candidate.is_file()):
        metadata = path.stat()
        result[str(path.relative_to(fragment))] = (
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            digest(path),
        )
    return result


class ExactProducerFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_root = root / "data"
        self.stamp = "20260918T000000.000001Z"
        self.stream_id = "a" * 32
        self.fragment = (
            self.data_root
            / "broadcast-fragments"
            / f"broadcast-{self.stamp}-{self.stream_id}"
        )
        self.recording = RecoveryFixture(
            root,
            fragment=self.fragment,
            stream_id=self.stream_id,
        )
        self.dead_pid = self.recording.dead_pid
        self.accounting_root = self.data_root / "jev-accounting"
        self.accounting_root.mkdir()
        self.accounting = (
            self.accounting_root
            / f"process-{self.stamp}-pid{self.dead_pid}"
        )
        self.accounting.mkdir()
        self.fragments_root = self.data_root / "fragments"
        self.current_fragment = self.fragments_root / "fragment-current"
        self.current_fragment.mkdir(parents=True)
        self.current_config = self.current_fragment / "config.json"
        self.current_config.write_text(json.dumps({"pid": os.getpid()}) + "\n")
        self.state = self.data_root / "state.json"
        self.state.write_text(
            json.dumps({"activeFragment": "fragments/fragment-current"}) + "\n"
        )
        self.derived_root = root / "publisher-derived"
        self.derived_root.mkdir()
        self.output = self.derived_root / "recovered-broadcast"

    @property
    def evidence(self) -> ExactProducerEvidence:
        return ExactProducerEvidence(self.data_root)

    def recover(self, **kwargs: object) -> dict[str, object]:
        return recover_video_tail(
            self.fragment,
            exact_producer=self.evidence,
            output_fragment=self.output,
            derived_root=self.derived_root,
            **kwargs,
        )


class RecoverVideoTailTests(unittest.TestCase):
    def test_python_only_iso_timestamp_forms_refuse(self) -> None:
        for value in (
            "2026-W38-5T00:00:00+00:00",
            "20260918T000000+00:00",
            "2026-09-18T00:00:00+00:00:30",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(TailRecoveryError, "RFC 3339"):
                recovery_module._parse_time(value, field="wall_time")

    def test_inactive_tail_recovers_original_frames_and_site_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            original_jsonl = digest(fixture.events)
            original_frames = {path.name: digest(path) for path in fixture.frames.glob("*.png")}

            result = recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            self.assertEqual(result["status"], "recovered")
            self.assertFalse(result["uploaded"])
            self.assertTrue(result["originalFramesRetained"])
            manifest = result["manifest"]
            self.assertTrue(manifest["completed"])
            self.assertTrue(manifest["recoveredAfterCrash"])
            self.assertEqual(manifest["gameOutcome"], "interrupted")
            self.assertEqual(manifest["frameCount"], 2)
            self.assertEqual(manifest["actionCount"], 1)
            self.assertEqual(manifest["firstSequence"], 40)
            self.assertEqual(manifest["lastSequence"], 41)
            self.assertEqual(manifest["sessionId"], "broadcast-abc123-seg0001")
            self.assertEqual(digest(fixture.events), original_jsonl)
            self.assertEqual({path.name: digest(path) for path in fixture.frames.glob("*.png")}, original_frames)

            mp4 = fixture.fragment / "segment-0001.mp4"
            media = probe_video(mp4)
            self.assertEqual(media["stream"]["codec_type"], "video")
            for artifact in manifest["artifacts"]:
                path = fixture.fragment / artifact["filename"]
                self.assertEqual(artifact["bytes"], path.stat().st_size)
                self.assertEqual(artifact["sha256"], digest(path))

    def test_any_active_signal_refuses_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            active_pid = ActivityEvidence(
                pids=(os.getpid(),),
                lock_files=(fixture.lock,),
                current_writer_files=(fixture.writer,),
            )
            with self.subTest(name="pid"), self.assertRaisesRegex(TailRecoveryError, "PID"):
                recover_video_tail(fixture.fragment, evidence=active_pid)

            fixture.writer.write_text(json.dumps({"activeFragment": str(fixture.fragment)}) + "\n")
            with self.subTest(name="current-writer"), self.assertRaisesRegex(TailRecoveryError, "current-writer"):
                recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            fixture.writer.write_text(json.dumps({"activeFragment": None}) + "\n")
            lock_stream = fixture.lock.open("r+b")
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.subTest(name="kernel-lock"), self.assertRaisesRegex(TailRecoveryError, "lock"):
                    recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
                lock_stream.close()
            self.assertFalse((fixture.fragment / "segment-0001.mp4").exists())
            self.assertFalse((fixture.fragment / "segment-0001.manifest.json").exists())

    def test_valid_existing_segment_is_a_byte_stable_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            first = recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            mp4 = fixture.fragment / "segment-0001.mp4"
            manifest = fixture.fragment / "segment-0001.manifest.json"
            before = (digest(mp4), digest(manifest), mp4.stat().st_mtime_ns, manifest.stat().st_mtime_ns)
            renderer = mock.Mock(side_effect=AssertionError("rendered"))
            second = recover_video_tail(fixture.fragment, evidence=fixture.evidence, render=renderer)
            after = (digest(mp4), digest(manifest), mp4.stat().st_mtime_ns, manifest.stat().st_mtime_ns)
            self.assertEqual(first["manifest"], second["manifest"])
            self.assertEqual(second["status"], "already_complete")
            self.assertEqual(before, after)
            renderer.assert_not_called()

    def test_self_consistent_but_unrelated_manifest_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            manifest_path = fixture.fragment / "segment-0001.manifest.json"
            unrelated = json.loads(manifest_path.read_text())
            unrelated["sessionId"] = "unrelated-session"
            unrelated["broadcastId"] = "unrelated"
            manifest_path.write_text(json.dumps(unrelated, separators=(",", ":")))
            before = (digest(fixture.fragment / "segment-0001.mp4"), digest(manifest_path))

            with self.assertRaisesRegex(TailRecoveryError, "does not match"):
                recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            self.assertEqual(
                (digest(fixture.fragment / "segment-0001.mp4"), digest(manifest_path)),
                before,
            )

    def test_boolean_schema_version_is_not_mistaken_for_site_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            manifest_path = fixture.fragment / "segment-0001.manifest.json"
            invalid = json.loads(manifest_path.read_text())
            invalid["schemaVersion"] = True
            manifest_path.write_text(json.dumps(invalid, separators=(",", ":")))

            result = recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            self.assertEqual(result["status"], "recovered")
            self.assertEqual(result["manifest"]["schemaVersion"], 1)
            self.assertIs(type(result["manifest"]["schemaVersion"]), int)

    def test_nonstandard_json_constant_is_not_mistaken_for_site_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            manifest_path = fixture.fragment / "segment-0001.manifest.json"
            invalid = json.loads(manifest_path.read_text())
            invalid["ignoredExtra"] = float("nan")
            manifest_path.write_text(json.dumps(invalid, separators=(",", ":")))
            self.assertIn("NaN", manifest_path.read_text())

            result = recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            self.assertEqual(result["status"], "recovered")
            self.assertNotIn("NaN", manifest_path.read_text())

    def test_corrupt_or_incomplete_source_refuses(self) -> None:
        def remove_observed_stream_id(fixture: RecoveryFixture) -> None:
            lines = fixture.events.read_text().splitlines()
            first = json.loads(lines[0])
            del first["stream_id"]
            lines[0] = json.dumps(first, separators=(",", ":"))
            fixture.events.write_text("\n".join(lines) + "\n")

        mutations = {
            "open-jsonl": lambda fixture: fixture.events.write_bytes(fixture.events.read_bytes().rstrip(b"\n")),
            "bad-json": lambda fixture: fixture.events.write_text("{bad}\n"),
            "missing-frame": lambda fixture: next(fixture.frames.glob("*.png")).unlink(),
            "corrupt-frame": lambda fixture: next(fixture.frames.glob("*.png")).write_bytes(b"not png"),
            "missing-observed-stream": remove_observed_stream_id,
            "nonstandard-json-constant": lambda fixture: fixture.events.write_text(
                fixture.events.read_text().replace('"action":null', '"extra":NaN,"action":null', 1)
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = RecoveryFixture(Path(temporary))
                mutate(fixture)
                with self.assertRaises(TailRecoveryError):
                    recover_video_tail(fixture.fragment, evidence=fixture.evidence)
                self.assertFalse((fixture.fragment / "segment-0001.manifest.json").exists())

    def test_wrong_segment_frame_prefix_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            for path in list(fixture.frames.glob("*.png")):
                path.rename(path.with_name(path.name.replace("segment-0000-", "segment-9999-")))
            with self.assertRaisesRegex(TailRecoveryError, "retained PNG"):
                recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            self.assertFalse((fixture.fragment / "segment-0001.mp4").exists())
            self.assertFalse((fixture.fragment / "segment-0001.manifest.json").exists())

    def test_corrupt_partial_outputs_are_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            mp4 = fixture.fragment / "segment-0001.mp4"
            manifest = fixture.fragment / "segment-0001.manifest.json"
            mp4.write_bytes(b"partial")
            manifest.write_text("{partial")
            result = recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            self.assertEqual(result["status"], "recovered")
            self.assertGreater(float(probe_video(mp4)["format"]["duration"]), 0)
            self.assertTrue(json.loads(manifest.read_text())["completed"])

    def test_dangling_derived_output_symlink_refuses_without_replacement(self) -> None:
        for filename in ("segment-0001.mp4", "segment-0001.manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                fixture = RecoveryFixture(Path(temporary))
                target = fixture.fragment / filename
                target.symlink_to(fixture.root / "missing-target")
                with self.assertRaisesRegex(TailRecoveryError, "non-symlink"):
                    recover_video_tail(fixture.fragment, evidence=fixture.evidence)
                self.assertTrue(target.is_symlink())
                other = fixture.fragment / (
                    "segment-0001.manifest.json" if filename.endswith(".mp4") else "segment-0001.mp4"
                )
                self.assertFalse(other.exists())

    def test_matching_valid_mp4_is_reused_after_manifest_only_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            mp4 = fixture.fragment / "segment-0001.mp4"
            manifest = fixture.fragment / "segment-0001.manifest.json"
            original = (digest(mp4), mp4.stat().st_ino)
            manifest.unlink()
            result = recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            self.assertTrue(result["reusedMatchingMp4"])
            self.assertEqual((digest(mp4), mp4.stat().st_ino), original)
            self.assertTrue(manifest.exists())

    def test_source_change_during_render_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))

            def mutate_after_render(frames: list[tuple[Path, float]], output: Path) -> None:
                render_segment_ffmpeg(frames, output)
                with fixture.events.open("ab") as stream:
                    stream.write(b"{}\n")

            with self.assertRaisesRegex(TailRecoveryError, "source changed"):
                recover_video_tail(fixture.fragment, evidence=fixture.evidence, render=mutate_after_render)
            self.assertFalse((fixture.fragment / "segment-0001.mp4").exists())
            self.assertFalse((fixture.fragment / "segment-0001.manifest.json").exists())

    def test_activity_evidence_change_during_render_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))

            def activate_after_render(frames: list[tuple[Path, float]], output: Path) -> None:
                render_segment_ffmpeg(frames, output)
                fixture.writer.write_text(json.dumps({"activeFragment": str(fixture.fragment)}) + "\n")

            with self.assertRaisesRegex(TailRecoveryError, "source changed|became active"):
                recover_video_tail(fixture.fragment, evidence=fixture.evidence, render=activate_after_render)
            self.assertFalse((fixture.fragment / "segment-0001.mp4").exists())
            self.assertFalse((fixture.fragment / "segment-0001.manifest.json").exists())

    def test_activity_change_after_mp4_commit_withholds_manifest_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            original_commit = recovery_module._commit_temp

            def commit_then_activate(source: Path, target: Path, expected: object, **kwargs: object) -> None:
                original_commit(source, target, expected, **kwargs)
                if target.suffix == ".mp4":
                    fixture.writer.write_text(json.dumps({"activeFragment": str(fixture.fragment)}) + "\n")

            with mock.patch.object(recovery_module, "_commit_temp", side_effect=commit_then_activate):
                with self.assertRaisesRegex(TailRecoveryError, "source changed|became active"):
                    recover_video_tail(fixture.fragment, evidence=fixture.evidence)

            mp4 = fixture.fragment / "segment-0001.mp4"
            manifest = fixture.fragment / "segment-0001.manifest.json"
            self.assertEqual(probe_video(mp4)["stream"]["codec_type"], "video")
            self.assertFalse(manifest.exists())
            fixture.writer.write_text(json.dumps({"activeFragment": None}) + "\n")
            resumed = recover_video_tail(fixture.fragment, evidence=fixture.evidence)
            self.assertTrue(resumed["reusedMatchingMp4"])
            self.assertTrue(manifest.exists())


class ExactProducerRecoveryTests(unittest.TestCase):
    def test_exact_mapping_recovers_into_separate_output_without_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            before = source_tree_snapshot(fixture.fragment)

            identity = inspect_exact_producer(fixture.fragment, fixture.evidence)
            result = fixture.recover()

            self.assertEqual(identity.stamp, fixture.stamp)
            self.assertEqual(identity.pid, fixture.dead_pid)
            self.assertEqual(identity.stream_id, fixture.stream_id)
            self.assertEqual(result["status"], "recovered")
            self.assertTrue(result["copiedJsonl"])
            self.assertTrue(result["sourceUnmodifiedByRecovery"])
            self.assertEqual(source_tree_snapshot(fixture.fragment), before)
            self.assertFalse((fixture.fragment / "segment-0001.mp4").exists())
            self.assertFalse(
                (fixture.fragment / "segment-0001.manifest.json").exists()
            )

            copied_events = fixture.output / "segment-0001.jsonl"
            self.assertEqual(copied_events.read_bytes(), fixture.recording.events.read_bytes())
            self.assertNotEqual(
                copied_events.stat().st_ino,
                fixture.recording.events.stat().st_ino,
            )
            mp4 = fixture.output / "segment-0001.mp4"
            self.assertEqual(probe_video(mp4)["stream"]["codec_type"], "video")
            manifest = json.loads(
                (fixture.output / "segment-0001.manifest.json").read_text()
            )
            self.assertTrue(manifest["recoveredAfterCrash"])
            self.assertEqual(manifest["gameOutcome"], "interrupted")
            self.assertFalse(result["uploaded"])
            for artifact in manifest["artifacts"]:
                artifact_path = fixture.output / artifact["filename"]
                self.assertEqual(artifact["bytes"], artifact_path.stat().st_size)
                self.assertEqual(artifact["sha256"], digest(artifact_path))

    def test_missing_ambiguous_alive_and_current_mappings_are_distinct_refusals(self) -> None:
        with self.subTest(case="missing"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.accounting.rmdir()
            with self.assertRaisesRegex(TailRecoveryError, "requires one"):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

        with self.subTest(case="ambiguous"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            second = subprocess.Popen(["/usr/bin/true"])
            second_pid = second.pid
            second.wait(timeout=5)
            (
                fixture.accounting_root
                / f"process-{fixture.stamp}-pid{second_pid}"
            ).mkdir()
            with self.assertRaisesRegex(TailRecoveryError, "requires one"):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

        with self.subTest(case="alive"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.accounting.rmdir()
            (
                fixture.accounting_root
                / f"process-{fixture.stamp}-pid{os.getpid()}"
            ).mkdir()
            with self.assertRaises(ActiveFragmentDeferred):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

        with self.subTest(case="current"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.current_config.write_text(
                json.dumps({"pid": fixture.dead_pid}) + "\n"
            )
            with self.assertRaises(ActiveFragmentDeferred):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

    def test_exact_mapping_rejects_symlink_and_stream_identity_mismatch(self) -> None:
        with self.subTest(case="accounting-symlink"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            real = fixture.accounting.with_name("unrelated")
            fixture.accounting.rename(real)
            fixture.accounting.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(TailRecoveryError, "non-symlink"):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

        with self.subTest(case="stream-mismatch"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.recording.events.write_text(
                fixture.recording.events.read_text().replace(
                    fixture.stream_id,
                    "b" * 32,
                )
            )
            with self.assertRaisesRegex(TailRecoveryError, "stream_id"):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

    def test_zero_frame_tail_is_deferred_without_output_but_malformed_is_hard(self) -> None:
        with self.subTest(case="zero"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.recording.events.write_bytes(b"")
            with self.assertRaises(NoRecoverableTail):
                fixture.recover()
            self.assertFalse(fixture.output.exists())

        with self.subTest(case="malformed"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.recording.events.write_bytes(b"{bad}\n")
            with self.assertRaises(TailRecoveryError) as caught:
                fixture.recover()
            self.assertNotIsInstance(caught.exception, NoRecoverableTail)
            self.assertFalse(fixture.output.exists())

    def test_producer_becoming_active_during_preflight_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            with mock.patch.object(
                recovery_module,
                "_pid_is_alive",
                side_effect=[False, True],
            ):
                with self.assertRaises(ActiveFragmentDeferred):
                    fixture.recover()
            self.assertFalse(fixture.output.exists())

    def test_source_or_mapping_change_during_render_withholds_completed_outputs(self) -> None:
        with self.subTest(case="source"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))

            def mutate_source(
                frames: list[tuple[Path, float]],
                output: Path,
            ) -> None:
                render_segment_ffmpeg(frames, output)
                with fixture.recording.events.open("ab") as stream:
                    stream.write(b"{}\n")

            with self.assertRaisesRegex(TailRecoveryError, "source changed"):
                fixture.recover(render=mutate_source)
            self.assertFalse((fixture.output / "segment-0001.mp4").exists())
            self.assertFalse(
                (fixture.output / "segment-0001.manifest.json").exists()
            )

        with self.subTest(case="mapping"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            before = source_tree_snapshot(fixture.fragment)

            def activate_old_producer(
                frames: list[tuple[Path, float]],
                output: Path,
            ) -> None:
                render_segment_ffmpeg(frames, output)
                fixture.current_config.write_text(
                    json.dumps({"pid": fixture.dead_pid}) + "\n"
                )

            with self.assertRaises(ActiveFragmentDeferred):
                fixture.recover(render=activate_old_producer)
            self.assertEqual(source_tree_snapshot(fixture.fragment), before)
            self.assertFalse((fixture.output / "segment-0001.mp4").exists())
            self.assertFalse(
                (fixture.output / "segment-0001.manifest.json").exists()
            )

    def test_mapping_change_after_mp4_commit_withholds_manifest_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            source_before = source_tree_snapshot(fixture.fragment)
            original_commit = recovery_module._commit_temp

            def commit_then_activate(
                source: Path,
                target: Path,
                expected: object,
                **kwargs: object,
            ) -> None:
                original_commit(source, target, expected, **kwargs)
                if target.suffix == ".mp4":
                    fixture.current_config.write_text(
                        json.dumps({"pid": fixture.dead_pid}) + "\n"
                    )

            with mock.patch.object(
                recovery_module,
                "_commit_temp",
                side_effect=commit_then_activate,
            ):
                with self.assertRaises(ActiveFragmentDeferred):
                    fixture.recover()

            mp4 = fixture.output / "segment-0001.mp4"
            manifest = fixture.output / "segment-0001.manifest.json"
            self.assertEqual(probe_video(mp4)["stream"]["codec_type"], "video")
            self.assertFalse(manifest.exists())
            fixture.current_config.write_text(json.dumps({"pid": os.getpid()}) + "\n")

            resumed = fixture.recover()

            self.assertTrue(resumed["reusedMatchingMp4"])
            self.assertTrue(manifest.exists())
            self.assertEqual(source_tree_snapshot(fixture.fragment), source_before)

    def test_destination_file_and_directory_races_do_not_overwrite(self) -> None:
        with self.subTest(case="file"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            competitor = b"concurrent writer"

            def create_competing_mp4(
                frames: list[tuple[Path, float]],
                output: Path,
            ) -> None:
                render_segment_ffmpeg(frames, output)
                (fixture.output / "segment-0001.mp4").write_bytes(competitor)

            with self.assertRaisesRegex(TailRecoveryError, "MP4 changed"):
                fixture.recover(render=create_competing_mp4)
            self.assertEqual(
                (fixture.output / "segment-0001.mp4").read_bytes(),
                competitor,
            )
            self.assertFalse(
                (fixture.output / "segment-0001.manifest.json").exists()
            )

        with self.subTest(case="directory"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            moved = fixture.derived_root / "moved-output"

            def swap_output_directory(
                frames: list[tuple[Path, float]],
                output: Path,
            ) -> None:
                render_segment_ffmpeg(frames, output)
                fixture.output.rename(moved)
                fixture.output.mkdir()

            with self.assertRaisesRegex(TailRecoveryError, "directory changed"):
                fixture.recover(render=swap_output_directory)
            self.assertEqual(list(fixture.output.iterdir()), [])
            self.assertFalse((moved / "segment-0001.mp4").exists())
            self.assertFalse((moved / "segment-0001.manifest.json").exists())

        with self.subTest(case="final-commit"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            competitor = b"final-window competitor"
            original_commit = recovery_module._commit_temp

            def inject_at_second_precommit(
                source: Path,
                target: Path,
                expected: object,
                **kwargs: object,
            ) -> None:
                if target.suffix != ".mp4":
                    original_commit(source, target, expected, **kwargs)
                    return
                original_precommit = kwargs["precommit"]
                calls = 0

                def precommit_with_race() -> None:
                    nonlocal calls
                    original_precommit()
                    calls += 1
                    if calls == 2:
                        target.write_bytes(competitor)

                kwargs["precommit"] = precommit_with_race
                original_commit(source, target, expected, **kwargs)

            with mock.patch.object(
                recovery_module,
                "_commit_temp",
                side_effect=inject_at_second_precommit,
            ):
                with self.assertRaisesRegex(TailRecoveryError, "appeared"):
                    fixture.recover()
            self.assertEqual(
                (fixture.output / "segment-0001.mp4").read_bytes(),
                competitor,
            )
            self.assertFalse(
                (fixture.output / "segment-0001.manifest.json").exists()
            )

    def test_output_is_byte_stable_idempotent_and_source_stays_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            source_before = source_tree_snapshot(fixture.fragment)
            first = fixture.recover()
            files = sorted(path for path in fixture.output.iterdir() if path.is_file())
            output_before = {
                path.name: (digest(path), path.stat().st_mtime_ns, path.stat().st_ino)
                for path in files
            }
            renderer = mock.Mock(side_effect=AssertionError("rendered"))

            second = fixture.recover(render=renderer)

            output_after = {
                path.name: (digest(path), path.stat().st_mtime_ns, path.stat().st_ino)
                for path in files
            }
            self.assertEqual(first["manifest"], second["manifest"])
            self.assertEqual(second["status"], "already_complete")
            self.assertEqual(output_before, output_after)
            self.assertEqual(source_tree_snapshot(fixture.fragment), source_before)
            renderer.assert_not_called()

    def test_output_escape_symlink_and_jsonl_conflict_refuse(self) -> None:
        with self.subTest(case="escape"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            other_root = fixture.root / "other"
            other_root.mkdir()
            with self.assertRaisesRegex(TailRecoveryError, "direct child"):
                recover_video_tail(
                    fixture.fragment,
                    exact_producer=fixture.evidence,
                    output_fragment=other_root / "outside",
                    derived_root=fixture.derived_root,
                )
            self.assertFalse((other_root / "outside").exists())

        with self.subTest(case="runtime-root"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            runtime_derived = fixture.data_root / "publisher-output"
            runtime_derived.mkdir()
            with self.assertRaisesRegex(TailRecoveryError, "runtime data root"):
                recover_video_tail(
                    fixture.fragment,
                    exact_producer=fixture.evidence,
                    output_fragment=runtime_derived / "recovered",
                    derived_root=runtime_derived,
                )
            self.assertFalse((runtime_derived / "recovered").exists())

        with self.subTest(case="symlink"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            target = fixture.root / "target"
            target.mkdir()
            fixture.output.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(TailRecoveryError, "non-symlink"):
                fixture.recover()
            self.assertEqual(list(target.iterdir()), [])

        with self.subTest(case="jsonl-conflict"), tempfile.TemporaryDirectory() as temporary:
            fixture = ExactProducerFixture(Path(temporary))
            fixture.output.mkdir()
            conflict = fixture.output / "segment-0001.jsonl"
            conflict.write_bytes(b"{\"different\":true}\n")
            before = conflict.read_bytes()
            with self.assertRaisesRegex(TailRecoveryError, "non-matching"):
                fixture.recover()
            self.assertEqual(conflict.read_bytes(), before)
            self.assertFalse((fixture.output / "segment-0001.mp4").exists())
            self.assertFalse(
                (fixture.output / "segment-0001.manifest.json").exists()
            )

        for filename in ("segment-0001.mp4", "segment-0001.manifest.json"):
            with self.subTest(case=f"invalid-{filename}"), tempfile.TemporaryDirectory() as temporary:
                fixture = ExactProducerFixture(Path(temporary))
                fixture.output.mkdir()
                conflict = fixture.output / filename
                conflict.write_bytes(b"invalid derived artifact")
                before = conflict.read_bytes()
                with self.assertRaisesRegex(TailRecoveryError, "refusing to replace"):
                    fixture.recover()
                self.assertEqual(conflict.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
