"""Tests for deterministic, source-preserving transition-tail recovery."""

from __future__ import annotations

import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
UNTIL_WIN = HERE.parent / "until-win"
for entry in (HERE, UNTIL_WIN):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from recover_training_tail import (  # noqa: E402
    TrainingTailRecoveryError,
    recover_training_tail,
)
import recover_training_tail as recovery_module  # noqa: E402
from transition_pack import (  # noqa: E402
    FRAME_HEADER,
    MAGIC,
    TransitionPackWriter,
    encode_transition,
    validate_shard,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecoveryFixture:
    def __init__(self, root: Path, *, event_ids: tuple[str, ...] = ("event-1", "event-2", "event-3")) -> None:
        self.root = root.resolve()
        root = self.root
        self.data_root = root / "source"
        self.fragments = self.data_root / "fragments"
        self.fragment = self.fragments / "fragment-old-pid-dead-abcd1234"
        self.training = self.fragment / "training"
        self.derived = root / "derived"
        self.fragments.mkdir(parents=True)
        self.fragment.mkdir()
        dead = subprocess.Popen(["/usr/bin/true"])
        self.dead_pid = dead.pid
        dead.wait(timeout=5)
        self.config = self.fragment / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema": "jev-nethack-until-win/v1",
                    "fragmentId": self.fragment.name,
                    "pid": self.dead_pid,
                    "startedAt": "2026-09-18T10:00:00+00:00",
                    "transitionSchema": "jev-nethack-transition-pack/v1",
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.state = self.data_root / "state.json"
        self.state.write_text(
            json.dumps(
                {
                    "schema": "jev-nethack-until-win/v1",
                    "activeFragment": "fragments/fragment-new-pid-live-ffff0000",
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.writer = TransitionPackWriter(self.training)
        for position, event_id in enumerate(event_ids):
            observation = {"glyphs": np.array([[position]], dtype=np.int16)}
            next_observation = {"glyphs": np.array([[position + 1]], dtype=np.int16)}
            self.writer.write(
                observation=observation,
                next_observation=next_observation,
                metadata={
                    "eventId": event_id,
                    "episodeId": 7,
                    "seed": 103,
                    "step": position,
                    "actionIndex": position + 1,
                    "keycode": 100 + position,
                    "observationDigest": f"obs-{position}",
                    "nextObservationDigest": f"next-{position}",
                    "reward": float(position),
                    "terminated": False,
                    "truncated": False,
                    "isAscended": False,
                    "verifiedAscension": False,
                    "endStatus": None,
                    "recordedAt": f"2026-09-18T10:00:0{position}+00:00",
                },
            )
        assert self.writer._pack is not None and self.writer._index is not None
        self.writer._pack.close()
        self.writer._index.close()
        self.writer._pack = None
        self.writer._index = None
        self.pack = self.training / "transitions-000001.npzpack"
        self.index = self.training / "transitions-000001.index.jsonl"

    def source_digests(self) -> dict[str, str]:
        return {str(path): digest(path) for path in (self.config, self.state, self.pack, self.index)}

    def output(self, suffix: str) -> Path:
        return self.derived / "fragments" / self.fragment.name / "training" / f"transitions-000001.{suffix}"


class RecoverTrainingTailTests(unittest.TestCase):
    def test_killed_multiframe_writer_recovers_standard_shard_without_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            before = fixture.source_digests()

            result = recover_training_tail(fixture.fragment, fixture.derived)

            self.assertEqual(result["status"], "recovered")
            self.assertTrue(result["sourceUnchanged"])
            self.assertEqual(fixture.source_digests(), before)
            self.assertEqual(fixture.output("npzpack").read_bytes(), fixture.pack.read_bytes())
            validated = validate_shard(fixture.output("npzpack"))
            self.assertTrue(validated["completed"])
            self.assertEqual(validated["records"], 3)
            copied_config = fixture.derived / "fragments" / fixture.fragment.name / "config.json"
            self.assertEqual(copied_config.read_bytes(), fixture.config.read_bytes())
            marker = json.loads(fixture.output("complete.json").read_text())
            self.assertTrue(marker["provenance"]["recoveredAfterCrash"])
            self.assertEqual(marker["provenance"]["indexedRecords"], 3)
            self.assertTrue(marker["provenance"]["payloadBytesUnchanged"])
            self.assertEqual(
                marker["provenance"]["sourcePaths"]["fragment"],
                f"fragments/{fixture.fragment.name}",
            )
            self.assertNotIn(str(fixture.root), json.dumps(marker["provenance"]["sourcePaths"]))

    def test_pack_ahead_of_index_rebuilds_every_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            fixture.index.write_bytes(fixture.index.read_bytes().splitlines(keepends=True)[0])

            result = recover_training_tail(fixture.fragment, fixture.derived)

            self.assertEqual(result["shards"][0]["records"], 3)
            self.assertEqual(result["shards"][0]["indexedRecords"], 1)
            records = [json.loads(line) for line in fixture.output("index.jsonl").read_text().splitlines()]
            self.assertEqual([row["eventId"] for row in records], ["event-1", "event-2", "event-3"])
            self.assertEqual([row["record"] for row in records], [1, 2, 3])
            self.assertEqual(validate_shard(fixture.output("npzpack"))["records"], 3)

    def test_incomplete_pack_and_index_tails_are_recorded_and_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            complete_pack = fixture.pack.read_bytes()
            complete_index_lines = fixture.index.read_bytes().splitlines(keepends=True)
            fixture.pack.write_bytes(complete_pack + b"\x00\x00\x00\x00")
            fixture.index.write_bytes(b"".join(complete_index_lines[:2]) + b'{"eventId":"event-3"')

            result = recover_training_tail(fixture.fragment, fixture.derived)

            shard = result["shards"][0]
            self.assertEqual(shard["packTailBytesDiscarded"], 4)
            self.assertEqual(shard["indexTailBytesDiscarded"], len(b'{"eventId":"event-3"'))
            self.assertEqual(fixture.output("npzpack").read_bytes(), complete_pack)
            self.assertEqual(validate_shard(fixture.output("npzpack"))["records"], 3)

    def test_active_pid_or_active_fragment_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            config = json.loads(fixture.config.read_text())
            config["pid"] = os.getpid()
            fixture.config.write_text(json.dumps(config) + "\n")
            with self.assertRaisesRegex(TrainingTailRecoveryError, "PID is still active"):
                recover_training_tail(fixture.fragment, fixture.derived)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            fixture.state.write_text(
                json.dumps(
                    {
                        "schema": "jev-nethack-until-win/v1",
                        "activeFragment": f"fragments/{fixture.fragment.name}",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(TrainingTailRecoveryError, "still the active fragment"):
                recover_training_tail(fixture.fragment, fixture.derived)

    def test_source_or_state_change_before_commit_refuses_without_marker(self) -> None:
        mutations = {
            "pack": lambda fixture: fixture.pack.write_bytes(fixture.pack.read_bytes() + b"x"),
            "state": lambda fixture: fixture.state.write_text(
                json.dumps(
                    {
                        "schema": "jev-nethack-until-win/v1",
                        "activeFragment": f"fragments/{fixture.fragment.name}",
                    }
                )
                + "\n"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = RecoveryFixture(Path(temporary))
                with self.assertRaisesRegex(TrainingTailRecoveryError, "changed|active"):
                    recover_training_tail(
                        fixture.fragment,
                        fixture.derived,
                        before_commit=lambda: mutate(fixture),
                    )
                self.assertFalse(fixture.output("complete.json").exists())

    def test_unrelated_live_state_updates_are_allowed_but_active_fragment_is_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))

            def advance_unrelated_state() -> None:
                state = json.loads(fixture.state.read_text())
                state.update({"totalActions": 42, "updatedAt": "2026-09-18T10:00:20+00:00"})
                fixture.state.write_text(json.dumps(state, sort_keys=True) + "\n")

            result = recover_training_tail(
                fixture.fragment, fixture.derived, before_commit=advance_unrelated_state
            )
            self.assertEqual(result["status"], "recovered")
            self.assertTrue(fixture.output("complete.json").is_file())

    def test_completed_rerun_is_byte_and_mtime_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            first = recover_training_tail(fixture.fragment, fixture.derived)
            outputs = [
                fixture.derived / "fragments" / fixture.fragment.name / "config.json",
                fixture.output("npzpack"),
                fixture.output("index.jsonl"),
                fixture.output("complete.json"),
            ]
            before = [(digest(path), path.stat().st_mtime_ns) for path in outputs]

            second = recover_training_tail(fixture.fragment, fixture.derived)

            self.assertEqual(first["shards"], second["shards"])
            self.assertEqual(second["status"], "already_recovered")
            self.assertEqual([(digest(path), path.stat().st_mtime_ns) for path in outputs], before)

    def test_corrupt_full_frame_or_duplicate_event_id_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            body = bytearray(fixture.pack.read_bytes())
            body[len(MAGIC) + FRAME_HEADER.size + 3] ^= 0x01
            fixture.pack.write_bytes(body)
            with self.assertRaisesRegex(TrainingTailRecoveryError, "digest mismatch"):
                recover_training_tail(fixture.fragment, fixture.derived)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary), event_ids=("same", "same"))
            with self.assertRaisesRegex(TrainingTailRecoveryError, "duplicate"):
                recover_training_tail(fixture.fragment, fixture.derived)

    def test_duplicate_event_ids_across_selected_shards_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            source = fixture.pack.read_bytes()
            payload_bytes, _ = FRAME_HEADER.unpack_from(source, len(MAGIC))
            frame_end = len(MAGIC) + FRAME_HEADER.size + payload_bytes
            (fixture.training / "transitions-000002.npzpack").write_bytes(
                MAGIC + source[len(MAGIC):frame_end]
            )
            record = json.loads(fixture.index.read_text().splitlines()[0])
            record["shard"] = 2
            (fixture.training / "transitions-000002.index.jsonl").write_text(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            with self.assertRaisesRegex(TrainingTailRecoveryError, "across shards"):
                recover_training_tail(
                    fixture.fragment, fixture.derived, shard_numbers=(1, 2)
                )

    def test_npz_decoded_resource_bound_refuses_high_expansion_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            payload = encode_transition(
                observation={
                    "huge": np.zeros(recovery_module.MAX_NPZ_MEMBER_BYTES + 1, dtype=np.uint8)
                },
                next_observation={"tiny": np.zeros(1, dtype=np.uint8)},
                metadata={
                    "eventId": "resource-bomb",
                    "recordedAt": "2026-09-18T10:00:00+00:00",
                },
            )
            frame_digest = hashlib.sha256(payload).digest()
            fixture.pack.write_bytes(MAGIC + FRAME_HEADER.pack(len(payload), frame_digest) + payload)
            with self.assertRaisesRegex(TrainingTailRecoveryError, "decoded size bound"):
                recover_training_tail(fixture.fragment, fixture.derived)

    def test_npy_header_cannot_lie_about_terabyte_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            npy = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                npy,
                {"descr": "|u1", "fortran_order": False, "shape": (1 << 40,)},
            )
            payload_buffer = io.BytesIO()
            with zipfile.ZipFile(payload_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("obs__bomb.npy", npy.getvalue())
            payload = payload_buffer.getvalue()
            frame_digest = hashlib.sha256(payload).digest()
            fixture.pack.write_bytes(MAGIC + FRAME_HEADER.pack(len(payload), frame_digest) + payload)

            with self.assertRaisesRegex(TrainingTailRecoveryError, "declared array"):
                recover_training_tail(fixture.fragment, fixture.derived)

    def test_malformed_state_or_config_and_conflicting_output_refuse(self) -> None:
        for target in ("state", "config"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                fixture = RecoveryFixture(Path(temporary))
                getattr(fixture, target).write_text("{bad\n")
                with self.assertRaises(TrainingTailRecoveryError):
                    recover_training_tail(fixture.fragment, fixture.derived)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            recover_training_tail(fixture.fragment, fixture.derived)
            fixture.output("index.jsonl").write_text("conflict\n")
            with self.assertRaisesRegex(TrainingTailRecoveryError, "conflicts"):
                recover_training_tail(fixture.fragment, fixture.derived)

    def test_dedicated_lock_serializes_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            fixture.derived.mkdir()
            lock = fixture.derived / ".recover-training-tail.lock"
            lock.touch()
            with lock.open("r+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(TrainingTailRecoveryError, "holds the lock"):
                    recover_training_tail(fixture.fragment, fixture.derived)

    def test_custom_lock_parent_rebind_refuses_before_marker_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            lock_parent = fixture.root / "custom-lock-parent"
            lock_parent.mkdir()
            lock_path = lock_parent / "recovery.lock"
            displaced = fixture.root / "custom-lock-parent-before-swap"
            second_streams = []

            def rebind_and_take_same_lexical_lock() -> None:
                lock_parent.rename(displaced)
                lock_parent.mkdir()
                stream = lock_path.open("w+b")
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                second_streams.append(stream)

            try:
                with self.assertRaisesRegex(TrainingTailRecoveryError, "component changed"):
                    recover_training_tail(
                        fixture.fragment,
                        fixture.derived,
                        lock_path=lock_path,
                        before_commit=rebind_and_take_same_lexical_lock,
                    )
                self.assertTrue(second_streams)
                self.assertFalse(fixture.output("complete.json").exists())
            finally:
                for stream in second_streams:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    stream.close()

    def test_destination_directory_swap_cannot_escape_derived_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            escape = fixture.root / "escape"
            escape.mkdir()
            original = recovery_module._install_exact_at
            swapped = False

            def swap_before_pack(directory_fd, name, body, *, label):
                nonlocal swapped
                if not swapped and name.endswith(".npzpack"):
                    swapped = True
                    training = fixture.derived / "fragments" / fixture.fragment.name / "training"
                    training.rename(training.with_name("training-before-swap"))
                    training.symlink_to(escape, target_is_directory=True)
                return original(directory_fd, name, body, label=label)

            with mock.patch.object(
                recovery_module, "_install_exact_at", side_effect=swap_before_pack
            ), self.assertRaisesRegex(TrainingTailRecoveryError, "directory component changed"):
                recover_training_tail(fixture.fragment, fixture.derived)

            self.assertEqual(list(escape.iterdir()), [])
            renamed = (
                fixture.derived
                / "fragments"
                / fixture.fragment.name
                / "training-before-swap"
            )
            self.assertFalse(any(renamed.glob("*.complete.json")))

    def test_derived_root_ancestor_rebind_refuses_without_escape_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            requested_parent = fixture.root / "requested-parent"
            requested_parent.mkdir()
            fixture.derived = requested_parent / "derived"
            escape_parent = fixture.root / "escape-parent"
            escape_parent.mkdir()
            displaced = fixture.root / "requested-parent-before-swap"
            original = recovery_module._install_exact_at
            swapped = False

            def swap_ancestor_before_pack(directory_fd, name, body, *, label):
                nonlocal swapped
                if not swapped and name.endswith(".npzpack"):
                    swapped = True
                    requested_parent.rename(displaced)
                    requested_parent.symlink_to(escape_parent, target_is_directory=True)
                return original(directory_fd, name, body, label=label)

            with mock.patch.object(
                recovery_module, "_install_exact_at", side_effect=swap_ancestor_before_pack
            ), self.assertRaisesRegex(TrainingTailRecoveryError, "directory component changed"):
                recover_training_tail(fixture.fragment, fixture.derived)

            self.assertEqual(list(escape_parent.iterdir()), [])
            self.assertFalse(any(displaced.rglob("*.complete.json")))

    def test_source_directory_swap_refuses_before_output_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(Path(temporary))
            escape = fixture.root / "source-escape"
            escape.mkdir()

            def swap_source() -> None:
                fixture.training.rename(fixture.training.with_name("training-before-swap"))
                fixture.training.symlink_to(escape, target_is_directory=True)

            with self.assertRaisesRegex(TrainingTailRecoveryError, "directory component changed"):
                recover_training_tail(
                    fixture.fragment, fixture.derived, before_commit=swap_source
                )
            self.assertFalse(fixture.output("complete.json").exists())


if __name__ == "__main__":
    unittest.main()
