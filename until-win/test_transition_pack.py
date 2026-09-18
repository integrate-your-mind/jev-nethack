import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from transition_pack import (
    FRAME_HEADER,
    MAGIC,
    PUBLIC_ARTIFACT_MAX_BYTES,
    SCHEMA,
    TransitionPackError,
    TransitionPackWriter,
    decode_transition,
    encode_transition,
    scan_pack,
    validate_shard,
)
from verify_training import verify


class TransitionPackTests(unittest.TestCase):
    def observation(self, marker: int):
        return {
            "chars": np.full((2, 3), marker, dtype=np.uint8),
            "blstats": np.array([marker, marker + 1], dtype=np.int64),
            "message": np.frombuffer(f"m{marker}".encode(), dtype=np.uint8),
        }

    def test_round_trip_retains_all_arrays_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            writer = TransitionPackWriter(root)
            receipt = writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1", "actionIndex": 7, "decision": {"model": "jev-1.13.0"}},
            )
            manifest = writer.close()
            frames = list(scan_pack(root / receipt["pack"]))
            self.assertEqual(1, len(frames))
            arrays, metadata = decode_transition(frames[0].payload)
            np.testing.assert_array_equal(self.observation(1)["chars"], arrays["obs__chars"])
            np.testing.assert_array_equal(self.observation(2)["blstats"], arrays["next_obs__blstats"])
            self.assertEqual(SCHEMA, metadata["schema"])
            self.assertEqual("jev-1.13.0", metadata["decision"]["model"])
            self.assertEqual(1, manifest["records"])
            self.assertEqual(2, len(manifest["artifacts"]))
            index = json.loads((root / "transitions-000001.index.jsonl").read_text())
            self.assertEqual(frames[0].sha256, index["sha256"])
            marker = json.loads((root / "transitions-000001.complete.json").read_text())
            self.assertTrue(marker["completed"])
            self.assertEqual(1, marker["transitionCount"])
            self.assertEqual(
                {"transitions-000001.npzpack", "transitions-000001.index.jsonl"},
                {artifact["filename"] for artifact in marker["artifacts"]},
            )
            self.assertTrue(validate_shard(root / receipt["pack"])["completed"])

    def test_partial_trailing_frame_is_ignored_after_complete_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            writer = TransitionPackWriter(root)
            receipt = writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1"},
            )
            writer.close()
            pack = root / receipt["pack"]
            with pack.open("ab") as stream:
                stream.write(struct.pack(">Q", 100))
            frames = list(scan_pack(pack))
            self.assertEqual(["e1"], [frame.metadata["eventId"] for frame in frames])
            with self.assertRaisesRegex(TransitionPackError, "completed transition shard"):
                validate_shard(pack)

    def test_complete_frame_digest_mismatch_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npzpack"
            payload = b"complete but bad"
            path.write_bytes(MAGIC + FRAME_HEADER.pack(len(payload), b"x" * 32) + payload)
            with self.assertRaisesRegex(TransitionPackError, "digest mismatch"):
                list(scan_pack(path))

    def test_rotation_removes_only_empty_eager_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            writer = TransitionPackWriter(root, max_records_per_shard=1)
            writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1"},
            )
            manifest = writer.close()
            self.assertTrue((root / "transitions-000001.npzpack").exists())
            self.assertFalse((root / "transitions-000002.npzpack").exists())
            self.assertEqual(1, manifest["records"])

    def test_unmarked_hard_crash_accepts_only_an_index_prefix_and_reports_partial_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            writer = TransitionPackWriter(root)
            receipt = writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1", "episodeId": 1, "seed": 2, "step": 0},
            )
            assert writer._pack is not None and writer._index is not None
            writer._pack.write(struct.pack(">Q", 100))
            writer._pack.flush()
            writer._pack.close()
            writer._index.close()
            validation = validate_shard(root / receipt["pack"])
            self.assertFalse(validation["completed"])
            self.assertTrue(validation["partialTail"])
            self.assertEqual(1, validation["records"])
            self.assertEqual(1, validation["indexedRecords"])

    def test_index_disagreement_is_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            writer = TransitionPackWriter(root)
            receipt = writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1"},
            )
            writer.close()
            index_path = root / "transitions-000001.index.jsonl"
            record = json.loads(index_path.read_text())
            record["sha256"] = "0" * 64
            index_path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(TransitionPackError, "index disagrees"):
                validate_shard(root / receipt["pack"])

    def test_offline_verifier_distinguishes_integrity_from_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            training = data_root / "fragments" / "fragment-1" / "training"
            writer = TransitionPackWriter(training)
            writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1"},
            )
            writer.close(completed=False, reason="simulated_interruption")
            result = verify(data_root)
            self.assertTrue(result["valid"])
            self.assertFalse(result["complete"])
            self.assertEqual(1, result["transitionCount"])

    def test_offline_verifier_accepts_closed_complete_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            training = data_root / "fragments" / "fragment-1" / "training"
            writer = TransitionPackWriter(training)
            writer.write(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata={"eventId": "e1"},
            )
            writer.close(completed=True)
            result = verify(data_root)
            self.assertTrue(result["valid"])
            self.assertTrue(result["complete"])

    def test_preflight_rotates_before_next_frame_would_cross_pack_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            metadata = {"eventId": "e1"}
            payload = encode_transition(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata=metadata,
            )
            one_frame_limit = len(MAGIC) + FRAME_HEADER.size + len(payload) + 5
            writer = TransitionPackWriter(root, max_bytes_per_shard=one_frame_limit)
            writer.write(observation=self.observation(1), next_observation=self.observation(2), metadata=metadata)
            writer.write(observation=self.observation(1), next_observation=self.observation(2), metadata={"eventId": "e2"})
            writer.close()
            packs = sorted(root.glob("*.npzpack"))
            self.assertEqual(2, len(packs))
            self.assertTrue(all(path.stat().st_size <= one_frame_limit for path in packs))
            self.assertEqual([1, 1], [validate_shard(path)["records"] for path in packs])

    def test_oversize_single_transition_fails_before_pack_or_index_diverge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "training"
            metadata = {"eventId": "e1"}
            payload = encode_transition(
                observation=self.observation(1),
                next_observation=self.observation(2),
                metadata=metadata,
            )
            writer = TransitionPackWriter(
                root,
                max_bytes_per_shard=len(MAGIC) + FRAME_HEADER.size + len(payload) - 1,
            )
            with self.assertRaisesRegex(TransitionPackError, "one transition frame"):
                writer.write(
                    observation=self.observation(1),
                    next_observation=self.observation(2),
                    metadata=metadata,
                )
            self.assertEqual(MAGIC, (root / "transitions-000001.npzpack").read_bytes())
            self.assertEqual(b"", (root / "transitions-000001.index.jsonl").read_bytes())
            writer.close(completed=False, reason="oversize_rejected")

    def test_configured_shard_limit_cannot_exceed_public_site_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TransitionPackError, "public artifact limit"):
                TransitionPackWriter(
                    Path(temporary) / "training",
                    max_bytes_per_shard=PUBLIC_ARTIFACT_MAX_BYTES + 1,
                )


if __name__ == "__main__":
    unittest.main()
