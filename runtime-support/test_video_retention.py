from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import publish_archives as publisher
import video_retention as retention


class RetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recordings = self.root / "recordings"
        self.catalog = self.root / "runtime" / "source-catalog"
        self.receipts = self.root / "runtime" / "publish-receipts"
        self.tombstones = self.root / "runtime" / "recording-tombstones"
        self.site: dict[str, bytes] = {}
        self.github: dict[tuple[str, str], str] = {}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def digest(body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    def write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    def make_item(self, number: int, *, parent: Path | None = None, payload: int = 32):
        directory = (parent or self.recordings) / f"fragment-{number}"
        directory.mkdir(parents=True, exist_ok=True)
        bodies = {
            f"segment-{number:04d}.mp4": bytes([number]) * payload,
            f"segment-{number:04d}.jsonl": (f'{{"sequence":{number}}}\n').encode(),
        }
        artifacts = []
        for filename, body in bodies.items():
            path = directory / filename
            path.write_bytes(body)
            artifacts.append(
                publisher.Artifact(
                    filename, path,
                    "video/mp4" if filename.endswith(".mp4") else publisher.INDEX_TYPE,
                    len(body), self.digest(body),
                )
            )
        item_id = f"broadcast-test-seg{number:04d}"
        manifest_value = {
            "schemaVersion": 1, "sessionId": item_id, "completed": True,
            "startedAt": f"2026-09-18T14:{number:02d}:00+00:00",
            "endedAt": f"2026-09-18T14:{number:02d}:01+00:00",
            "artifacts": [artifact.public_spec() for artifact in artifacts],
        }
        manifest = publisher.canonical_json(manifest_value)
        source = directory / f"segment-{number:04d}.manifest.json"
        source.write_bytes(manifest)
        source_sha = self.digest(manifest)
        releases = tuple(
            publisher.ArchiveFile(
                f"recordings/{item_id}/{artifact.filename}",
                artifact.path, artifact.size, artifact.sha256,
            )
            for artifact in artifacts
        ) + (
            publisher.ArchiveFile(
                f"recordings/{item_id}/manifest.json", source, len(manifest), source_sha
            ),
        )
        item = publisher.ClosedItem(
            "recording", item_id, item_id, source, source_sha,
            manifest_value["startedAt"], manifest_value["endedAt"],
            tuple(artifacts), manifest, releases,
        )
        self.write_json(
            directory / f"segment-{number:04d}.receipt.json",
            {
                "schemaVersion": 1, "segmentId": item_id, "completed": True,
                "endedAt": f"2026-09-18T14:{number:02d}:02+00:00",
                "artifacts": [
                    {
                        "filename": artifact.filename, "bytes": artifact.size,
                        "sha256": artifact.sha256,
                    }
                    for artifact in artifacts
                ] + [{
                    "filename": "manifest.json", "bytes": len(manifest),
                    "sha256": source_sha,
                }],
            },
        )
        key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
        dependencies = {
            value.path.resolve() for value in (*item.artifacts, *item.release_files)
        } | {source.resolve()}
        self.write_json(
            self.catalog / f"recording-{key}.json",
            {
                "schema": "jev-nethack-source-catalog/v1",
                "sourcePath": str(source.resolve()), "lastFullValidatedEpoch": 1,
                "dependencies": [publisher._file_fingerprint(path) for path in sorted(dependencies)],
                "items": [publisher._serialize_item(item)],
            },
        )
        tag, asset = "jev-nethack-archive-2026-09-18", f"batch-{number}.tar.gz"
        archive_sha = self.digest(f"archive-{number}".encode())
        self.write_json(
            self.receipts / "site" / f"recording-{item_id}.verified.json",
            {
                "schema": "jev-nethack-site-publish-receipt/v1", "verified": True,
                "sourceSha256": source_sha, "manifestSha256": self.digest(manifest),
            },
        )
        self.write_json(
            self.receipts / "github-items" / f"recording-{item_id}.verified.json",
            {
                "schema": "jev-nethack-github-item-receipt/v1", "verified": True,
                "sourceSha256": source_sha, "tag": tag, "asset": asset,
                "archiveSha256": archive_sha,
            },
        )
        self.write_json(
            self.receipts / "github-batches" / f"{asset}.verified.json",
            {
                "schema": "jev-nethack-github-publish-receipt/v1", "verified": True,
                "tag": tag, "asset": asset, "sha256": archive_sha,
                "items": [{"kind": "recording", "id": item_id, "sourceSha256": source_sha}],
            },
        )
        for artifact in artifacts:
            self.site[publisher._site_path(item, artifact.filename)] = artifact.path.read_bytes()
        self.site[publisher._site_path(item, "manifest.json")] = manifest
        self.github[(tag, asset)] = archive_sha
        return item

    def site_verify(self, path: str, size: int, digest: str) -> bool:
        body = self.site.get(path)
        return body is not None and len(body) == size and self.digest(body) == digest

    def github_verify(self, tag: str, asset: str, digest: str) -> bool:
        return self.github.get((tag, asset)) == digest

    def make_release(self, number: int, *, size: int = 32):
        release_root = self.root / "runtime" / "release-outbox"
        release_root.mkdir(parents=True, exist_ok=True)
        path = release_root / f"jev-nethack-archive-20260918T{number:06d}Z-test{number}.tar.gz"
        path.write_bytes(bytes([number]) * size)
        digest = self.digest(path.read_bytes())
        tag, item_id = "jev-nethack-archive-2026-09-18", f"broadcast-release-{number}"
        source_sha = self.digest(f"source-{number}".encode())
        self.write_json(
            self.receipts / "github-batches" / f"{path.name}.verified.json",
            {
                "schema": "jev-nethack-github-publish-receipt/v1", "verified": True,
                "verifiedAt": f"2026-09-18T14:00:{number:02d}+00:00",
                "tag": tag, "asset": path.name, "sha256": digest,
                "items": [{"kind": "recording", "id": item_id, "sourceSha256": source_sha}],
            },
        )
        item_path = self.receipts / "github-items" / f"recording-{item_id}.verified.json"
        self.write_json(
            item_path,
            {
                "schema": "jev-nethack-github-item-receipt/v1", "verified": True,
                "sourceSha256": source_sha, "tag": tag, "asset": path.name,
                "archiveSha256": digest,
            },
        )
        self.github[(tag, path.name)] = digest
        return path, item_path

    def prune(self, items, budget: int, **kwargs):
        return retention.prune_recording_cache(
            items, [self.recordings], budget_bytes=budget,
            tombstone_root=self.tombstones, catalog_root=self.catalog, receipts=self.receipts,
            site_verify=self.site_verify, github_verify=self.github_verify,
            site_path=publisher._site_path, open_writer=lambda _path: False,
            now=lambda: datetime(2026, 9, 18, 18, 0, tzinfo=timezone.utc),
            **kwargs,
        )

    def test_cache_bound_prunes_oldest_closed_group_and_never_training_or_other_files(self) -> None:
        oldest, newest = self.make_item(1), self.make_item(2)
        protected = {
            self.recordings / "events.jsonl": b"events",
            self.recordings / "segment-9999.png": b"png",
            self.recordings / "transitions-000001.npzpack": b"training",
            self.recordings / "episode.ttyrec": b"tty",
        }
        for path, body in protected.items():
            path.write_bytes(body)
        newest_size = sum(value.size for value in newest.artifacts)
        report = self.prune([oldest, newest], newest_size)
        self.assertTrue(report["withinBudget"])
        self.assertEqual(report["afterBytes"], newest_size)
        self.assertTrue(all(not value.path.exists() for value in oldest.artifacts))
        self.assertTrue(all(value.path.exists() for value in newest.artifacts))
        self.assertTrue(all(path.read_bytes() == body for path, body in protected.items()))
        tombstone = next(self.tombstones.glob("*.json"))
        self.assertEqual(json.loads(tombstone.read_text())["state"], "pruned")

    def test_remote_unavailable_or_hash_mismatch_never_prunes(self) -> None:
        item = self.make_item(1)
        self.site.clear()
        report = self.prune([item], 1)
        self.assertFalse(report["withinBudget"])
        self.assertTrue(report["blocked"])
        self.assertTrue(all(value.path.exists() for value in item.artifacts))

    def test_completed_manifest_without_final_recorder_receipt_is_not_pruned(self) -> None:
        item = self.make_item(1)
        item.source_path.with_name("segment-0001.receipt.json").unlink()
        report = self.prune([item], 1)
        self.assertFalse(report["withinBudget"])
        self.assertIn("metadata", report["blocked"][0]["error"])
        self.assertTrue(all(value.path.exists() for value in item.artifacts))

        self.site[publisher._site_path(item, item.artifacts[0].filename)] = b"wrong"
        report = self.prune([item], 1)
        self.assertFalse(report["withinBudget"])
        self.assertTrue(all(value.path.exists() for value in item.artifacts))

    def test_symlink_cross_root_open_writer_and_mutation_fail_closed(self) -> None:
        item = self.make_item(1)
        video = item.artifacts[0].path
        original = video.read_bytes()
        outside = self.root / "outside.mp4"
        outside.write_bytes(original)
        video.unlink()
        video.symlink_to(outside)
        report = self.prune([item], 1)
        self.assertTrue(report["blocked"])
        self.assertEqual(outside.read_bytes(), original)

        external = self.make_item(2, parent=self.root / "external")
        with self.assertRaises(retention.RetentionError):
            self.prune([external], 1)

        changed = self.make_item(3)
        target = changed.artifacts[0].path
        report = self.prune(
            [changed], 1,
            before_unlink=lambda path: path.write_bytes(b"mutated") if path.name == target.name else None,
        )
        self.assertTrue(report["blocked"])
        self.assertTrue(target.exists())
        self.assertTrue(changed.artifacts[1].path.exists())

        busy = self.make_item(4)
        report = retention.prune_recording_cache(
            [busy], [self.recordings], budget_bytes=1,
            tombstone_root=self.tombstones, catalog_root=self.catalog, receipts=self.receipts,
            site_verify=self.site_verify, github_verify=self.github_verify,
            site_path=publisher._site_path,
            open_writer=lambda path: path.name == busy.artifacts[0].path.name,
        )
        self.assertTrue(report["blocked"])
        self.assertTrue(all(value.path.exists() for value in busy.artifacts))

    def test_prepared_tombstone_recovers_idempotently_after_crash(self) -> None:
        item = self.make_item(1)
        calls = 0

        def crash(_path: Path) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("simulated process death")

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.prune([item], 1, before_unlink=crash)
        marker = next(self.tombstones.glob("*.json"))
        self.assertEqual(json.loads(marker.read_text())["state"], "prepared")
        self.assertTrue(all(value.path.exists() for value in item.artifacts))

        recovered = retention.recover_tombstones(
            self.tombstones, [self.recordings], open_writer=lambda _path: False,
        )
        self.assertEqual(
            set(recovered), {str(value.path.resolve(strict=False)) for value in item.artifacts}
        )
        self.assertEqual(
            retention.recover_tombstones(
                self.tombstones, [self.recordings], open_writer=lambda _path: False,
            ),
            [],
        )
        self.assertEqual(json.loads(marker.read_text())["state"], "pruned")

    def test_pruned_catalog_rediscovery_and_full_remote_recheck_need_no_local_artifact(self) -> None:
        item = self.make_item(1)
        report = self.prune([item], 1)
        self.assertTrue(report["withinBudget"])
        discovered, errors = publisher.collect_items(
            self.recordings, self.recordings, self.root / "runtime" / "repackaged",
            catalog_root=self.catalog, tombstone_root=self.tombstones,
            now_epoch=10**9, scrub_seconds=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual([(value.kind, value.item_id) for value in discovered], [("recording", item.item_id)])

        site_receipt = self.receipts / "site" / f"recording-{item.item_id}.verified.json"
        receipt = json.loads(site_receipt.read_text())
        receipt.update(
            {
                "verifiedAt": "2026-09-17T00:00:00+00:00",
                "lastFullVerifiedAt": "2026-09-17T00:00:00+00:00",
                "lastRemoteVerifiedAt": "2026-09-17T00:00:00+00:00",
                "nextRemoteCheckEpoch": 0,
            }
        )
        self.write_json(site_receipt, receipt)

        class DigestSite:
            def __init__(self, objects):
                self.objects = objects
                self.digest_checks = 0

            def verify_digest(inner, path, size, digest):
                inner.digest_checks += 1
                body = inner.objects.get(path)
                return body is not None and len(body) == size and self.digest(body) == digest

            def verify(inner, path, body):
                return inner.objects.get(path) == body

            def head_size_matches(inner, path, size):
                return path in inner.objects and len(inner.objects[path]) == size

            def put_verified(inner, *_args, **_kwargs):
                self.fail("pruned recording must not be rebuilt when remote bytes are intact")

        client = DigestSite(self.site)
        result = publisher.publish_site_item(
            discovered[0], client, self.receipts,
            full_recheck_seconds=0, remote_recheck_seconds=900,
            now=lambda: datetime(2026, 9, 18, 18, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(result["state"], "already_verified")
        self.assertEqual(client.digest_checks, len(item.artifacts))

    def test_zero_budget_disables_pruning(self) -> None:
        item = self.make_item(1)
        report = self.prune([item], 0)
        self.assertTrue(report["disabled"])
        self.assertTrue(report["withinBudget"])
        self.assertTrue(all(value.path.exists() for value in item.artifacts))

    def test_release_cache_removes_oldest_fully_verified_archive_and_is_idempotent(self) -> None:
        oldest, _ = self.make_release(1)
        newest, _ = self.make_release(2)
        unrelated = oldest.parent / "unrelated.tar.gz"
        unrelated.write_bytes(b"preserve")
        report = retention.prune_release_cache(
            oldest.parent, self.receipts, budget_bytes=newest.stat().st_size,
            github_verify=self.github_verify,
        )
        self.assertTrue(report["withinBudget"])
        self.assertFalse(oldest.exists())
        self.assertTrue(newest.exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(report["removed"][0]["path"], str(oldest.resolve(strict=False)))
        second = retention.prune_release_cache(
            oldest.parent, self.receipts, budget_bytes=newest.stat().st_size,
            github_verify=lambda *_args: self.fail("at-bound cache must not hit GitHub"),
        )
        self.assertEqual(second["removed"], [])

    def test_release_cache_preserves_remote_failure_pending_and_incomplete_receipts(self) -> None:
        remote_error, _ = self.make_release(1)
        remote_mismatch, _ = self.make_release(2)
        pending, _ = self.make_release(3)
        incomplete, incomplete_item = self.make_release(4)
        pending_outbox = self.receipts / "outbox" / f"github-{pending.name}.json"
        self.write_json(pending_outbox, {"state": "pending"})
        value = json.loads(incomplete_item.read_text())
        value["archiveSha256"] = "0" * 64
        self.write_json(incomplete_item, value)

        def verify(_tag: str, asset: str, _digest: str) -> bool:
            if asset == remote_error.name:
                raise OSError("offline")
            if asset == remote_mismatch.name:
                return False
            return True

        report = retention.prune_release_cache(
            remote_error.parent, self.receipts, budget_bytes=1, github_verify=verify,
        )
        self.assertFalse(report["withinBudget"])
        self.assertEqual(len(report["blocked"]), 4)
        self.assertTrue(all(path.exists() for path in (
            remote_error, remote_mismatch, pending, incomplete
        )))

    def test_release_cache_rejects_symlink_root_without_touching_target(self) -> None:
        outside = self.root / "outside-release"
        outside.mkdir()
        artifact = outside / "jev-nethack-archive-test.tar.gz"
        artifact.write_bytes(b"outside")
        linked = self.root / "linked-release"
        linked.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(retention.RetentionError, "root may not be a symlink"):
            retention.prune_release_cache(
                linked, self.receipts, budget_bytes=1, github_verify=self.github_verify,
            )
        self.assertEqual(artifact.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
