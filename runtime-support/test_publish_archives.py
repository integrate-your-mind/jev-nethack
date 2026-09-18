from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "until-win"))

from transition_pack import TransitionPackWriter, scan_pack  # noqa: E402
import publish_archives as publisher  # noqa: E402


START = "2026-09-18T14:00:00+00:00"


class MemorySite:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[str] = []
        self.verify_calls = 0
        self.head_calls = 0

    def verify(self, path: str, body: bytes) -> bool:
        self.verify_calls += 1
        return self.objects.get(path) == body

    def verify_digest(self, path: str, size: int, digest: str) -> bool:
        self.verify_calls += 1
        body = self.objects.get(path)
        return (
            body is not None and len(body) == size
            and hashlib.sha256(body).hexdigest() == digest
        )

    def head_size_matches(self, path: str, size: int) -> bool:
        self.head_calls += 1
        return path in self.objects and len(self.objects[path]) == size

    def put_verified(self, path: str, body: bytes, *, content_type: str) -> None:
        existing = self.objects.get(path)
        if existing is not None and existing != body:
            raise publisher.PublishError("immutable conflict")
        self.objects[path] = bytes(body)
        self.puts.append(path)


class FakeGH:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.assets: dict[tuple[str, str], str] = {}
        self.checks = 0

    def upload_verified(self, tag: str, archive: Path, expected_sha: str) -> None:
        if publisher.sha256_file(archive) != expected_sha:
            raise publisher.PublishError("local archive mismatch")
        self.uploads.append((tag, archive.name))
        self.assets[(tag, archive.name)] = expected_sha

    def asset_verified(self, tag: str, asset: str, expected_sha: str) -> bool:
        self.checks += 1
        actual = self.assets.get((tag, asset))
        if actual is None:
            return False
        if actual != expected_sha:
            raise publisher.PublishError("remote conflict")
        return True


class PublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_training(self, *, records: int = 2, bytes_per_observation: int = 64,
                      max_records: int = 128) -> tuple[Path, Path, list[dict]]:
        data = self.root / "data"
        fragment_id = "fragment-20260918T140000.000000Z-pid1234-a1b2c3d4"
        fragment = data / "fragments" / fragment_id
        fragment.mkdir(parents=True)
        (fragment / "config.json").write_text(
            json.dumps({"fragmentId": fragment_id, "startedAt": START}), encoding="utf-8"
        )
        writer = TransitionPackWriter(fragment / "training", max_records_per_shard=max_records)
        receipts = []
        for number in range(records):
            random = np.random.default_rng(number)
            observation = {"glyphs": random.integers(0, 256, bytes_per_observation, dtype=np.uint8)}
            next_observation = {"glyphs": random.integers(0, 256, bytes_per_observation, dtype=np.uint8)}
            receipts.append(
                writer.write(
                    observation=observation,
                    next_observation=next_observation,
                    metadata={
                        "eventId": f"event-{number}", "recordedAt": f"2026-09-18T14:00:{number:02d}+00:00",
                        "episodeId": 1, "seed": 103, "step": number,
                    },
                )
            )
        writer.close(completed=False, reason="controlled_test_stop")
        return data, fragment, receipts

    def make_recording(self) -> tuple[Path, Path]:
        root = self.root / "recordings"
        directory = root / "fragment-a"
        directory.mkdir(parents=True)
        video = directory / "segment-0001.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=64x64:d=0.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(video)],
            check=True,
        )
        events = directory / "segment-0001.jsonl"
        events.write_text('{"sequence":1}\n', encoding="utf-8")
        session = "broadcast-test-seg0001"
        manifest = {
            "schemaVersion": 1, "sessionId": session, "broadcastId": "test",
            "completed": True, "startedAt": START, "endedAt": "2026-09-18T14:00:01+00:00",
            "frameCount": 1, "actionCount": 1,
            "artifacts": [
                {"filename": video.name, "bytes": video.stat().st_size,
                 "sha256": publisher.sha256_file(video), "contentType": "video/mp4"},
                {"filename": events.name, "bytes": events.stat().st_size,
                 "sha256": publisher.sha256_file(events), "contentType": "application/x-ndjson"},
            ],
        }
        path = directory / "segment-0001.manifest.json"
        path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        (directory / "segment-0001.receipt.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1, "segmentId": session, "completed": True,
                    "endedAt": "2026-09-18T14:00:02+00:00",
                    "artifacts": [
                        {
                            "filename": artifact["filename"], "bytes": artifact["bytes"],
                            "sha256": artifact["sha256"],
                        }
                        for artifact in manifest["artifacts"]
                    ] + [{
                        "filename": "manifest.json", "bytes": path.stat().st_size,
                        "sha256": publisher.sha256_file(path),
                    }],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return root, path

    def make_crashed_broadcast_stub(self) -> tuple[Path, Path]:
        data = self.root / "crash-data"
        stamp = "20260918T142018.869019Z"
        stream_id = "a" * 32
        fragment = data / "broadcast-fragments" / f"broadcast-{stamp}-{stream_id}"
        fragment.mkdir(parents=True)
        (fragment / "segment-0002.jsonl").write_text(
            json.dumps({"sequence": 2, "stream_id": stream_id}) + "\n",
            encoding="utf-8",
        )
        return data, fragment

    def parse_training(self, data: Path, marker: Path) -> list[publisher.ClosedItem]:
        return publisher.parse_training_marker(marker, data, self.root / "derived")

    def test_real_writer_multiframe_and_rotation_contract(self) -> None:
        data, fragment, _ = self.make_training(records=3, max_records=2)
        markers = sorted((fragment / "training").glob("transitions-*.complete.json"))
        self.assertEqual(len(markers), 2)
        source_bytes = {path: path.read_bytes() for path in (fragment / "training").iterdir()}
        parsed = [self.parse_training(data, marker)[0] for marker in markers]
        self.assertEqual([len(item.artifacts) for item in parsed], [2, 2])
        self.assertEqual(len({item.item_id for item in parsed}), 2)
        for item, expected_count in zip(parsed, (2, 1)):
            public = json.loads(item.site_manifest)
            self.assertEqual(public["schema"], publisher.PACK_SCHEMA)
            self.assertEqual(public["schemaVersion"], 1)
            self.assertIsInstance(public["artifacts"], list)
            self.assertEqual(public["transitionCount"], expected_count)
            self.assertRegex(public["sessionId"], publisher.SITE_ID_RE)
            self.assertNotIn("broadcastId", public)
        self.assertEqual(source_bytes, {path: path.read_bytes() for path in source_bytes})

    def test_deterministic_subshards_preserve_npz_payload_bytes(self) -> None:
        data, fragment, _ = self.make_training(records=3, bytes_per_observation=700)
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        original_pack = next((fragment / "training").glob("*.npzpack"))
        original_payloads = [frame.payload for frame in scan_pack(original_pack)]
        size_one = len(publisher.MAGIC) + publisher.FRAME_HEADER.size + len(original_payloads[0])
        artificial_limit = size_one + 100
        with mock.patch.object(publisher, "MAX_UPLOAD", artificial_limit):
            items = self.parse_training(data, marker)
        self.assertGreater(len(items), 1)
        rebuilt_payloads = []
        for item in items:
            self.assertLessEqual(item.artifacts[0].size, artificial_limit)
            rebuilt_payloads.extend(frame.payload for frame in scan_pack(item.artifacts[0].path))
            manifest = json.loads(item.site_manifest)
            self.assertTrue(manifest["provenance"]["payloadBytesUnchanged"])
        self.assertEqual(original_payloads, rebuilt_payloads)

    def test_corrupt_index_and_symlink_artifact_are_rejected(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        index = next((fragment / "training").glob("*.index.jsonl"))
        original = index.read_bytes()
        index.write_bytes(original.replace(b'"offset":10', b'"offset":11', 1))
        marker_doc = json.loads(marker.read_text())
        for artifact in marker_doc["artifacts"]:
            if artifact["filename"] == index.name:
                artifact["sha256"] = publisher.sha256_file(index)
        marker.write_text(json.dumps(marker_doc))
        with self.assertRaises(publisher.PublishError):
            self.parse_training(data, marker)
        index.unlink()
        outside = self.root / "outside.index.jsonl"
        outside.write_bytes(original)
        index.symlink_to(outside)
        marker_doc["artifacts"][1]["sha256"] = publisher.sha256_file(outside)
        marker_doc["artifacts"][1]["bytes"] = outside.stat().st_size
        marker.write_text(json.dumps(marker_doc))
        with self.assertRaises(publisher.PublishError):
            self.parse_training(data, marker)

    def test_recording_completed_true_and_symlink_exfiltration_guard(self) -> None:
        recordings, manifest = self.make_recording()
        item = publisher.parse_recording_manifest(manifest, recordings)
        self.assertEqual(item.kind, "recording")
        self.assertEqual(len(item.artifacts), 2)
        video = manifest.parent / "segment-0001.mp4"
        video.unlink()
        secret = self.root / "secret"
        secret.write_bytes(b"private")
        video.symlink_to(secret)
        with self.assertRaises(publisher.PublishError):
            publisher.parse_recording_manifest(manifest, recordings)

    def test_toctou_change_is_rejected_before_site_write(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        item = self.parse_training(data, marker)[0]
        artifact = item.artifacts[0]
        changed = bytearray(artifact.path.read_bytes())
        changed[-1] ^= 1
        artifact.path.write_bytes(changed)
        site = MemorySite()
        with self.assertRaises(publisher.PublishError):
            publisher.publish_site_item(item, site, self.root / "receipts")
        self.assertEqual(site.objects, {})

    def test_site_http_response_loss_recording_409_and_restart_receipt(self) -> None:
        recordings, manifest = self.make_recording()
        item = publisher.parse_recording_manifest(manifest, recordings)
        state: dict[str, bytes] = {}
        put_counts: dict[str, int] = {}
        seen_agents: list[str] = []
        drop_once = {publisher._site_path(item, item.artifacts[0].filename)}

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                seen_agents.append(self.headers.get("User-Agent", ""))
                put_counts[self.path] = put_counts.get(self.path, 0) + 1
                if self.headers.get("Authorization") != "Bearer test-token":
                    self.send_response(401); self.end_headers(); return
                if self.headers.get("X-Content-SHA256") != hashlib.sha256(body).hexdigest():
                    self.send_response(422); self.end_headers(); return
                if self.path in state:
                    self.send_response(409); self.end_headers(); return
                state[self.path] = body
                if self.path in drop_once:
                    drop_once.remove(self.path)
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.send_response(201); self.end_headers()

            def do_GET(self):
                body = state.get(self.path)
                if body is None:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self):
                body = state.get(self.path)
                if body is None:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = object.__new__(publisher.SiteClient)
            client.origin = f"http://127.0.0.1:{server.server_port}"
            client.token, client.timeout = "test-token", 5
            client._opener = urllib.request.build_opener(publisher._NoRedirect())
            receipts = self.root / "receipts"
            first = publisher.publish_site_item(item, client, receipts, remote_recheck_seconds=0)
            self.assertEqual(first["state"], "published")
            before = dict(put_counts)
            second = publisher.publish_site_item(item, client, receipts, remote_recheck_seconds=0)
            self.assertEqual(second["state"], "already_verified")
            self.assertEqual(put_counts, before)
            self.assertTrue(all(agent == publisher.USER_AGENT for agent in seen_agents))
            manifest_path = publisher._site_path(item, "manifest.json")
            self.assertEqual(state[manifest_path], item.site_manifest)
            missing_artifact = publisher._site_path(item, item.artifacts[0].filename)
            state.pop(missing_artifact)
            repaired = publisher.publish_site_item(item, client, receipts, remote_recheck_seconds=0)
            self.assertEqual(repaired["state"], "published")
            self.assertEqual(state[missing_artifact], item.artifacts[0].path.read_bytes())
            state[manifest_path] = b"different immutable bytes"
            with self.assertRaises(publisher.PublishError):
                publisher.publish_site_item(item, client, receipts, remote_recheck_seconds=0)
        finally:
            server.shutdown()
            server.server_close()

    def test_redirect_does_not_forward_bearer_token(self) -> None:
        target_requests: list[str | None] = []

        class Target(BaseHTTPRequestHandler):
            def do_PUT(self):
                target_requests.append(self.headers.get("Authorization")); self.send_response(200); self.end_headers()
            def do_GET(self):
                target_requests.append(self.headers.get("Authorization")); self.send_response(200); self.end_headers()
            def log_message(self, *_): pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
        threading.Thread(target=target.serve_forever, daemon=True).start()

        class Redirect(BaseHTTPRequestHandler):
            def _redirect(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/capture")
                self.end_headers()
            do_PUT = do_GET = _redirect
            def log_message(self, *_): pass

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        threading.Thread(target=redirect.serve_forever, daemon=True).start()
        try:
            client = object.__new__(publisher.SiteClient)
            client.origin = f"http://127.0.0.1:{redirect.server_port}"
            client.token, client.timeout = "do-not-forward", 5
            client._opener = urllib.request.build_opener(publisher._NoRedirect())
            with self.assertRaises(publisher.PublishError):
                client.put_verified("/redirect", b"value", content_type="application/octet-stream")
            self.assertEqual(target_requests, [])
        finally:
            redirect.shutdown(); redirect.server_close(); target.shutdown(); target.server_close()

    def test_durable_retry_restart_and_backoff_receipt_window(self) -> None:
        outbox = self.root / "outbox" / "operation.json"
        durable_receipt = self.root / "verified"
        calls = 0

        def interrupted():
            nonlocal calls
            calls += 1
            durable_receipt.write_text("verified")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            publisher.durable_retry(
                operation_id="test", outbox_path=outbox, operation=interrupted,
                delays=(), sleep=lambda _: None,
            )
        self.assertTrue(outbox.exists())

        def resume():
            self.assertTrue(durable_receipt.exists())
            return "settled"

        self.assertEqual(
            publisher.durable_retry(
                operation_id="test", outbox_path=outbox, operation=resume,
                delays=(), sleep=lambda _: None,
            ),
            "settled",
        )
        self.assertFalse(outbox.exists())

    def test_single_instance_lock(self) -> None:
        path = self.root / "publisher.lock"
        with publisher.PublisherLock(path):
            with self.assertRaises(publisher.LockBusy):
                with publisher.PublisherLock(path):
                    pass
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_lock_rejects_symlink_fifo_directory_and_releases_after_killed_holder(self) -> None:
        target = self.root / "do-not-touch"
        target.write_text("DO-NOT-TOUCH")
        symlink = self.root / "symlink.lock"
        symlink.symlink_to(target)
        with self.assertRaises(publisher.PublishError):
            with publisher.PublisherLock(symlink): pass
        self.assertEqual(target.read_text(), "DO-NOT-TOUCH")
        fifo = self.root / "fifo.lock"
        os.mkfifo(fifo, 0o600)
        with self.assertRaises(publisher.PublishError):
            with publisher.PublisherLock(fifo): pass
        directory = self.root / "directory.lock"
        directory.mkdir()
        with self.assertRaises(publisher.PublishError):
            with publisher.PublisherLock(directory): pass

        process_lock = self.root / "process.lock"
        code = (
            "import time; from pathlib import Path; from publish_archives import PublisherLock; "
            f"p=Path({str(process_lock)!r}); "
            "lock=PublisherLock(p); lock.__enter__(); print('ready',flush=True); time.sleep(60)"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", code], cwd=HERE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            with self.assertRaises(publisher.LockBusy):
                with publisher.PublisherLock(process_lock): pass
        finally:
            holder.kill(); holder.communicate(timeout=5)
        with publisher.PublisherLock(process_lock):
            pass

    def test_due_full_site_recheck_advances_receipt_then_returns_to_head_checks(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        item = self.parse_training(data, marker)[0]
        site, receipts = MemorySite(), self.root / "receipts"
        initial = datetime(2026, 9, 18, 14, tzinfo=timezone.utc)
        publisher.publish_site_item(
            item, site, receipts, remote_recheck_seconds=0, now=lambda: initial,
        )
        site.verify_calls = site.head_calls = 0
        due = initial + timedelta(days=1, seconds=1)
        publisher.publish_site_item(
            item, site, receipts, remote_recheck_seconds=0, now=lambda: due,
        )
        self.assertEqual(site.verify_calls, len(item.artifacts) + 1)
        self.assertEqual(site.head_calls, 0)
        receipt = publisher.read_json(
            receipts / "site" / f"{item.kind}-{item.item_id}.verified.json"
        )
        self.assertEqual(receipt["lastFullVerifiedAt"], due.isoformat())
        site.verify_calls = site.head_calls = 0
        publisher.publish_site_item(
            item, site, receipts, remote_recheck_seconds=0,
            now=lambda: due + timedelta(seconds=1),
        )
        self.assertEqual(site.head_calls, len(item.artifacts))
        self.assertEqual(site.verify_calls, 1)

    def test_remote_receipt_skips_outbox_until_due_and_budget_bounds_checks(self) -> None:
        data, _, _ = self.make_training(records=4, max_records=1)
        runtime = self.root / "runtime"
        token = self.root / "token"
        token.write_text("test-token")
        token.chmod(0o600)
        site = MemorySite()
        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=None, runtime_root=str(runtime),
            site_url="https://example.invalid", token_file=str(token),
            github_repo=None, gh="gh", batch_seconds=600,
            remote_recheck_seconds=900, remote_recheck_budget=2,
        )
        initial = datetime(2026, 9, 18, 14, tzinfo=timezone.utc).timestamp()
        with mock.patch.object(publisher, "SiteClient", return_value=site):
            first = publisher.run_once(arguments, delays=(), sleep=lambda _: None,
                                       now_epoch=lambda: initial)
            self.assertEqual(first["errors"], [])
            outbox = runtime / "publish-receipts" / "outbox"
            first_outbox_mtime = outbox.stat().st_mtime_ns
            first_calls = (site.verify_calls, site.head_calls, tuple(site.puts))
            second = publisher.run_once(arguments, delays=(), sleep=lambda _: None,
                                        now_epoch=lambda: initial + 60)
            self.assertEqual(second["errors"], [])
            self.assertTrue(all(row["state"] == "receipt_current" for row in second["site"]))
            self.assertEqual((site.verify_calls, site.head_calls, tuple(site.puts)), first_calls)
            self.assertEqual(outbox.stat().st_mtime_ns, first_outbox_mtime)

            due = initial + 2 * 900 + 1
            third = publisher.run_once(arguments, delays=(), sleep=lambda _: None,
                                       now_epoch=lambda: due)
            self.assertEqual(third["errors"], [])
            self.assertEqual(sum(row["remoteChecked"] for row in third["site"]), 2)
            self.assertEqual(sum(row["state"] == "recheck_deferred" for row in third["site"]), 2)

        receipt_values = [
            publisher.read_json(path)
            for path in sorted((runtime / "publish-receipts" / "site").glob("*.json"))
        ]
        initial_next = {
            value["nextRemoteCheckEpoch"] for value in receipt_values
            if value["verifiedAt"] == datetime.fromtimestamp(initial, timezone.utc).isoformat()
        }
        self.assertGreater(len(initial_next), 1, "stable identity jitter must stagger remote checks")

    def test_release_archive_is_deterministic_and_contains_only_allowlist(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        item = self.parse_training(data, marker)[0]
        bucket = publisher._bucket(item.ended_at, 600)
        first = publisher.build_release_archive([item], self.root / "release-a", bucket)
        second = publisher.build_release_archive([item], self.root / "release-b", bucket)
        self.assertEqual(first["sha256"], second["sha256"])
        with tarfile.open(first["archive"]) as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.isfile() for member in members))
            names = {member.name for member in members}
            self.assertTrue(any(name.endswith("SHA256SUMS.json") for name in names))
            self.assertFalse(any("config.json" in name or "token" in name for name in names))

    def test_github_ambiguous_create_upload_and_conflict(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        item = self.parse_training(data, marker)[0]
        batch = publisher.build_release_archive(
            [item], self.root / "release", publisher._bucket(item.ended_at, 600)
        )
        releases: dict[str, dict[str, bytes]] = {}
        create_calls = upload_calls = 0

        def runner(command, **_):
            nonlocal create_calls, upload_calls
            arguments = command[1:]
            action, tag = arguments[1], arguments[2]
            if action == "view":
                if tag not in releases:
                    return subprocess.CompletedProcess(command, 1, "", "release not found")
                assets = [
                    {"name": name, "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
                    for name, data in releases[tag].items()
                ]
                return subprocess.CompletedProcess(command, 0, json.dumps({"assets": assets, "url": "x"}), "")
            if action == "create":
                create_calls += 1
                releases.setdefault(tag, {})
                raise subprocess.TimeoutExpired(command, 120)
            if action == "upload":
                upload_calls += 1
                archive = Path(arguments[3])
                releases.setdefault(tag, {})[archive.name] = archive.read_bytes()
                raise subprocess.TimeoutExpired(command, 120)
            if action == "download":
                name = arguments[arguments.index("--pattern") + 1]
                directory = Path(arguments[arguments.index("--dir") + 1])
                (directory / name).write_bytes(releases[tag][name])
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(arguments)

        client = publisher.GitHubClient("owner/repo", runner=runner)
        archive = Path(batch["archive"])
        client.upload_verified(batch["tag"], archive, batch["sha256"])
        client.upload_verified(batch["tag"], archive, batch["sha256"])
        self.assertEqual((create_calls, upload_calls), (1, 1))
        receipts = self.root / "gh-receipts"
        publisher.publish_github_batch(batch, client, receipts, remote_recheck_seconds=0)
        releases[batch["tag"]].pop(archive.name)
        fresh = publisher.GitHubClient("owner/repo", runner=runner)
        self.assertTrue(publisher._release_pending(receipts, item, fresh))
        releases[batch["tag"]][archive.name] = b"conflict"
        with self.assertRaises(publisher.PublishError):
            publisher._release_pending(receipts, item, publisher.GitHubClient("owner/repo", runner=runner))

    def test_batch_receipt_repairs_item_receipt_and_orphaned_outbox_after_crash(self) -> None:
        data, fragment, _ = self.make_training()
        marker = next((fragment / "training").glob("transitions-*.complete.json"))
        item = self.parse_training(data, marker)[0]
        batch = publisher.build_release_archive(
            [item], self.root / "release", publisher._bucket(item.ended_at, 600)
        )
        receipts = self.root / "receipts"
        outbox = receipts / "outbox" / f"github-{batch['asset']}.json"
        real_atomic_json = publisher.atomic_json

        def crash_before_item_receipt(path, value):
            if Path(path).parent.name == "github-items":
                raise KeyboardInterrupt()
            real_atomic_json(Path(path), value)

        with mock.patch.object(publisher, "atomic_json", side_effect=crash_before_item_receipt):
            with self.assertRaises(KeyboardInterrupt):
                publisher.durable_retry(
                    operation_id=f"github:{batch['asset']}", outbox_path=outbox,
                    operation=lambda: publisher.publish_github_batch(batch, FakeGH(), receipts),
                    delays=(), sleep=lambda _: None,
                )
        self.assertTrue(outbox.exists())
        self.assertTrue(any((receipts / "github-batches").glob("*.verified.json")))
        self.assertFalse(publisher._release_receipt_path(receipts, item).exists())
        publisher.reconcile_github_item_receipts(receipts, [item])
        self.assertTrue(publisher._release_receipt_path(receipts, item).exists())
        self.assertFalse(outbox.exists(), "settled batch outbox must not remain orphaned")

    def test_run_once_wires_raw_recording_site_github_and_restart(self) -> None:
        data, fragment, _ = self.make_training()
        recordings, _ = self.make_recording()
        runtime = self.root / "runtime"
        token = self.root / "token"
        token.write_text("test-token")
        token.chmod(0o600)
        site, github = MemorySite(), FakeGH()
        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=str(recordings), runtime_root=str(runtime),
            site_url="https://example.invalid", token_file=str(token),
            github_repo="owner/repo", gh="gh", batch_seconds=600,
        )
        with mock.patch.object(publisher, "SiteClient", return_value=site), mock.patch.object(
            publisher, "GitHubClient", return_value=github
        ):
            first = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            self.assertEqual(first["errors"], [])
            self.assertTrue(any(path.startswith("/api/training/") for path in site.objects))
            self.assertTrue(any(path.startswith("/api/recordings/") for path in site.objects))
            first_upload_count = len(github.uploads)
            self.assertGreater(first_upload_count, 0)
            missing_item_receipt = next((runtime / "publish-receipts" / "github-items").glob("*.json"))
            missing_item_receipt.unlink()
            second = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            self.assertEqual(second["errors"], [])
            self.assertEqual(len(github.uploads), first_upload_count)
            self.assertEqual(github.checks, 0, "current GitHub receipts must not hit the network")
            self.assertTrue(missing_item_receipt.exists(), "batch receipt must repair a crash during item receipts")

    def test_run_once_prunes_only_after_dual_publish_and_rediscovery_does_not_rebuild(self) -> None:
        data = self.root / "retention-data"
        data.mkdir()
        recordings, manifest = self.make_recording()
        runtime = self.root / "retention-runtime"
        token = self.root / "retention-token"
        token.write_text("secret")
        token.chmod(0o600)
        site, github = MemorySite(), FakeGH()
        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=str(recordings), runtime_root=str(runtime),
            site_url="https://example.invalid", token_file=str(token),
            github_repo="owner/repo", gh="gh", batch_seconds=600,
            recording_cache_bytes=1, release_cache_bytes=1, remote_recheck_seconds=0,
        )
        with mock.patch.object(publisher, "SiteClient", return_value=site), mock.patch.object(
            publisher, "GitHubClient", return_value=github
        ), mock.patch("video_retention._default_open_writer", return_value=False):
            first = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            second = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            github.assets.clear()
            third = publisher.run_once(arguments, delays=(), sleep=lambda _: None)

        self.assertEqual(first["errors"], [])
        self.assertTrue(first["recordingRetention"]["withinBudget"])
        self.assertEqual(first["recordingRetention"]["afterBytes"], 0)
        self.assertFalse((manifest.parent / "segment-0001.mp4").exists())
        self.assertFalse((manifest.parent / "segment-0001.jsonl").exists())
        self.assertTrue(manifest.exists())
        self.assertTrue((manifest.parent / "segment-0001.receipt.json").exists())
        self.assertEqual(list((runtime / "release-outbox").glob("*.tar.gz")), [])
        self.assertTrue(first["releaseCache"]["withinBudget"])
        self.assertTrue(first["releaseCache"]["removed"])
        self.assertEqual(second["errors"], [])
        self.assertEqual(second["sources"], first["sources"])
        self.assertEqual(len(github.uploads), 1)
        self.assertGreaterEqual(github.checks, 2)
        self.assertTrue(any("archive rebuild is impossible" in row["error"] for row in third["errors"]))
        self.assertEqual(len(github.uploads), 1)

    def test_run_once_recovers_dead_unmarked_training_tail_without_mutating_source(self) -> None:
        data, fragment, _ = self.make_training(records=3)
        training = fragment / "training"
        marker = next(training.glob("transitions-*.complete.json"))
        source_pack = next(training.glob("*.npzpack"))
        source_index = next(training.glob("*.index.jsonl"))
        source_bytes = {source_pack: source_pack.read_bytes(), source_index: source_index.read_bytes()}
        marker.unlink()
        config = json.loads((fragment / "config.json").read_text())
        config.update(
            schema="jev-nethack-until-win/v1", pid=2_147_483_647,
            transitionSchema="jev-nethack-transition-pack/v1",
        )
        (fragment / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (data / "state.json").write_text(
            json.dumps({"schema": "jev-nethack-until-win/v1", "activeFragment": None}),
            encoding="utf-8",
        )
        recovered = self.root / "recovered-training"
        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=None, runtime_root=str(self.root / "runtime"),
            recovered_training_root=str(recovered), site_url=None, token_file=None,
            github_repo=None, gh="gh", batch_seconds=600,
        )
        first = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
        self.assertEqual(first["errors"], [])
        self.assertEqual(first["sources"], 1)
        self.assertEqual(first["trainingRecovery"][0]["status"], "recovered")
        recovered_marker = next(recovered.glob("fragments/*/training/*.complete.json"))
        recovered_items = publisher.parse_training_marker(
            recovered_marker, recovered, self.root / "derived-check",
        )
        self.assertEqual(len(recovered_items), 1)
        self.assertFalse(marker.exists(), "source crash fragment must remain unmodified")
        self.assertEqual(source_bytes, {path: path.read_bytes() for path in source_bytes})
        second = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
        self.assertEqual(second["errors"], [])
        self.assertEqual(second["trainingRecovery"], [])
        self.assertEqual(second["sources"], 1)

    def test_run_once_recovers_and_collects_inactive_video_tail_without_source_mutation(self) -> None:
        data, fragment = self.make_crashed_broadcast_stub()
        source_before = {
            path: path.read_bytes() for path in fragment.rglob("*") if path.is_file()
        }
        runtime = self.root / "runtime-video"
        recovered_root = runtime / "recovered-recordings"
        producer = SimpleNamespace(pid=2_147_483_647, stamp="20260918T142018.869019Z")

        def derive_video(
            source_fragment: Path, *, output_fragment: Path, derived_root: Path,
            segment_index: int, **_: object,
        ) -> dict[str, object]:
            self.assertEqual(source_fragment, fragment.resolve())
            self.assertEqual(derived_root, recovered_root.resolve())
            self.assertEqual(segment_index, 2)
            output_fragment.mkdir(mode=0o700)
            events = output_fragment / "segment-0002.jsonl"
            events.write_bytes((fragment / "segment-0002.jsonl").read_bytes())
            video = output_fragment / "segment-0002.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=black:s=64x64:d=0.2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-y", str(video)],
                check=True,
            )
            manifest = {
                "schemaVersion": 1,
                "sessionId": f"broadcast-{'a' * 32}-seg0002",
                "broadcastId": "a" * 32,
                "completed": True,
                "startedAt": START,
                "endedAt": "2026-09-18T14:00:01+00:00",
                "frameCount": 1,
                "actionCount": 1,
                "artifacts": [
                    {"filename": video.name, "bytes": video.stat().st_size,
                     "sha256": publisher.sha256_file(video), "contentType": "video/mp4"},
                    {"filename": events.name, "bytes": events.stat().st_size,
                     "sha256": publisher.sha256_file(events),
                     "contentType": "application/x-ndjson"},
                ],
            }
            manifest_path = output_fragment / "segment-0002.manifest.json"
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
            return {
                "status": "recovered", "manifestPath": str(manifest_path),
                "sourceUnmodifiedByRecovery": True,
            }

        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=str(data), runtime_root=str(runtime),
            recovered_recordings_root=str(recovered_root), site_url=None, token_file=None,
            github_repo=None, gh="gh", batch_seconds=600,
        )
        with mock.patch(
            "recover_video_tail.inspect_exact_producer", return_value=producer
        ) as inspect, mock.patch(
            "recover_video_tail.recover_video_tail", side_effect=derive_video
        ) as recover:
            first = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            second = publisher.run_once(arguments, delays=(), sleep=lambda _: None)

        self.assertEqual(first["errors"], [])
        self.assertEqual(first["sources"], 1)
        self.assertEqual(first["videoRecovery"][0]["status"], "recovered")
        self.assertEqual(second["errors"], [])
        self.assertEqual(second["sources"], 1)
        self.assertEqual(second["videoRecovery"], [])
        inspect.assert_called_once()
        recover.assert_called_once()
        self.assertEqual(
            source_before,
            {path: path.read_bytes() for path in fragment.rglob("*") if path.is_file()},
        )

    def test_video_recovery_distinguishes_deferrals_from_unsafe_mapping(self) -> None:
        from recover_video_tail import (
            ActiveFragmentDeferred, NoRecoverableTail, TailRecoveryError,
        )

        data, fragment = self.make_crashed_broadcast_stub()
        recovered_root = self.root / "recovered-video-status"
        producer = SimpleNamespace(pid=2_147_483_647, stamp="20260918T142018.869019Z")

        with mock.patch(
            "recover_video_tail.inspect_exact_producer",
            side_effect=ActiveFragmentDeferred("still active"),
        ):
            recovered, errors = publisher.recover_inactive_video_tails(data, recovered_root)
        self.assertEqual(errors, [])
        self.assertEqual(recovered[0]["status"], "deferred_active")

        with mock.patch(
            "recover_video_tail.inspect_exact_producer", return_value=producer,
        ), mock.patch(
            "recover_video_tail.recover_video_tail",
            side_effect=NoRecoverableTail("empty final segment"),
        ):
            recovered, errors = publisher.recover_inactive_video_tails(data, recovered_root)
        self.assertEqual(errors, [])
        self.assertEqual(recovered[0]["status"], "no_recoverable_tail")

        with mock.patch(
            "recover_video_tail.inspect_exact_producer",
            side_effect=TailRecoveryError("ambiguous mapping"),
        ):
            recovered, errors = publisher.recover_inactive_video_tails(data, recovered_root)
        self.assertEqual(recovered, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("ambiguous mapping", errors[0]["error"])
        self.assertFalse((fragment / "segment-0002.manifest.json").exists())

    def test_video_recovery_scan_leaves_completed_validation_to_catalog(self) -> None:
        template_root, template_manifest = self.make_recording()
        data = self.root / "video-history"
        broadcasts = data / "broadcast-fragments"
        source_fragment = broadcasts / (
            "broadcast-20260918T140000.000000Z-" + "a" * 32
        )
        source_fragment.parent.mkdir(parents=True)
        shutil.copytree(template_manifest.parent, source_fragment)

        pending_fragment = broadcasts / (
            "broadcast-20260918T140100.000000Z-" + "b" * 32
        )
        pending_fragment.mkdir()
        (pending_fragment / "segment-0002.jsonl").write_text(
            json.dumps({"sequence": 2, "stream_id": "b" * 32}) + "\n"
        )

        runtime = self.root / "video-history-runtime"
        recovered_root = runtime / "recovered-recordings"
        recovered_fragment = recovered_root / pending_fragment.name
        recovered_fragment.mkdir(parents=True)
        recovered_video = recovered_fragment / "segment-0002.mp4"
        shutil.copy2(template_manifest.parent / "segment-0001.mp4", recovered_video)
        recovered_events = recovered_fragment / "segment-0002.jsonl"
        recovered_events.write_text('{"sequence":2}\n')
        recovered_manifest = {
            "schemaVersion": 1,
            "sessionId": f"broadcast-{'b' * 32}-seg0002",
            "broadcastId": "b" * 32,
            "completed": True,
            "startedAt": START,
            "endedAt": "2026-09-18T14:00:01+00:00",
            "frameCount": 1,
            "actionCount": 1,
            "artifacts": [
                {"filename": recovered_video.name, "bytes": recovered_video.stat().st_size,
                 "sha256": publisher.sha256_file(recovered_video), "contentType": "video/mp4"},
                {"filename": recovered_events.name, "bytes": recovered_events.stat().st_size,
                 "sha256": publisher.sha256_file(recovered_events),
                 "contentType": "application/x-ndjson"},
            ],
        }
        (recovered_fragment / "segment-0002.manifest.json").write_text(
            json.dumps(recovered_manifest, separators=(",", ":"))
        )
        arguments = SimpleNamespace(
            data_root=str(data), recordings_root=str(data), runtime_root=str(runtime),
            site_url=None, token_file=None, github_repo=None, gh="gh", batch_seconds=600,
        )

        with mock.patch.object(
            publisher, "verify_mp4", wraps=publisher.verify_mp4
        ) as verify_mp4:
            first = publisher.run_once(arguments, delays=(), sleep=lambda _: None)
            first_probe_count = verify_mp4.call_count
            second = publisher.run_once(arguments, delays=(), sleep=lambda _: None)

        self.assertEqual(first["errors"], [])
        self.assertEqual(second["errors"], [])
        self.assertEqual(first["sources"], 2)
        self.assertEqual(second["sources"], 2)
        self.assertEqual(first["videoRecovery"], [])
        self.assertEqual(second["videoRecovery"], [])
        self.assertEqual(first_probe_count, 2)
        self.assertEqual(verify_mp4.call_count, first_probe_count)
        self.assertTrue(template_root.is_dir())

    def test_default_recovery_roots_reject_leaf_symlink_escape(self) -> None:
        data = self.root / "escape-data"
        data.mkdir()
        outside = self.root / "outside-recovery"
        outside.mkdir()

        for leaf, recordings_root in (
            ("recovered-training", None),
            ("recovered-recordings", str(data)),
        ):
            with self.subTest(leaf=leaf):
                runtime = self.root / f"runtime-{leaf}"
                runtime.mkdir()
                (runtime / leaf).symlink_to(outside, target_is_directory=True)
                arguments = SimpleNamespace(
                    data_root=str(data), recordings_root=recordings_root,
                    runtime_root=str(runtime), site_url=None, token_file=None,
                    github_repo=None, gh="gh", batch_seconds=600,
                )
                with self.assertRaisesRegex(publisher.PublishError, "contains a symlink"):
                    publisher.run_once(arguments, delays=(), sleep=lambda _: None)
        self.assertEqual(list(outside.iterdir()), [])

    def test_unchanged_history_uses_catalog_without_reopening_npz_or_ffprobe(self) -> None:
        data, _, _ = self.make_training(records=5, max_records=2)
        recordings, _ = self.make_recording()
        derived = self.root / "derived"
        catalog = self.root / "catalog"
        first, errors = publisher.collect_items(
            data, recordings, derived, catalog_root=catalog, now_epoch=1000,
            scrub_seconds=publisher.FULL_LOCAL_SCRUB_SECONDS,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len([item for item in first if item.kind == "training"]), 3)
        with mock.patch.object(
            publisher, "parse_training_marker", side_effect=AssertionError("historical NPZ reopened")
        ), mock.patch.object(
            publisher, "parse_recording_manifest", side_effect=AssertionError("historical MP4 reprobed")
        ):
            second, errors = publisher.collect_items(
                data, recordings, derived, catalog_root=catalog, now_epoch=1001,
                scrub_seconds=publisher.FULL_LOCAL_SCRUB_SECONDS,
            )
        self.assertEqual(errors, [])
        self.assertEqual(
            [(item.kind, item.item_id, item.source_sha256) for item in second],
            [(item.kind, item.item_id, item.source_sha256) for item in first],
        )


if __name__ == "__main__":
    unittest.main()
