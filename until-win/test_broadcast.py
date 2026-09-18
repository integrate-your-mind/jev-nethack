"""Offline lifecycle and contract tests for the local broadcast recorder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request
import sys
import subprocess
from unittest import mock
import time

sys.path.insert(0, str(Path(__file__).parent))

from broadcast import (BroadcastError, BroadcastRecorder, IngestClient,
                       _NoRedirectHandler, render_segment_ffmpeg, render_terminal_png,
                       render_terminal_ppm)
import broadcast as broadcast_module


class FakeResponse:
    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self.body = body
        self.status = status
        self.closed = False

    def read(self, *_args: object) -> bytes:
        return self.body

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def sample_state(terminal: str = "@ You see a door.") -> dict[str, object]:
    return {"player": {"score": 0, "hp": 14}, "terminal": terminal,
            "message": "", "adjacent_cells": {}, "inventory": [], "recent_actions": []}


class FakeIngest:
    base_url = "https://origin.example"

    def __init__(self, *, readback_ok: bool = True) -> None:
        self.readback_ok = readback_ok
        self.uploads: list[tuple[str, str]] = []
        self.manifests: list[str] = []
        self.frames: list[dict[str, object]] = []

    def frame(self, _stream_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.frames.append(payload)
        return {}

    def upload_artifact(self, _stream_id: str, path: Path, *, segment_id: str) -> dict[str, object]:
        self.uploads.append((segment_id, path.name))
        return {}

    def upload_manifest(self, _stream_id: str, _manifest: dict[str, object], *, segment_id: str) -> dict[str, object]:
        self.manifests.append(segment_id)
        return {}

    def readback(self, _url: str, _expected_sha256: str) -> bool:
        return self.readback_ok


def private_segment_inputs(recorder: BroadcastRecorder, root: Path) -> tuple[list[tuple[Path, float]], Path]:
    frame = root / "input.ppm"
    render_terminal_ppm("@", frame)
    return [(frame, time.time())], recorder.segment_events_path


class BroadcastTests(unittest.TestCase):
    def test_redirect_handler_never_forwards_authenticated_request(self) -> None:
        request = urllib.request.Request("https://origin.example/api/live", headers={"Authorization": "Bearer secret"})
        redirected = _NoRedirectHandler().redirect_request(request, None, 307, "redirect", {}, "https://other.example/")
        self.assertIsNone(redirected)

    def test_ingest_contract_and_secret_is_only_header(self) -> None:
        transport = FakeOpener(FakeResponse())
        client = IngestClient("https://origin.example", "private-token", retries=0, opener=transport)
        client.frame("session1", {"sequence": 4, "capturedAt": "2026-09-18T00:00:00+00:00",
                                   "state": {"terminal": "@"}, "ended": False})
        request = transport.requests[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://origin.example/api/live")
        self.assertEqual(payload["schemaVersion"], 1)
        self.assertEqual(payload["sessionId"], "session1")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-token")
        self.assertNotIn("private-token", request.data.decode())
        self.assertEqual(json.loads(request.data)["broadcastId"], "session1")

    def test_http_failure_does_not_store_response_body(self) -> None:
        secret = b"provider secret response"
        transport = FakeOpener(urllib.error.HTTPError("https://origin.example/api/live", 503, secret.decode(), {}, None))
        client = IngestClient("https://origin.example", "private-token", retries=0, opener=transport)
        with self.assertRaises(Exception):
            client.frame("session1", {"sequence": 1})
        self.assertTrue(client.failures)
        self.assertNotIn(secret.decode(), client.failures[0])

    def test_terminal_renderer_preserves_spaces_and_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blank = root / "blank.ppm"
            lower = root / "lower.ppm"
            upper = root / "upper.ppm"
            render_terminal_ppm(" " * 80, blank)
            render_terminal_ppm("a", lower)
            render_terminal_ppm("A", upper)
            blank_bytes = blank.read_bytes()
            self.assertEqual(blank_bytes.count(bytes((220, 230, 240))), 0)
            self.assertNotEqual(lower.read_bytes(), upper.read_bytes())

    def test_single_frame_segment_has_video_track_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "frame.png"
            output = root / "segment.mp4"
            render_terminal_png("ended", frame)
            render_segment_ffmpeg([(frame, time.time())], output)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_type,nb_frames,duration", "-of", "json", str(output)],
                check=True, capture_output=True, text=True)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_type"], "video")
            self.assertGreater(float(stream["duration"]), 0.0)
            self.assertGreaterEqual(int(stream["nb_frames"]), 1)

    def test_png_decodes_to_same_pixels_as_ppm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ppm = root / "frame.ppm"
            png = root / "frame.png"
            render_terminal_ppm("a A !", ppm)
            render_terminal_png("a A !", png)
            from PIL import Image
            with Image.open(ppm) as ppm_image, Image.open(png) as png_image:
                self.assertEqual(ppm_image.size, png_image.size)
                self.assertEqual(ppm_image.convert("RGB").tobytes(),
                                 png_image.convert("RGB").tobytes())

    def test_episode_termination_does_not_end_public_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeIngest()
            recorder = BroadcastRecorder(Path(temp) / "broadcast", stream_id="session1", ingest=None)
            recorder.ingest = fake  # install the fake after construction to avoid any network client
            from broadcast import LivePublisher
            recorder.live = LivePublisher(fake, "session1")
            recorder.observed_frame(state=sample_state(), phase="after_action", episode=0, step=4,
                                    terminated=True, truncated=False)
            recorder.live.close()
            recorder.close()
            self.assertEqual(len(fake.frames), 1)
            self.assertFalse(fake.frames[0]["ended"])
            self.assertTrue(fake.frames[0]["terminated"])
            self.assertEqual(fake.frames[0]["episode"], 0)
            self.assertEqual(fake.frames[0]["step"], 4)

    def test_live_close_waits_for_sender(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeIngest()
            recorder = BroadcastRecorder(Path(temp) / "broadcast", stream_id="session1", ingest=None)
            recorder.ingest = fake
            from broadcast import LivePublisher
            recorder.live = LivePublisher(fake, "session1")
            recorder.live.submit({"sequence": 0, "ended": False})
            recorder.live.close()
            recorder.close()
            self.assertFalse(recorder.live._thread.is_alive())
            self.assertEqual(len(fake.frames), 1)

    def test_readback_mismatch_prevents_manifest_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorder = BroadcastRecorder(root / "broadcast", stream_id="session1", ingest=None)
            fake = FakeIngest(readback_ok=False)
            recorder.ingest = fake
            frames, events = private_segment_inputs(recorder, root)
            with mock.patch.object(broadcast_module, "render_segment_ffmpeg",
                                   side_effect=lambda _frames, output: output.write_bytes(b"mp4")), \
                 mock.patch.object(broadcast_module, "probe_video", return_value={"stream": {"codec_type": "video"}}):
                recorder._finish_segment(1, frames, 1, events, recorder.segment_started_at)
            recorder.close()
            self.assertEqual(len(fake.uploads), 2)
            self.assertEqual(fake.manifests, [])
            self.assertTrue(any("readback hash mismatch" in item.get("upload_error", "")
                                for item in recorder.segments))

    def test_segment_uploads_share_id_and_manifest_is_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorder = BroadcastRecorder(root / "broadcast", stream_id="session1", ingest=None)
            fake = FakeIngest()
            recorder.ingest = fake
            frames, events = private_segment_inputs(recorder, root)
            with mock.patch.object(broadcast_module, "render_segment_ffmpeg",
                                   side_effect=lambda _frames, output: output.write_bytes(b"mp4")), \
                 mock.patch.object(broadcast_module, "probe_video", return_value={"stream": {"codec_type": "video"}}):
                recorder._finish_segment(1, frames, 1, events, recorder.segment_started_at)
            recorder.close()
            self.assertEqual({segment for segment, _name in fake.uploads}, {"broadcast-session1-seg0001"})
            self.assertEqual(fake.manifests, ["broadcast-session1-seg0001"])
            self.assertEqual([name for _segment, name in fake.uploads], ["segment-0001.mp4", "segment-0001.jsonl"])
            receipt = json.loads((recorder.output / "segment-0001.receipt.json").read_text())
            self.assertTrue(receipt["completed"])
            self.assertEqual(len(receipt["artifacts"]), 3)
            self.assertTrue(all(item["readback_ok"] for item in receipt["artifacts"]))

    def test_render_failure_retains_frames_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorder = BroadcastRecorder(root / "broadcast", stream_id="session1", ingest=None)
            frames, events = private_segment_inputs(recorder, root)
            with mock.patch.object(broadcast_module, "render_segment_ffmpeg",
                                   side_effect=BroadcastError("synthetic render failure")):
                recorder._finish_segment(1, frames, 1, events, recorder.segment_started_at)
            recorder.close()
            self.assertTrue(frames[0][0].exists())
            self.assertTrue(events.exists())
            self.assertIn("render_error", recorder.segments[0])

    def test_top_level_manifest_marks_render_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "broadcast"
            recorder = BroadcastRecorder(output, stream_id="session1", ingest=None)
            recorder.observed_frame(state=sample_state(), phase="before_action", episode=0, step=0)
            with mock.patch.object(broadcast_module, "render_segment_ffmpeg",
                                   side_effect=BroadcastError("synthetic render failure")):
                manifest = recorder.finalize(reason="runner_error")
            self.assertFalse(manifest["recordingComplete"])
            self.assertEqual(manifest["partialSegments"], [1])

    def test_local_recording_renders_mp4_and_matching_ndjson_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "broadcast"
            recorder = BroadcastRecorder(output, stream_id="session1", ingest=None, segment_actions=1)
            recorder.observed_frame(state=sample_state("A"), phase="before_action", episode=0, step=0)
            recorder.observed_frame(state=sample_state("a"), phase="after_action", episode=0, step=0,
                                    action={"index": 1, "keycode": 97, "label": "Press a"})
            recorder.action_finished()
            manifest = recorder.finalize(reason="action_cap")
            self.assertTrue((output / "events.jsonl").exists())
            self.assertTrue((output / "manifest.json").exists())
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertTrue(any(item["filename"].endswith(".mp4") for item in manifest["artifacts"]))
            self.assertTrue(any(item["filename"].endswith(".jsonl") for item in manifest["artifacts"]))
            self.assertTrue(manifest["recordingComplete"])
            self.assertEqual(manifest["partialSegments"], [])
            for artifact in manifest["artifacts"]:
                path = output / artifact["filename"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            self.assertGreater((output / "segment-0001.mp4").stat().st_size, 0)

    def test_invalid_ingest_origin_is_rejected(self) -> None:
        with self.assertRaises(BroadcastError):
            IngestClient("http://origin.example", "token")
        with self.assertRaises(BroadcastError):
            IngestClient("https://user:pass@origin.example", "token")


if __name__ == "__main__":
    unittest.main()
