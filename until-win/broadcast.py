"""Bounded local Jev/NetHack broadcast recorder with optional hosted ingest.

The local event log and MP4 segments are written before network delivery is
attempted.  Hosted ingest is best-effort and idempotent: a failed upload never
removes or rewrites the local recording.  Secrets are read from the environment
or private token files and are never included in event logs, manifests, or
public payloads.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping


DEFAULT_INGEST_URL = "https://jev-nethack-live.poppybyte.chatgpt.site"
DEFAULT_TOKEN_FILE = str(Path.home() / "Library" / "Application Support" / "JevNetHack" / "ingest.token")
DEFAULT_SEGMENT_SECONDS = 60.0
DEFAULT_SEGMENT_ACTIONS = 100
MAX_HTTP_RESPONSE_BYTES = 32 * 1024 * 1024
INGEST_USER_AGENT = "Jev-NetHack-Broadcaster/1.0"
TERMINAL_ROWS = 24
TERMINAL_COLS = 80
CELL_WIDTH = 10
CELL_HEIGHT = 18
FRAME_WIDTH = TERMINAL_COLS * CELL_WIDTH
FRAME_HEIGHT = TERMINAL_ROWS * CELL_HEIGHT + 18


class BroadcastError(Exception):
    """Base class for local broadcast failures."""


class IngestError(BroadcastError):
    """Bounded hosted ingest failure; local recording remains available."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Publish metadata only after its complete contents are durable."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    write_bytes_atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    """Validate that an encoded segment has a real, nonempty video track."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise BroadcastError("ffprobe is required to validate recorded video")
    command = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=codec_type,codec_name,nb_frames,duration,width,height",
               "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        raise BroadcastError("ffprobe could not validate the local segment") from exc
    streams = payload.get("streams") or []
    if not streams or streams[0].get("codec_type") != "video":
        raise BroadcastError("encoded segment has no video stream")
    stream = dict(streams[0])
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    try:
        if duration is None or float(duration) <= 0:
            raise BroadcastError("encoded segment has zero duration")
    except (TypeError, ValueError) as exc:
        raise BroadcastError("encoded segment duration is invalid") from exc
    return {"stream": stream, "format": payload.get("format", {})}


def read_secret(*, env_name: str, token_file: str | None) -> str:
    value = os.environ.get(env_name)
    if value and value.strip():
        return value.strip()
    if token_file:
        path = Path(token_file).expanduser()
        try:
            mode = path.stat().st_mode & 0o777
        except OSError as exc:
            raise BroadcastError(f"cannot read {env_name} token file") from exc
        if mode & 0o077:
            raise BroadcastError(f"{env_name} token file must not be group/world readable")
        try:
            value = path.read_text().strip()
        except OSError as exc:
            raise BroadcastError(f"cannot read {env_name} token file") from exc
        if value:
            return value
    raise BroadcastError(f"{env_name} is required")


class IngestClient:
    """Small authenticated Worker client with bounded idempotent retries."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 10,
                 retries: int = 2, opener: Callable[..., Any] | None = None) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise BroadcastError("ingest URL must be an https origin without credentials, query, or fragment")
        if timeout <= 0 or retries < 0:
            raise BroadcastError("timeout must be positive and retries non-negative")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = float(timeout)
        self.retries = retries
        self._opener = _NO_REDIRECT_OPENER.open if opener is None else opener
        self.failures: list[str] = []
        self._last_live_post = 0.0

    def _request(self, method: str, path: str, body: bytes, *, content_type: str,
                 idempotency_key: str, extra_headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        headers = {"Authorization": "Bearer " + self._token,
                   "Content-Type": content_type,
                   "Content-Length": str(len(body)),
                   "Accept": "application/json",
                   "User-Agent": INGEST_USER_AGENT,
                   "Idempotency-Key": idempotency_key}
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(
            self.base_url + "/" + path.lstrip("/"), data=body,
            headers=headers, method=method)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            response: Any = None
            try:
                response = self._opener(request, timeout=self.timeout)
                status = getattr(response, "status", None)
                if status is None and callable(getattr(response, "getcode", None)):
                    status = response.getcode()
                if status is not None and int(status) >= 400:
                    raise IngestError(f"ingest HTTP status {status}")
                try:
                    raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
                except TypeError:
                    raw = response.read()
                if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                    raise IngestError("ingest response exceeds size cap")
                if not raw:
                    return {}
                decoded = json.loads(raw.decode("utf-8"))
                return decoded if isinstance(decoded, dict) else {"response": decoded}
            except (IngestError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.05 * (2 ** attempt))
            finally:
                if response is not None and callable(getattr(response, "close", None)):
                    response.close()
        message = str(last) if isinstance(last, IngestError) else "hosted ingest transport failed"
        self.failures.append(message)
        raise IngestError(message)

    def frame(self, stream_id: str, frame: Mapping[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self._last_live_post
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        payload = dict(frame)
        payload.update({"schemaVersion": 1, "sessionId": stream_id, "broadcastId": stream_id})
        key = f"{stream_id}:frame:{frame['sequence']}"
        response = self._request("POST", "api/live", json.dumps(payload, separators=(",", ":")).encode(),
                             content_type="application/json", idempotency_key=key)
        self._last_live_post = time.monotonic()
        return response

    def upload_artifact(self, stream_id: str, path: Path, *, segment_id: str) -> dict[str, Any]:
        digest = sha256_file(path)
        key = f"{stream_id}:artifact:{segment_id}:{path.name}:{digest}"
        return self._request("PUT", f"api/recordings/{segment_id}/{path.name}", path.read_bytes(),
                             content_type="video/mp4" if path.suffix == ".mp4" else "application/x-ndjson" if path.suffix == ".jsonl" else "application/octet-stream",
                             idempotency_key=key,
                             extra_headers={"X-Content-SHA256": digest})

    def upload_manifest(self, stream_id: str, manifest: Mapping[str, Any], *, segment_id: str) -> dict[str, Any]:
        body = json.dumps(dict(manifest), separators=(",", ":")).encode()
        return self._request("PUT", f"api/recordings/{segment_id}/manifest.json", body,
                             content_type="application/json", idempotency_key=segment_id + ":manifest",
                             extra_headers={"X-Content-SHA256": hashlib.sha256(body).hexdigest()})

    def readback(self, url: str, expected_sha256: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        origin = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "https" or parsed.netloc != origin.netloc or parsed.username or parsed.password:
            raise IngestError("artifact readback URL is outside ingest origin")
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self._token,
                                                       "User-Agent": INGEST_USER_AGENT}, method="GET")
        response: Any = None
        try:
            response = self._opener(request, timeout=self.timeout)
            raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                raise IngestError("readback exceeds response-size cap")
            return hashlib.sha256(raw).hexdigest() == expected_sha256
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise IngestError("artifact readback failed") from exc
        finally:
            if response is not None and callable(getattr(response, "close", None)):
                response.close()


class LivePublisher:
    """Coalescing live sender: local gameplay never waits on hosted ingest."""

    def __init__(self, client: IngestClient, stream_id: str) -> None:
        self.client = client
        self.stream_id = stream_id
        self._queue: queue.Queue[Mapping[str, Any] | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.failures: list[str] = []
        self._thread = threading.Thread(target=self._run, name="broadcast-live-ingest", daemon=True)
        self._thread.start()

    def submit(self, payload: Mapping[str, Any]) -> None:
        try:
            self._queue.put_nowait(dict(payload))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(dict(payload))
            except queue.Full:
                pass

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                continue
            try:
                self.client.frame(self.stream_id, payload)
            except IngestError as exc:
                self.failures.append(str(exc))

    def close(self) -> None:
        self._stop.set()
        # The client has bounded request/retry timeouts.  Wait for the sender
        # so finalization cannot return while an authenticated live request is
        # still in flight and the final ended frame is still pending.
        self._thread.join()


def _terminal_image(terminal: str):
    """Build one terminal frame image used by both archival formats."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16, index=0)
    except (ImportError, OSError) as exc:
        raise BroadcastError("Pillow and macOS Menlo are required for terminal footage") from exc
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (12, 18, 28))
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(terminal.splitlines()[:TERMINAL_ROWS]):
        draw.text((0, row * CELL_HEIGHT), line[:TERMINAL_COLS], font=font, fill=(220, 230, 240))
    return image


def render_terminal_ppm(terminal: str, path: Path) -> None:
    """Render a terminal observation into an exact 800x450 PPM frame."""
    image = _terminal_image(terminal)
    image.save(path, format="PPM")


def render_terminal_png(terminal: str, path: Path) -> None:
    """Render a lossless terminal frame for transient segment input."""
    image = _terminal_image(terminal)
    image.save(path, format="PNG", optimize=True)


def render_segment_ffmpeg(frames: list[tuple[Path, float]], output: Path) -> None:
    if not frames:
        raise BroadcastError("cannot render an empty segment")
    concat = output.with_suffix(".concat.txt")
    lines: list[str] = []
    for index in range(len(frames) - 1):
        frame, timestamp = frames[index]
        duration = max(0.04, frames[index + 1][1] - timestamp)
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines += [f"file '{escaped}'", f"duration {duration:.6f}"]
    escaped = str(frames[-1][0].resolve()).replace("'", "'\\''")
    lines.append(f"file '{escaped}'")
    # The concat demuxer ignores the duration of its final file.  Repeat the
    # last observation after a short explicit hold so an ended-only segment
    # still contains a real video frame and nonzero duration.
    lines.append("duration 0.100000")
    lines.append(f"file '{escaped}'")
    concat.write_text("\n".join(lines) + "\n")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat), "-vf", "fps=10,format=yuv420p", "-movflags", "+faststart", str(output)]
    try:
        subprocess.run(command, check=True, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BroadcastError("ffmpeg could not render the local segment") from exc
    finally:
        concat.unlink(missing_ok=True)


_PUBLIC_EPISODE_STATUSES = {
    "engine_episode_limit",
    "engine_truncation",
    "game_end",
    "stop_requested",
    "verified_ascension",
}
_PUBLIC_SCORE_ERRORS = {
    "ambiguous_xlogfile",
    "episode_id_mismatch",
    "episode_seed_mismatch",
    "malformed_xlogfile",
    "missing_bound_ttyrec",
    "missing_xlogfile",
    "unbound_episode_identity",
    "unsafe_episode_directory",
    "unsafe_ttyrec_directory",
    "unsafe_ttyrec",
    "unsafe_xlogfile",
    "xlog_ttyrec_binding_mismatch",
}


def _public_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _public_text(value: Any, *, limit: int) -> str | None:
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def _public_relative_path(value: Any) -> str | None:
    text = _public_text(value, limit=256)
    if text is None or text.startswith("/") or "\\" in text:
        return None
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return text


def _sanitize_score_evidence(
    value: Any, *, episode_id: int | None, seed: int | None
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    relative_path = _public_relative_path(value.get("relativePath"))
    ttyrec_path = _public_relative_path(value.get("ttyrecRelativePath"))
    digest = value.get("sha256")
    evidence_episode_id = _public_int(value.get("episodeId"))
    evidence_seed = _public_int(value.get("seed"))
    ttyrec_bytes = _public_int(value.get("ttyrecBytes"))
    xlog_bytes = _public_int(value.get("xlogBytes"))
    if (
        value.get("source") != "native_xlogfile"
        or value.get("record") != "single_native_xlog_record"
        or relative_path is None
        or ttyrec_path is None
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or episode_id is None
        or seed is None
        or evidence_episode_id != episode_id
        or evidence_seed != seed
        or ttyrec_bytes is None
        or xlog_bytes is None
    ):
        return None
    xlog_match = re.fullmatch(r"nle-ttyrec/nle\.([1-9][0-9]*)\.xlogfile", relative_path)
    if xlog_match is None:
        return None
    pid = xlog_match.group(1)
    if re.fullmatch(rf"nle-ttyrec/nle\.{pid}\.[0-9]+\.ttyrec3\.bz2", ttyrec_path) is None:
        return None
    return {
        "source": "native_xlogfile",
        "relativePath": relative_path,
        "sha256": digest,
        "xlogBytes": xlog_bytes,
        "record": "single_native_xlog_record",
        "ttyrecRelativePath": ttyrec_path,
        "ttyrecBytes": ttyrec_bytes,
        "episodeId": evidence_episode_id,
        "seed": evidence_seed,
    }


def _sanitize_episode_result(row: Mapping[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key in (
        "episodeId",
        "seed",
        "score",
        "officialFinalScore",
        "lastObservedScore",
        "maxObservedScore",
        "maxDepth",
        "steps",
    ):
        if key in row and row[key] is None and key in {
            "score",
            "officialFinalScore",
            "lastObservedScore",
            "maxObservedScore",
        }:
            public[key] = None
            continue
        value = _public_int(row.get(key))
        if value is not None:
            public[key] = value
    reward = row.get("totalReward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        try:
            public_reward = float(reward)
        except (OverflowError, ValueError):
            pass
        else:
            if math.isfinite(public_reward):
                public["totalReward"] = public_reward
    semantics = row.get("scoreSemantics")
    if semantics in {"official_final", "last_observed"}:
        public["scoreSemantics"] = semantics
    status = row.get("status")
    if status in _PUBLIC_EPISODE_STATUSES:
        public["status"] = status
    end_status = row.get("endStatus")
    if end_status is None and "endStatus" in row:
        public["endStatus"] = None
    elif isinstance(end_status, str) and re.fullmatch(r"[A-Z_]{1,32}", end_status):
        public["endStatus"] = end_status
    for key, limit in (("deathCause", 256), ("deathWhile", 128)):
        value = row.get(key)
        if value is None and key in row:
            public[key] = None
        else:
            text = _public_text(value, limit=limit)
            if text is not None:
                public[key] = text
    error = row.get("officialScoreError")
    if error in _PUBLIC_SCORE_ERRORS:
        public["officialScoreError"] = error
    evidence = _sanitize_score_evidence(
        row.get("officialScoreEvidence"),
        episode_id=public.get("episodeId"),
        seed=public.get("seed"),
    )
    if evidence is not None:
        public["officialScoreEvidence"] = evidence
    for key in ("isAscended", "terminated", "truncated"):
        if isinstance(row.get(key), bool):
            public[key] = row[key]
    return public


class BroadcastRecorder:
    def __init__(self, output: Path, *, stream_id: str, ingest: IngestClient | None,
                 segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
                 segment_actions: int = DEFAULT_SEGMENT_ACTIONS) -> None:
        if segment_seconds <= 0 or segment_actions <= 0:
            raise BroadcastError("segment bounds must be positive")
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.stream_id = stream_id
        self.ingest = ingest
        self.live = LivePublisher(ingest, stream_id) if ingest else None
        self.segment_seconds = segment_seconds
        self.segment_actions = segment_actions
        self.events_path = output / "events.jsonl"
        self.events = self.events_path.open("x", encoding="utf-8")
        self.frames_root = output / ".frames"
        self.frames_root.mkdir()
        self.segment_index = 0
        self.segment_action_count = 0
        self.segment_started = time.time()
        self.segment_frames: list[tuple[Path, float]] = []
        self.segment_started_at = utc_now()
        self.segment_events_path = output / "segment-0001.jsonl"
        self.segment_events = self.segment_events_path.open("x", encoding="utf-8")
        self.segments: list[dict[str, Any]] = []
        self.frame_sequence = 0
        self.ingest_failures: list[str] = []
        self.last_state: dict[str, Any] | None = None
        self.last_episode = 0
        self.last_step = 0
        self.last_action: Any = None
        self.last_decision: dict[str, Any] | None = None
        self.public_metrics: dict[str, Any] = {}
        self.runtime_status: dict[str, Any] = {"phase": "starting"}
        self.last_public_frame: dict[str, Any] | None = None
        self.started_at = utc_now()
        self.action_count = 0
        self._jobs: list[threading.Thread] = []
        self._jobs_lock = threading.Lock()

    def close(self) -> None:
        if not self.events.closed:
            self.events.close()
        if not self.segment_events.closed:
            self.segment_events.close()

    def _write_event(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.events.write(line + "\n")
        self.events.flush()
        self.segment_events.write(line + "\n")
        self.segment_events.flush()

    def set_telemetry(self, *, metrics: Mapping[str, Any] | None = None,
                      runtime_status: Mapping[str, Any] | None = None) -> None:
        """Expose only explicit public counters; provider bodies remain local."""
        if metrics is not None:
            public = {"scope": "continuous_run"}
            for key in ("totalActions", "completedEpisodes", "ascensions", "interruptedEpisodes",
                        "bestScore", "maxDepth", "recoveries", "deaths"):
                value = metrics.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    public[key] = value
            results = metrics.get("episodeResults")
            if isinstance(results, list):
                public["episodeResults"] = [
                    _sanitize_episode_result(row)
                    for row in results[-100:]
                    if isinstance(row, Mapping)
                ]
                public["episodeResultsLimit"] = 100
            self.public_metrics = public
        if runtime_status is not None:
            self.runtime_status = {key: runtime_status[key] for key in ("phase", "errorType", "retryAt")
                                   if isinstance(runtime_status.get(key), str)}

    def report_status(self, phase: str, *, error_type: str | None = None,
                      retry_at: str | None = None) -> None:
        """Report a wait/recovery without presenting the old game frame as new."""
        self.set_telemetry(runtime_status={"phase": phase, "errorType": error_type, "retryAt": retry_at})
        self._write_event({"event_type": "runtime_status", "wall_time": utc_now(),
                           "stream_id": self.stream_id, "runtime": self.runtime_status})
        if self.live is not None and self.last_public_frame is not None:
            public = dict(self.last_public_frame)
            public.update(sequence=self.frame_sequence, runtime=self.runtime_status,
                          recovery=self.runtime_status, metrics=self.public_metrics, heartbeatAt=utc_now())
            self.frame_sequence += 1
            # capturedAt deliberately remains the time of the last game observation.
            self.live.submit(public)
            self.last_public_frame = public

    def observed_frame(self, *, state: Mapping[str, Any], phase: str, episode: int, step: int,
                       action: Any = None, reward: float | None = None,
                       terminated: bool = False, truncated: bool = False,
                       is_ascended: bool = False, broadcast_ended: bool = False) -> dict[str, Any]:
        now = time.time()
        terminal = state.get("terminal", "") if isinstance(state, Mapping) else ""
        if not isinstance(terminal, str):
            terminal = str(terminal)
        sequence = self.frame_sequence
        self.frame_sequence += 1
        frame_path = self.frames_root / f"segment-{self.segment_index:04d}-frame-{sequence:08d}.png"
        render_terminal_png(terminal, frame_path)
        event = {"event_type": "observed_frame", "phase": phase, "sequence": sequence,
                 "wall_time": utc_now(), "wall_epoch": now, "stream_id": self.stream_id,
                 "episode": episode, "step": step, "state": dict(state),
                 "terminal": terminal, "action": action, "reward": reward,
                 "terminated": bool(terminated), "truncated": bool(truncated), "broadcast_ended": bool(broadcast_ended)}
        self.last_state = dict(state)
        self.last_episode = episode
        self.last_step = step
        if action is not None:
            self.last_action = action
        self._write_event(event)
        self.segment_frames.append((frame_path, now))
        if self.ingest:
            public_action = action if action is not None else self.last_action
            public = {"capturedAt": event["wall_time"], "sequence": sequence,
                      "episode": episode, "step": step,
                      "phase": phase, "decision": self.last_decision,
                      "metrics": self.public_metrics, "runtime": self.runtime_status,
                      "recovery": self.runtime_status,
                      "state": {"player": state.get("player"), "adjacent_cells": state.get("adjacent_cells"),
                                "message": state.get("message"), "inventory": state.get("inventory"),
                                "recent_actions": state.get("recent_actions"), "terminal": terminal},
                      "action": public_action, "score": (state.get("player") or {}).get("score") if isinstance(state.get("player"), Mapping) else None,
                      "isAscended": bool(is_ascended), "ended": bool(broadcast_ended),
                      "terminated": bool(terminated), "truncated": bool(truncated)}
            self.last_public_frame = public
            try:
                if self.live is None:
                    raise AttributeError
                self.live.submit(public)
            except AttributeError:
                self.ingest_failures.append("live publisher unavailable")
        return event

    def decision(self, *, episode: int, step: int, decision: Mapping[str, Any],
                 criteria: Mapping[str, str] | None = None) -> None:
        # Provider usage and request digests are useful locally, but are excluded
        # from public payloads to keep the public stream provider-neutral.
        safe = {"choice": decision.get("choice"), "confidence": decision.get("confidence"),
                "probabilities": decision.get("probabilities"), "model": decision.get("model")}
        labels = criteria if criteria is not None else decision.get("criteria", {})
        probabilities = decision.get("probabilities")
        self.last_decision = None
        if isinstance(probabilities, Mapping) and isinstance(labels, Mapping):
            valid = {key: float(value) for key, value in probabilities.items()
                     if isinstance(key, str) and isinstance(value, (int, float))
                     and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1}
            if len(valid) == len(probabilities) and decision.get("choice") in valid:
                self.last_decision = {**safe, "probabilities": valid,
                                      "criteria": {key: str(labels[key]) for key in valid if key in labels},
                                      "episode": episode, "step": step}
            else:
                self.last_decision = None
        self._write_event({"event_type": "decision", "wall_time": utc_now(), "wall_epoch": time.time(),
                           "stream_id": self.stream_id, "episode": episode, "step": step, "decision": safe})

    def action_finished(self) -> None:
        self.segment_action_count += 1
        self.action_count += 1

    def should_finalize_segment(self) -> bool:
        return self.segment_action_count >= self.segment_actions or time.time() - self.segment_started >= self.segment_seconds

    def finalize_segment(self) -> None:
        if not self.segment_frames:
            return
        self.segment_index += 1
        index = self.segment_index
        frames = self.segment_frames
        action_count = self.segment_action_count
        segment_path = self.segment_events_path
        segment_started_at = self.segment_started_at
        self.segment_frames = []
        self.segment_action_count = 0
        self.segment_started = time.time()
        self.segment_started_at = utc_now()
        self.segment_events.close()
        self.segment_events_path = self.output / f"segment-{index + 1:04d}.jsonl"
        self.segment_events = self.segment_events_path.open("x", encoding="utf-8")
        job = threading.Thread(target=self._finish_segment, args=(index, frames, action_count, segment_path, segment_started_at), daemon=True)
        with self._jobs_lock:
            self._jobs.append(job)
        job.start()

    def _finish_segment(self, index: int, frames: list[tuple[Path, float]], action_count: int,
                        segment_events_path: Path, started_at: str) -> None:
        output = self.output / f"segment-{index:04d}.mp4"
        segment_id = f"broadcast-{self.stream_id}-seg{index:04d}"
        receipt_path = self.output / f"segment-{index:04d}.receipt.json"
        receipt: dict[str, Any] = {"schemaVersion": 1, "segmentId": segment_id,
                                   "streamId": self.stream_id, "startedAt": started_at,
                                   "frameCount": len(frames), "actionCount": action_count,
                                   "completed": False, "artifacts": []}
        rendered = False
        artifacts: list[dict[str, Any]] = []
        try:
            render_segment_ffmpeg(frames, output)
            media_ffprobe = probe_video(output)
            receipt["media_ffprobe"] = media_ffprobe
            artifact = {"index": index, "path": output.name, "jsonl_path": segment_events_path.name,
                        "sha256": sha256_file(output), "frames": len(frames), "actions": action_count,
                        "startedAt": started_at, "endedAt": utc_now()}
            artifacts = [{"filename": output.name, "sha256": artifact["sha256"], "bytes": output.stat().st_size,
                          "contentType": "video/mp4"},
                         {"filename": segment_events_path.name, "sha256": sha256_file(segment_events_path),
                          "bytes": segment_events_path.stat().st_size, "contentType": "application/x-ndjson"}]
            if output.stat().st_size > 32 * 1024 * 1024:
                raise BroadcastError("segment MP4 exceeds Worker 32 MiB upload cap")
            segment_manifest = {"schemaVersion": 1, "sessionId": segment_id, "broadcastId": self.stream_id,
                                "completed": True, "startedAt": started_at, "endedAt": artifact["endedAt"],
                                "frameCount": len(frames), "actionCount": action_count, "artifacts": artifacts}
            manifest_bytes = json.dumps(segment_manifest, separators=(",", ":")).encode()
            write_bytes_atomic(self.output / f"segment-{index:04d}.manifest.json", manifest_bytes)
            receipt["artifacts"] = [dict(item, upload_url=None, readback_ok=None) for item in artifacts]
            receipt["artifacts"].append({"filename": "manifest.json",
                                         "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                         "bytes": len(manifest_bytes), "contentType": "application/json",
                                         "upload_url": None, "readback_ok": None})
            rendered = True
            if self.ingest:
                response = self.ingest.upload_artifact(self.stream_id, output, segment_id=segment_id)
                mp4_url = response.get("download_url") or response.get("artifact_url") or f"{self.ingest.base_url}/api/recordings/{segment_id}/{output.name}"
                receipt["artifacts"][0]["upload_url"] = mp4_url
                artifact["cloud_mp4_sha256_ok"] = self.ingest.readback(mp4_url, artifact["sha256"])
                receipt["artifacts"][0]["readback_ok"] = artifact["cloud_mp4_sha256_ok"]
                response = self.ingest.upload_artifact(self.stream_id, segment_events_path, segment_id=segment_id)
                jsonl_url = response.get("download_url") or response.get("artifact_url") or f"{self.ingest.base_url}/api/recordings/{segment_id}/{segment_events_path.name}"
                receipt["artifacts"][1]["upload_url"] = jsonl_url
                artifact["cloud_jsonl_sha256_ok"] = self.ingest.readback(jsonl_url, artifacts[1]["sha256"])
                receipt["artifacts"][1]["readback_ok"] = artifact["cloud_jsonl_sha256_ok"]
                if not artifact["cloud_mp4_sha256_ok"] or not artifact["cloud_jsonl_sha256_ok"]:
                    raise IngestError("segment readback hash mismatch")
                response = self.ingest.upload_manifest(self.stream_id, segment_manifest, segment_id=segment_id)
                manifest_url = response.get("download_url") or response.get("artifact_url") or f"{self.ingest.base_url}/api/recordings/{segment_id}/manifest.json"
                receipt["artifacts"][2]["upload_url"] = manifest_url
                artifact["cloud_manifest_sha256_ok"] = self.ingest.readback(manifest_url, hashlib.sha256(manifest_bytes).hexdigest())
                receipt["artifacts"][2]["readback_ok"] = artifact["cloud_manifest_sha256_ok"]
                if not artifact["cloud_manifest_sha256_ok"]:
                    raise IngestError("segment manifest readback hash mismatch")
            with self._jobs_lock:
                self.segments.append({**artifact, "artifacts": artifacts, "segment_id": segment_id})
            receipt["completed"] = True
        except IngestError as exc:
            receipt["error"] = str(exc)
            with self._jobs_lock:
                self.ingest_failures.append(str(exc))
                if rendered and "artifact" in locals():
                    self.segments.append({**artifact, "artifacts": artifacts, "segment_id": segment_id,
                                          "upload_error": str(exc)})
        except BroadcastError as exc:
            receipt["error"] = str(exc)
            with self._jobs_lock:
                self.ingest_failures.append(str(exc))
            with self._jobs_lock:
                self.segments.append({"index": index, "path": output.name if output.exists() else None,
                                      "jsonl_path": segment_events_path.name, "frames": len(frames),
                                      "actions": action_count, "render_error": str(exc),
                                      "artifacts": [{"filename": segment_events_path.name,
                                                     "sha256": sha256_file(segment_events_path),
                                                     "bytes": segment_events_path.stat().st_size,
                                                     "contentType": "application/x-ndjson"}]})
        finally:
            receipt["endedAt"] = utc_now()
            write_json_atomic(receipt_path, receipt)
            if rendered:
                for frame_path, _ in frames:
                    frame_path.unlink(missing_ok=True)

    def _wait_jobs(self) -> None:
        with self._jobs_lock:
            jobs = list(self._jobs)
        for job in jobs:
            # Each owned job has bounded ffmpeg and HTTP operations.  Waiting
            # for completion keeps the local manifest truthful: a completed
            # manifest must include every segment that was detached from the
            # live recorder.
            job.join()

    def finalize(self, *, reason: str) -> dict[str, Any]:
        if self.last_state is not None:
            self.observed_frame(state=self.last_state, phase="ended", episode=self.last_episode,
                                step=self.last_step, broadcast_ended=True)
        self.finalize_segment()
        if self.live:
            self.live.close()
            self.ingest_failures.extend(self.live.failures)
        self._wait_jobs()
        self.segment_events.close()
        self.segment_events_path.unlink(missing_ok=True)
        self.close()
        partial_segments = sorted(
            int(segment["index"])
            for segment in self.segments
            if segment.get("render_error") or not (self.output / f"segment-{int(segment['index']):04d}.mp4").exists()
        )
        archive_verified: bool | None
        if self.ingest is None:
            archive_verified = None
        else:
            archive_verified = bool(self.segments) and all(
                segment.get("cloud_mp4_sha256_ok") is True
                and segment.get("cloud_jsonl_sha256_ok") is True
                and segment.get("cloud_manifest_sha256_ok") is True
                for segment in self.segments
            )
        segment_receipts: list[dict[str, Any]] = []
        for segment in sorted(self.segments, key=lambda item: int(item["index"])):
            receipt_file = self.output / f"segment-{int(segment['index']):04d}.receipt.json"
            if receipt_file.exists():
                try:
                    segment_receipts.append(json.loads(receipt_file.read_text()))
                except (OSError, ValueError):
                    segment_receipts.append({"segmentId": segment.get("segment_id"), "error": "receipt unreadable"})
        artifacts = []
        for segment in self.segments:
            artifacts.extend(segment["artifacts"])
        artifacts.append({"filename": self.events_path.name, "sha256": sha256_file(self.events_path),
                          "bytes": self.events_path.stat().st_size, "contentType": "application/x-ndjson"})
        manifest = {"schemaVersion": 1, "sessionId": self.stream_id, "completed": True,
                    "startedAt": self.started_at, "endedAt": utc_now(),
                    "frameCount": self.frame_sequence, "actionCount": self.action_count,
                    "artifacts": artifacts, "recordingComplete": not partial_segments,
                    "partialSegments": partial_segments, "archiveVerified": archive_verified,
                    "segments": [{"segmentId": item.get("segmentId"), "completed": item.get("completed", False),
                                  "receipt": f"segment-{int(segment['index']):04d}.receipt.json"}
                                 for item, segment in zip(segment_receipts, sorted(self.segments, key=lambda x: int(x["index"]))) ]}
        manifest_path = self.output / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        manifest["segments"] = segment_receipts
        write_json_atomic(self.output / "manifest.receipt.json", manifest)
        return manifest


def run_broadcast(args: argparse.Namespace) -> Path:
    from jev_client import JevBudgetExceededError, JevClient, JevClientError
    import run
    try:
        from menu_labels import contextual_choices
    except ImportError:
        contextual_choices = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream_id = secrets.token_hex(16)
    source_files = [Path(__file__), Path(__file__).with_name("run.py"), Path(__file__).with_name("jev_client.py"), Path(__file__).with_name("menu_labels.py")]
    config = {"schemaVersion": 1, "stream_id": stream_id, "started_at": utc_now(),
              "source_sha256": {path.name: sha256_file(path) for path in source_files if path.exists()},
              "settings": {key: value for key, value in vars(args).items() if "token" not in key}}
    ingest = None
    if not args.no_ingest:
        ingest_token = read_secret(env_name="INGEST_TOKEN", token_file=args.ingest_token_file)
        ingest = IngestClient(args.ingest_url, ingest_token, timeout=args.ingest_timeout, retries=args.ingest_retries)
    jev_key = read_secret(env_name="TYPESAFE_API_KEY", token_file=args.jev_token_file)
    client = JevClient(max_calls=args.max_calls, max_input_bytes=args.max_input_bytes, api_key=jev_key)
    recorder = BroadcastRecorder(output, stream_id=stream_id, ingest=ingest,
                                 segment_seconds=args.segment_seconds, segment_actions=args.segment_actions)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    stop = {"requested": False}
    previous_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    def handle_sigint(_signum: int, _frame: Any) -> None:
        stop["requested"] = True
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    episodes = 0
    total_actions = 0
    started = time.monotonic()
    reason = "duration_cap"
    failure_type: str | None = None
    try:
        env = run.make_env(args.character)
        try:
            seed = args.seed
            env.unwrapped.seed(core=seed, disp=seed + 100000, lgen=seed + 200000, reseed=False)
            obs, _ = env.reset()
            while not stop["requested"] and time.monotonic() - started < args.duration_seconds and total_actions < args.max_actions:
                visits, history = Counter(), deque(maxlen=8)
                episode_steps = 0
                terminated = truncated = False
                while not stop["requested"] and episode_steps < args.max_episode_steps and time.monotonic() - started < args.duration_seconds and total_actions < args.max_actions:
                    before = run.stats(obs)
                    visits[(before["dungeon"], before["level"], before["x"], before["y"])] += 1
                    state = run.make_state(obs, visits, history)
                    criteria = run.action_choices(env.unwrapped.actions)
                    if contextual_choices:
                        criteria = contextual_choices(state, env.unwrapped.actions, criteria)
                    recorder.observed_frame(state=state, phase="before_action", episode=episodes, step=episode_steps)
                    decision = client.choose(state, criteria, run.INSTRUCTIONS)
                    recorder.decision(episode=episodes, step=episode_steps, decision=decision)
                    action = int(decision["choice"][1:])
                    obs, reward, terminated, truncated, info = env.step(action)
                    after = run.stats(obs)
                    after_state = run.make_state(obs, visits, history)
                    action_record = {"index": action, "keycode": int(env.unwrapped.actions[action]),
                                     "label": criteria[decision["choice"]]}
                    recorder.observed_frame(state=after_state, phase="after_action", episode=episodes, step=episode_steps,
                                            action=action_record, reward=float(reward), terminated=bool(terminated),
                                            truncated=bool(truncated), is_ascended=bool(info.get("is_ascended", False)))
                    history.append({"action": criteria[decision["choice"]], "before": before, "after": after,
                                    "message_after": run.text_line(obs["message"])})
                    recorder.action_finished()
                    total_actions += 1
                    episode_steps += 1
                    if recorder.should_finalize_segment():
                        recorder.finalize_segment()
                    if terminated or truncated:
                        break
                if stop["requested"]:
                    reason = "interrupt"
                    break
                if total_actions >= args.max_actions:
                    reason = "action_cap"
                    break
                if time.monotonic() - started >= args.duration_seconds:
                    reason = "duration_cap"
                    break
                # Episode action caps and game ends restart with explicit NLE seeds.
                episodes += 1
                seed += 1
                env.close()
                env = run.make_env(args.character)
                env.unwrapped.seed(core=seed, disp=seed + 100000, lgen=seed + 200000, reseed=False)
                obs, _ = env.reset()
                if recorder.should_finalize_segment():
                    recorder.finalize_segment()
            if reason == "duration_cap" and stop["requested"]:
                reason = "interrupt"
        finally:
            env.close()
    except KeyboardInterrupt:
        reason = "interrupt"
    except JevBudgetExceededError:
        reason = "budget_cap"
        failure_type = "JevBudgetExceededError"
    except JevClientError as exc:
        reason = "jev_error"
        failure_type = type(exc).__name__
    except Exception as exc:
        reason = "runner_error"
        failure_type = type(exc).__name__
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        signal.signal(signal.SIGTERM, previous_term_handler)
        manifest = recorder.finalize(reason=reason)
        summary = {"stream_id": stream_id, "actions": total_actions, "episodes": episodes + 1,
                   "wall_seconds": time.monotonic() - started, "jev_calls": client.calls_used,
                   "jev_input_bytes": client.input_bytes_used, "reason": reason,
                   "api_cost_note": "Provider estimate intentionally omitted from public payloads.",
                   "manifest_sha256": manifest.get("manifest_sha256"), "failure_type": failure_type,
                   "ingest_failures": recorder.ingest_failures}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ingest-url", default=DEFAULT_INGEST_URL)
    parser.add_argument("--ingest-token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--jev-token-file", default=None, help="private TYPESAFE_API_KEY file; env is preferred")
    parser.add_argument("--no-ingest", action="store_true", help="record locally without hosted delivery")
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--max-actions", type=int, default=1000000)
    parser.add_argument("--max-episode-steps", type=int, default=2048)
    parser.add_argument("--max-calls", type=int, default=6000)
    parser.add_argument("--max-input-bytes", type=int, default=144 * 1024 * 1024)
    parser.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS)
    parser.add_argument("--segment-actions", type=int, default=DEFAULT_SEGMENT_ACTIONS)
    parser.add_argument("--ingest-timeout", type=float, default=10)
    parser.add_argument("--ingest-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--character", default="mon-hum-neu-mal")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if min(args.duration_seconds, args.max_actions, args.max_episode_steps, args.max_calls, args.max_input_bytes, args.segment_seconds, args.segment_actions) <= 0:
        parser.error("duration and all caps must be positive")
    run_broadcast(args)
