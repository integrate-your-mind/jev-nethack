# Jev-NetHack broadcast runner

`broadcast.py` records every observed terminal state locally and optionally
publishes a coalesced live view to the Sites Worker. Local recording is the
source of truth: each observation is appended to `events.jsonl`, rendered with
the macOS Menlo font into an 800x450 frame, and retained until its segment has
been rendered successfully. `ffmpeg` turns each 60-second or 100-action
segment into an MP4. A segment's MP4 and matching NDJSON are uploaded under the
same Worker segment ID, followed by its manifest only after both artifact
readbacks match their local SHA-256 values.

The live sender runs in a one-item coalescing background queue. Gameplay and
local logging continue if a live POST, upload, or readback fails. Live posts
are throttled to the Worker same-key limit and use the descriptive
`Jev-NetHack-Broadcaster/1.0` user agent. Authenticated writes do not follow
redirects. Readbacks are restricted to the configured HTTPS origin. Tokens are
read from environment variables or private mode-600 token files and are not
written to the event log, config, manifest, or public state.

The runner uses the existing `run.make_env`, `run.make_state`, `run.stats`,
`run.action_choices`, and `JevClient`. It records action objects with the action
index, keyboard code, and contextual menu label. Episode termination is stored
locally as `terminated`/`truncated`; the Worker `ended` flag is reserved for
the final broadcast frame. SIGINT and SIGTERM finalize the local recording.
Budget and runner failures are recorded in `summary.json` as distinct reasons.

## Runtime and commands

The tested isolated interpreter is:

```text
isolated-venv/bin/python
```

It contains the frozen NLE, Gymnasium, NumPy, Torch, and Pillow requirements;
`ffmpeg` must be available on `PATH`. From `outputs/jev-nethack`, a bounded
authenticated smoke run is:

```sh
PYTHONDONTWRITEBYTECODE=1 isolated-venv/bin/python broadcast.py \
  --output results/broadcast-smoke \
  --ingest-url configured-public-site \
  --ingest-token-file work/live-ingest.token \
  --duration-seconds 60 --max-actions 12 --max-episode-steps 2048 \
  --max-calls 12 --max-input-bytes 288000 --segment-actions 12 --seed 101
```

The planned one-hour run uses the defaults selected by the live decision:
`--duration-seconds 3600 --max-calls 6000
--max-input-bytes 150994944 --max-episode-steps 2048`, with the ingest token
file above and a new output directory. Use a new directory for every run;
the runner refuses to overwrite one.

For an offline local recording, pass `--no-ingest`; the Jev API key is still
required because actions are selected by the direct Jev client. `config.json`
contains source SHA-256 hashes and non-secret settings, while
`manifest.receipt.json` records the local manifest hash. `recording-preview.png`
is a local renderer check and is not a public upload artifact.

## Validation and limitations

`test_broadcast.py` exercises the redirect guard, bounded/sanitized transport
errors, Worker payload fields, Menlo rendering of spaces and case, episode
termination versus public broadcast end, same-segment artifact IDs, manifest
ordering and readback gating, render-failure retention, and a local MP4 plus
NDJSON manifest. The full NLE/ffmpeg lifecycle tests require the isolated
runtime and macOS Menlo font.

Network delivery is best effort and a process killed during an active ffmpeg
or upload operation can leave a local segment pending; local frames and logs
remain for recovery. The live queue intentionally coalesces intermediate
frames, so the public stream is not an exhaustive action log. The local
recording is bounded by the command-line duration, action, episode, Jev call,
and input-byte caps; it is a pilot broadcast, not a claim of NetHack mastery
or a production recording service.
