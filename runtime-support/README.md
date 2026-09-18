# Jev NetHack archive publisher

This directory contains the restart-safe archive sidecar for the `until-win`
runner. Keep it at `runtime-support/` beside the repository's existing
`until-win/` directory; the recovery modules deliberately import the exact
writer and recording contracts from that sibling.

The sidecar publishes only closed, checksummed artifacts. It does not make game
decisions, alter the active game, post live frames, or regenerate model inputs.
Native and crash-recovered training shards follow the same public schema.
Interrupted video recovery copies the original final JSONL and renders retained
PNG frames into a separate publisher-owned directory, leaving the producer's
fragment unchanged.

The [recording storage policy](../docs/recording-storage.md) bounds the local
closed MP4/segment JSONL cache at 64 MiB by default. Eviction requires fresh
remote verification and durable deletion receipts. Configure the continuous
runner's `--recording-tombstone-root` to `<runtime-root>/recording-tombstones`.
`--recording-cache-bytes 0` disables recording eviction. Derived upload tar
copies have their own 64 MiB default cache (`--release-cache-bytes`); pending
or unverified copies are preserved. The recorder checks both backlogs.

## Requirements

- macOS with Python and the dependencies from `until-win/requirements.txt`
- `ffmpeg` and `ffprobe` on `PATH`
- GitHub CLI (`gh`) authenticated for the target repository when GitHub
  publication is enabled
- A private Site ingest-token file, mode `0600`, when Site publication is
  enabled

The release was verified with Python 3.12.13, FFmpeg/ffprobe 8.1.1, and GitHub
CLI 2.93.0. These versions record the tested environment; they are not claimed
as broad compatibility bounds.

Create the project environment from the repository root if one is not already
available:

```sh
cd /absolute/path/to/jev-nethack
python3 -m venv .venv
.venv/bin/python -m pip install -r until-win/requirements.txt
```

## Verify the package

With `runtime-support/` and `until-win/` as siblings, run from the
`runtime-support/` directory so the frozen tests import the local publisher:

```sh
cd /absolute/path/to/jev-nethack/runtime-support
PYTHONDONTWRITEBYTECODE=1 \
  ../.venv/bin/python -m unittest -v \
  test_publish_archives.py \
  test_recover_training_tail.py \
  test_recover_video_tail.py \
  test_video_retention.py
```

The recording-retention update passed 72 publisher, recovery and retention tests. The earlier archive release passed 64 publisher and recovery tests. The corresponding
writer and recorder passed 25 producer tests, and the Site training API passed
its contract suite. Tests use temporary directories and do not publish, start a
game, install a service, or write to a configured Site or GitHub repository.
The staged copy was also rerun against the public `until-win/` tree at commit
`dfeab45915b072c9b426ac1d5f56d5c85e69f96f`; the temporary test link was
removed afterward.

The following table records the earlier archive release snapshot; the updated
recording-retention source is covered by `PUBLIC_MANIFEST.json`:

| File | SHA-256 |
| --- | --- |
| `publish_archives.py` | `21b5479abbae23967d30b32c6482b0c1e67f1d99793095e00d00c7d2fa4130e6` |
| `test_publish_archives.py` | `051ca84c8fba41d33cf47e8024db8bd9351fd1cf9cb39f7c59b2b93c8e40ef4f` |
| `recover_training_tail.py` | `7b9d744617957019cc975c9c6370a8712340424374fd4dbdc731efbea9094b5f` |
| `test_recover_training_tail.py` | `16fe6670f8b2b452cad98cba215dfa44b51a8e425499a81751f23ce35371c955` |
| `recover_video_tail.py` | `b04e4c63a5d84ad1e3bac183f8871535f2674931f71b20374dd7723666593e3c` |
| `test_recover_video_tail.py` | `59bb2a56d01f6145d6fad0b33c57d55d070e7e8b44762238e3891ce43889a03f` |

## Configure a one-shot run

Use absolute paths. The recording root should be the runner's
`recordings/until-win/broadcast-fragments` directory. The data root is its
parent `recordings/until-win` directory because exact crash recovery also reads
`state.json`, fragment configs, and the process ledger.

```sh
REPO_ROOT=/absolute/path/to/jev-nethack
DATA_ROOT=/absolute/path/to/private-data/recordings/until-win
PUBLISHER_STATE=/absolute/path/to/private-data/publisher-state
TOKEN_FILE=/absolute/path/to/private-data/ingest.token

"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/runtime-support/publish_archives.py" \
  --data-root "${DATA_ROOT}" \
  --recordings-root "${DATA_ROOT}/broadcast-fragments" \
  --runtime-root "${PUBLISHER_STATE}" \
  --site-url 'https://your-site.example' \
  --token-file "${TOKEN_FILE}" \
  --github-repo OWNER/REPOSITORY \
  --gh /absolute/path/to/gh \
  --once
```

Site publishing requires both `--site-url` and `--token-file`. Omit both to
disable Site publication. Omit `--github-repo` to disable GitHub publication.
The JSON report must be inspected: process exit 0 with `"errors":[]` is the
one-shot success condition. A provider response alone is not sufficient; the
publisher verifies the returned bytes.

The default derived roots are `recovered-training/` and
`recovered-recordings/` under the private runtime root. They must be real
directories, not symlinks. Closed source files, derived artifacts, receipts,
outbox records, and prepared release archives must be retained for reliable
restart and audit, except MP4/segment JSONL copies explicitly evicted under
the verified recording policy. Keep their tombstones and manifests.

## Publication and retry behavior

For Site publication, artifacts are uploaded before the manifest. Every upload
is followed by anonymous readback and exact size/SHA-256 verification. A `409`
is treated as ambiguous until the existing object is read back and matches.

For GitHub publication, deterministic archives use content-hash asset names.
The publisher inspects existing releases and assets before create or upload,
does not clobber a different asset, and verifies the downloaded asset hash.
Durable outbox records preserve the exact bytes across retries and crashes.

Retries use finite backoff. Verified receipts keep normal minute cycles local;
remote existence checks are jittered, limited to eight Site items and eight
GitHub release tags per cycle, and full artifact checks are scheduled daily.
Network failure remains visible in the report and is retried by a later cycle;
the code does not claim that every remote service will eventually accept data.

## Production evidence for this release

On 2026-09-18, the first authorized production `--once` completed with exit 0
and `errors: []`. It verified 55 recording sources and 38 training sources on
the Site and GitHub, produced seven content-addressed daily release assets, and
recovered both observed hard-crash tails without changing their source
fragments. This is evidence for that bounded run, not a guarantee of future
uptime or provider availability.

## macOS LaunchAgent example

`com.example.jev-nethack-archive-publisher.plist.example` is intentionally
non-ready. Replace every placeholder, verify the one-shot command first, then
copy the edited plist into `~/Library/LaunchAgents/`. LaunchAgents are tied to a
macOS user login session: they do not run while the Mac is shut down, normally
stop at logout, may pause during sleep, and depend on the host and network being
available. `RunAtLoad` and `KeepAlive` restart the sidecar at login or after a
process exit; they do not provide an "always online" or "forever" guarantee.

The persistent command omits `--once`; the process scans at the configured
interval. Keep logs and the private publisher state outside the repository.
Create the parent directories for `StandardOutPath` and `StandardErrorPath`
before loading the LaunchAgent; launchd does not create them.

## Recovery invariants

Training recovery accepts only hash-valid complete NPZ frames from a dead
fragment, rebuilds the canonical JSONL index, and commits a portable completion
marker last. The real hard-crash shard used in validation retained 123 complete
transitions and the same stable public identity while its source stayed
unmodified.

Video recovery accepts only a direct
`broadcast-{timestamp}-{32-hex-stream-id}` fragment with exactly one matching
`process-{timestamp}-pid{pid}` ledger directory. The mapped PID must remain
dead, and the current fragment must have a different PID. It derives the MP4 in
private staging, copies the exact JSONL, and commits the completed manifest
last. Active producers and valid empty tails are deferred; ambiguous mappings,
unsafe paths, corruption, and conflicting outputs fail closed.
