# Native terminal metrics and historical correction

> **Rollout status:** deployed September 18, 2026. Historical correction and native-score display were verified, and an exact 5,280-action same-game replay completed. The public API and browser show the restored observation and provider backoff; no new action is invented. New gameplay is unavailable while the provider returns HTTP 402; actual disk write failures have also interrupted recovery. The viewer labels stale observations offline. See [production verification](../research/runtime-metrics-v1/production-verification.json) for the bounded result.

JevNetHack records several score-like values because they answer different questions. Keeping them separate prevents a last observation, an episode maximum, or accumulated environment reward from being presented as NetHack's official terminal score.

## Metric meanings

| Field | Meaning | Source |
| --- | --- | --- |
| `officialFinalScore` | NetHack's official score for a completed terminal game | The terminal game's native xlog record, after provenance checks |
| `lastObservedScore` | The last valid observed score retained before terminal cleanup | The environment observation stream |
| `maxObservedScore` | The greatest score observed at any step | The environment observation stream |
| `totalReward` | The sum of environment rewards across the episode | Step rewards, including deterministic replay reconstruction |
| `score` | Compatibility field for older consumers | Official final score when available; otherwise the last observed score |
| `scoreSemantics` | Declares what the compatibility `score` means | `official_final` or `last_observed` |

These values can coincide, but coincidence does not make them interchangeable. For example, an environment may return a zeroed observation during terminal cleanup. That buffer must not replace the last valid observed score. Likewise, `totalReward` is an environment quantity and is never relabeled as an official score.

An official score of zero is valid. The implementation distinguishes zero from missing evidence with an explicit null check.

## Native terminal provenance

The official result is read only after the environment has closed. A native result is accepted only when all of these conditions hold:

- The episode directory identity matches the expected episode ID and seed.
- Exactly one bounded, regular native xlog file exists.
- The xlog contains exactly one well-formed record.
- Its filename contains a valid NLE process ID.
- Its `ttyrecname` names a retained regular ttyrec in the same episode directory.
- The xlog filename and ttyrec filename contain the same NLE process ID.
- Symlinks, unexpected path shapes, oversized files, duplicate fields, malformed integer fields, and changing files are rejected.

Accepted provenance records the source type, episode ID, seed, relative xlog and ttyrec paths, xlog hash and size, ttyrec size, and the fact that one native xlog record was used. Public telemetry applies a typed allowlist and rechecks that nested provenance matches the containing episode and seed.

If evidence is missing, malformed, ambiguous, unsafe, or cannot be bound to the episode, `officialFinalScore` remains null and `officialScoreError` records a stable reason. The runtime does not guess an official result from observation or reward fields.

## Replay and exact same-game resume

Recovery replays the durable action prefix without provider calls. During replay, nonterminal observations reconstruct `lastObservedScore` and `maxObservedScore`, while retained rewards reconstruct `totalReward`. Native terminal evidence is consulted only for a terminal episode.

A rollout must preserve the existing recovery contract:

1. Keep the episode ID, seed, action ordering, observation digests, and replay source identities unchanged.
2. Keep retained transitions, packs, indexes, markers, ttyrecs, and xlogs unchanged.
3. Never replace uncertain recovery evidence with a fresh seed.
4. After the correction and source deployment are read back successfully, start through the normal runner recovery path.
5. Verify that the resumed runtime selects the same episode and seed, validates the entire retained prefix with zero provider calls, and continues only after the exact durable tip.
6. Treat a replay mismatch, source mismatch, dependency mismatch, or uncertain tip as a visible blocked recovery. Do not continue from a guessed state.

The historical correction changes a completed episode's metrics row and writes an audit receipt. It does not edit gameplay journals or training artifacts.

## Correcting a retained historical result

The correction tool is read-only unless `--apply` is supplied. A dry run validates the complete evidence chain before proposing a change:

- the expected runtime schema;
- one unique completed terminal row, even when the same episode has an earlier interrupted row;
- contiguous observation-digest and score chains;
- the terminal transition;
- the immediately following episode summary;
- the bound native xlog and ttyrec;
- hashes of every source artifact used for the metric decision.

Training packs are deliberately outside this metric-input set and are not parsed by the correction. Tests use pack, index, and completion-marker sentinels to prove that applying a correction does not change them.

### Required stopped state

Define a portable installation root rather than copying a host-specific path:

```sh
export JEV_NETHACK_ROOT="/path/to/JevNetHack"
export DATA_ROOT="$JEV_NETHACK_ROOT/recordings/until-win"
export PYTHON="$JEV_NETHACK_ROOT/venv/bin/python"
export CORRECTION="$JEV_NETHACK_ROOT/until-win/historical_metrics_correction.py"
export GLOBAL_LOCK="$JEV_NETHACK_ROOT/global-gameplay.lock"
```

Before generating apply evidence:

1. Create the regular `STOP` latch used by the runner.
2. Wait for the runner process to exit.
3. Read state and require `status=paused`, `activeEpisode=null`, and `activeFragment=null`.
4. Require the handoff-bound global gameplay lock and `$DATA_ROOT/runtime.lock`.
5. Ensure the intended receipt parent already exists as a real, non-symlink directory.
6. Do not reuse a dry run generated while gameplay was active.

The global lock serializes gameplay ownership across the installation. The runtime lock serializes the selected data root. The correction requires both. This guarantee assumes all legitimate writers honor both locks.

### Fresh dry run and apply

Generate a fresh dry run after the stopped-state checks, retain it atomically, and calculate its SHA-256:

```sh
STATE="$DATA_ROOT/state.json"
REVIEW="$JEV_NETHACK_ROOT/recovery/episode-0-metrics-dry-run.json"
RECEIPT="$JEV_NETHACK_ROOT/recovery/episode-0-metrics-correction.json"

review_tmp="$REVIEW.tmp.$$"
umask 077
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$CORRECTION" \
  --data-root "$DATA_ROOT" \
  --state "$STATE" \
  --episode-id 0 > "$review_tmp"

"$PYTHON" - "$review_tmp" "$REVIEW" <<'PY'
import os
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
with source.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(source, target)
directory = os.open(target.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

REVIEW_SHA="$(shasum -a 256 "$REVIEW" | awk '{print $1}')"
```

Review that exact file. Apply only with its exact path and digest:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$CORRECTION" \
  --data-root "$DATA_ROOT" \
  --state "$STATE" \
  --episode-id 0 \
  --apply \
  --receipt "$RECEIPT" \
  --global-lock "$GLOBAL_LOCK" \
  --expected-dry-run "$REVIEW" \
  --expected-dry-run-sha256 "$REVIEW_SHA"
```

The apply path uses the reviewed state hash as a compare-and-swap boundary. Under both locks it rebuilds and compares the correction, checks source artifacts before and after the state write, fsyncs a prepared receipt before atomically replacing state, reads the result back, and completes the receipt. The only intended writes are `state.json` and the correction receipt.

A retry may finish a prepared transaction after a state-write or receipt-write failure, including after a launcher reboot refreshes a semantically identical handoff receipt. It must not widen the target, accept a changed reviewed state, or reuse stale review evidence.

## Deployment acceptance checks

The candidate review authorizes the bounded rollout procedure; it does not establish that production now runs these files. Deployment is complete only after the operator records all of the following:

- Exact predeployment source hashes and absence assertions match the reviewed manifest.
- A rollback copy exists.
- The runner is stopped and the paused, inactive state and both locks are verified.
- Only the reviewed metrics source files are installed, and their deployed hashes match the accepted candidate.
- A fresh post-STOP dry run is retained and reviewed.
- The one authorized historical row is corrected; earlier interrupted rows and all nontarget episode results remain unchanged.
- The completed receipt, corrected row, native source hashes, and raw-pack sentinels pass readback.
- The runner resumes the exact same game through verified replay without provider calls during the prefix.
- Runtime and public telemetry expose the separated fields and bound native provenance.
- No new privacy, overflow, or publication error appears.

Until those checks pass on the deployed hashes, describe the feature as accepted and locally verified, with production verification pending.

## Verification evidence

The frozen candidate suite completed **95 tests with 1 opt-in skip**. The skipped test is the copied-production, two-restart recovery fixture; it was not run in the final suite and must not be reported as passing. Independent focused review also exercised the correction and receipt-parent paths.

The passing suite covers:

- terminal scalar capture before a zeroed close observation;
- native xlog and ttyrec binding;
- explicit unknown results for missing, malformed, or ambiguous evidence;
- zero-valued official scores;
- replay-derived metrics with zero provider calls;
- schema, digest-chain, and score-chain continuity;
- STOP, paused-state, global-lock, and runtime-lock gates;
- stale dry-run and state-drift rejection;
- source time-of-check/time-of-use detection;
- prepared/completed receipt validation, idempotence, and reboot-safe completion;
- unchanged nontarget state and raw training-pack sentinels;
- typed public telemetry, privacy filtering, and oversized-number omission;
- existing recovery, transition-pack, broadcast, and local NLE behavior.

The accepted historical dry run found one completed terminal record with 5,227 contiguous transitions and derived official, last-observed, and maximum score 138, reward 138.0, status `DEATH`, death cause `died of starvation`, and death context `fainted`. That run was read-only and its state snapshot later became stale while gameplay continued. It is evidence that the reconstruction worked; it is not reusable authorization for an apply. A new dry run is mandatory after STOP.

## Clean shutdown fragment state

The runtime clears activeFragment only after the training writer has durably closed its shard marker and manifest. A failed close retains the fragment pointer and recovery state. The older runtime left a stale pointer after a successful stop; a separately reviewed, dual-lock compare-and-swap normalization cleared only that pointer while preserving the exact resume candidate and all result rows.

## Provider waits

A current observation is published before the first provider decision. If the provider rejects that request, the recorder republishes status and metrics with the same capture timestamp, null decision and null action. The viewer therefore ages the observation normally; a heartbeat cannot masquerade as new gameplay. HTTP errors expose only a sanitized code such as JevHTTPError:402. The service retains the game and retries with backoff.
