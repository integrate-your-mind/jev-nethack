# Jev NetHack until-win runtime

This directory contains the restartable runtime prepared for the explicitly
requested experiment: let Jev choose every NetHack action, preserve the data,
start another seeded game after each terminal non-ascension, and stop only after
NLE reports a real ascension. Process failures, host restarts, and clean STOPs
resume the same game by deterministic seed/action replay; they never authorize
a silent fresh seed.

This runtime is separate from the bounded experiment runner. The deployed instance resumed its saved game and passed an actual process-crash/automatic-restart check on September 18, 2026. Downloading this source does not start a game or install a service.

## Exact win condition

The installed NLE 1.3.0 implementation sets `info["is_ascended"]` from the
native NetHack result `how_done() == nethack.ASCENDED`. The supervisor records a
win and creates `WIN.json` only when the same step has both:

```text
terminated == true AND info["is_ascended"] == true
```

`end_status` alone is not accepted because the base NLE status enum groups
normal terminal games as death-like outcomes and does not have a distinct
ascension member. A nonterminal `is_ascended` value is also rejected.

## Lifetimes and recovery

- A NetHack episode has an explicit 1,000,000-action engine limit by default.
  Jev budget rotation never resets that episode.
- Each Jev accounting chunk allows at most 1,000 attempts and 24 MiB of
  reserved request bytes by default. The supervisor records the chunk receipt,
  creates a fresh bounded client, and continues the same game.
- The kernel `flock` on `runtime.lock` prevents concurrent supervisors. PID text
  is informational and is never trusted for ownership or process termination.
- After a process or host interruption, the next supervisor reconstructs the
  active episode with the original NLE RNG tuple and replays every durable
  action. Native transitions must match all raw-array hashes, rewards, terminal
  flags, action indices, and keycodes. Imported legacy transitions must match
  every preserved compact before/after state; replayed raw arrays are labeled
  as regenerated rather than historical captures.
- A durable action intent covers the crash window after a decision and before a
  transition commit. Recovery applies at most one pending intent from the last
  verified state without another provider call. A complete pack frame is the
  commit authority when a crash happens before its index or JSONL event.
- Dependency versions, action mapping, source/config contract, initial raw
  observation, or replay mismatches latch `recovery_blocked`. The supervisor
  exits instead of allocating a fresh game.
- Provider failures use bounded exponential backoff (5 seconds through 300
  seconds by default) before asking Jev about the same observation again.
- A raw-transition or durable-event write failure halts gameplay. Continuing
  after losing training data would violate the requested preservation contract.
- Recording capacity is checked before every new Jev decision and game action.
  A full backlog pauses and retries automatically; see the [storage policy](../docs/recording-storage.md).
  Set `--recording-tombstone-root <publisher-runtime-root>/recording-tombstones`
  to the publisher's proof directory when local recording eviction is enabled.

For a clean stop, create `<data-root>/control/STOP` or send SIGINT/SIGTERM. The
file remains in place so a LoginAgent cannot immediately restart the run. Remove
that exact file before an intentional resume.

## Training artifacts

Every action is first journaled as an `action_intent`, then committed after the
environment step to both `transitions.jsonl` and a raw transition pack. Native
NLE ttyrec files and the existing MP4/terminal recording artifacts are retained
alongside the training data.

Raw shards use schema `jev-nethack-transition-pack/v1`:

```text
transitions-000001.npzpack
transitions-000001.index.jsonl
transitions-000001.complete.json
```

The pack media type is
`application/vnd.jev-nethack.transitions+npz`. It begins with
`JEVNHNPZ1\n`, followed by independently checksummed frames:

```text
uint64_be payload_length | 32-byte SHA-256 | np.savez_compressed payload
```

Each payload contains every available pre-action NLE array as `obs__<key>`,
every post-action array as `next_obs__<key>`, and UTF-8 `metadata_json` with the
seed, step, action, keycode, reward, termination flags, verified-ascension flag,
compact state/next-state, and complete validated Jev decision provenance. The
provenance includes the exact bounded request and response JSON bodies encoded
as base64, their SHA-256 digests, the full contextual action criteria, and the
instructions; HTTP headers and bearer credentials are excluded. The reader
disables pickle. A scanner keeps every complete hash-valid frame and
ignores only an incomplete trailing frame after a crash.

Every closed shard has a durable `*.complete.json` marker containing
`completed`, `transitionCount`/`records`, start/end timestamps, and the exact
pack/index hashes and sizes. The offline verifier checks the framing, NPZ
payloads, index prefix, completion marker, hashes, sizes, and fragment manifest.
It exits nonzero for corruption or incomplete training output.

The writer preflights both the next binary frame and its index row. It rotates
before either artifact would exceed the public 32 MiB upload limit, and rejects
an oversized single transition before writing a pack frame or index row.

Use the offline verifier with a runtime data root:

```sh
"$HOME/Library/Application Support/JevNetHack/venv/bin/python" verify_training.py \
  --data-root "$HOME/Library/Application Support/JevNetHack/recordings/until-win"
```

## Manual run contract

Preparing the verified bridge recovery root is an offline, zero-provider-call
operation and is allowed only once for an unused data root:

```sh
"$HOME/Library/Application Support/JevNetHack/venv/bin/python" until_win.py import-legacy \
  --data-root "$HOME/Library/Application Support/JevNetHack/recordings/until-win" \
  --source "$HOME/Library/Application Support/JevNetHack/recordings/bridge-20260918T054907Z" \
  --episode 0
```

The selected bridge episode is seed 103. Its durable source currently contains
1,967 actions and 3,935 compact observations, including the trailing
before-action state at step 1,967. Import records source hashes and prepares
`recovery_pending`; it does not start NLE or contact Jev.

If a stopped imported run has already added native transitions, validate the
combined legacy prefix and native suffix before resuming:

```sh
"$HOME/Library/Application Support/JevNetHack/venv/bin/python" until_win.py migrate-recovery \
  --data-root "$HOME/Library/Application Support/JevNetHack/recordings/until-win"
```

This command requires the persistent STOP latch and no active episode. It
writes an atomic migration receipt containing the contiguous-history digest,
source hashes, native and regenerated step ranges, and explicit zero-provider/
no-fresh-seed assertions. It validates evidence only and does not start NLE.

Install the dependencies from `requirements.txt`, plus ffmpeg/ffprobe for recordings. Run tests with `python -B -m unittest discover`. The opt-in production recovery test copies a stopped fixture before writing and never calls Jev. The run command is:

```sh
"$HOME/Library/Application Support/JevNetHack/venv/bin/python" until_win.py run \
  --data-root "$HOME/Library/Application Support/JevNetHack/recordings/until-win" \
  --jev-token-file /absolute/private/path/to/typesafe.token \
  --ingest-token-file "$HOME/Library/Application Support/JevNetHack/ingest.token"
```

The token files must stay private and are never copied into recordings,
manifests, source archives, or a public repository. Environment variables are
also supported by the inherited client, but a private file is easier for a
LoginAgent because login jobs do not inherit an interactive shell environment.

`launchd.example.plist` is a template only. Its `KeepAlive` rule restarts a
nonzero crash, while a verified win or clean stop exits successfully. `RunAtLoad`
starts it again after login or a machine restart unless the STOP file remains.

## Material limits

“Until it wins” has no predictable duration or cost, and this policy has not
demonstrated full-game competence. The computer must be powered on for local
play. Recovery re-executes the action history from the original seed and can
take progressively longer; it proceeds only while deterministic checks pass.
Legacy raw arrays did not exist historically, so only the compact legacy states
and saved decision/action evidence are original; raw arrays captured during
replay are explicit reconstructions. Local and hosted storage, account
availability, credentials, and provider service remain external dependencies.
The runtime preserves data without an automatic deletion path, but no finite
system can guarantee literal forever retention.
