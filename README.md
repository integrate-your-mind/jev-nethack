# Jev × NetHack

Public source, gameplay recordings, and research for Jev-controlled NetHack.

Watch the [public viewer](https://jev-nethack-live.poppybyte.chatgpt.site). Its live indicator checks the age of the captured game frame; an offline frame is not presented as live play.

The repository contains the local runner, authenticated live-site worker, recurrent behavior-cloning/PPO pilot, tests, selected research notes, and immutable release bundles for completed recordings. Credentials, private host paths, preservation archives, and runtime launch receipts are intentionally excluded.

## Research status

The included pilots are development experiments. They are negative or inconclusive: no ascension or state-of-the-art result is claimed. The benchmark plan describes the controls and evaluation contract required before making a stronger claim. The initial training sample is a small schema-preserving example, not the complete corpus. The deployed archive publisher now uploads complete observation/action transition packs and video segments to the public site and daily GitHub Releases. Its first verified pass covered 38 training shards and 55 video segments; the background service then reused those receipts and published 11 newly completed items. Active, unfinished tails remain local until closed or recovered after a crash.

The [cycle-aware-v1 experiment](research/cycle-aware-v1/README.md) reduced backtracking in a paired 64-action comparison but produced no new cells, score gain, or depth gain. It was rejected for the persistent live player. Both saved-state iterations and the complete raw paired branch data are preserved in the [negative-result release](https://github.com/integrate-your-mind/jev-nethack/releases/tag/v0.3.0-cycle-aware-v1-negative).

The [frontier-v2 validation-fix comparison](research/frontier-v2-validation-fix/README.md) found 14 more new cells across two saved positions, but its preregistered result is **negative** because safety gates failed. Detailed evidence shows a combat-priority lapse and a menu-evaluator false positive: the menu action actually opened the door correctly. All 143 provider responses and 128 raw transitions are preserved, with no live policy migration. The original zero-call infrastructure failure is preserved separately.

## Layout

- `until-win/`: persistent direct Jev policy with verified same-game recovery and raw transition capture
- `runtime-support/`: automatic verified archive publishing and recovery of interrupted video/training tails
- `runner/`: bounded experiment runner and broadcast client
- `site/`: public read-only viewer and authenticated ingest Worker
- `research/`: bounded recurrent BC/PPO pilot and frozen direct-policy experiments
- `data/training/`: sanitized sample and schema/provenance
- `docs/`: research and operations notes
- `release-bundles/`: completed recording MP4/NDJSON artifacts and per-segment manifests, with a SHA-256 manifest

Run Python tests from `runner/` with the dependencies in `runner/requirements.txt`. Run site tests with `npm test` from `site/`. To build, copy `site/.openai/hosting.example.json` to the ignored `site/.openai/hosting.json`, set your own hosting project and bucket binding, and run `npm run build` from `site/`. Placeholder values are sufficient for a local build check; deployment needs real project configuration.

## Data and privacy

The sample removes wall-clock fields, local paths, stream identifiers, hosted URLs, and credentials while retaining observation/action/reward structure. The old dataless pilot checkpoints and result files remain unrecovered from iCloud and are not regenerated here. See `data/training/README.md`, `release-bundles/README.md`, and `PUBLIC_MANIFEST.json`.

## Viewer and evidence

The viewer displays Jev’s returned probabilities for candidate choices and labels the selected action. These probabilities and model confidence are not empirical action accuracy or a calibrated chance of winning. A recent-movement panel measures distinct squares, immediate returns, game turns, score change, and depth change from the last recorded action window. It makes repeated movement visible without calling it success. Current-episode score, depth, health, and game turns have separate charts. The completed-episode score chart accepts only explicit native final-score fields; zeroed terminal observations cannot masquerade as a final score. Last observed score and accumulated reward remain separate. Ascension totals and explicitly reported deaths have visible denominators, and interrupted games are kept separate.

Site v7 source commit: `e3d59aacabfb59d52fe47a21b0c8b0145a1782ac`. Deployed and verified through anonymous HTTP and a live browser on September 18, 2026. Site tests cover old-frame freshness, archive integrity, multi-frame NPZ/JSONL training upload validation, probabilities, and episode counters.

The archive loads nine recordings per page, with an explicit Load more control and no fixed history cap. Refresh and pagination preserve playback of existing video cards. Pages follow archive order rather than a global newest-first ranking.

Completed footage is available in [GitHub Releases](https://github.com/integrate-your-mind/jev-nethack/releases) and the public viewer archive. Release checksums verify copied bytes; they do not guarantee indefinite availability from a hosting provider.

## Continuous run verification

The persistent direct Jev runtime resumed episode 0, seed 103, from move 1,980. A controlled process crash at move 2,055 caused the installed background service to restart automatically, verify 2,056 saved actions (including the pending action) with zero provider calls during replay, and continue the same game past move 2,080. This validates that crash path, not arbitrary hardware failures. A Mac restart runs the service after login; local play requires the host to remain powered on. That recovered game subsequently ended in death after 5,227 actions, and the same process automatically began episode 1, seed 104. Native ascension is still unachieved.

See `until-win/README.md` for the save/replay contract, STOP controls, training format, and compatibility checks; [archive operations](runtime-support/README.md) documents publication, readback verification, retries, and the portable background-service example; `docs/typesafe-skill.md` describes the TypeSafe workflow.
