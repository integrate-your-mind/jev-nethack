# Jev × NetHack

Public source, gameplay recordings, and research for Jev-controlled NetHack.

Watch the [public viewer](https://jev-nethack-live.poppybyte.chatgpt.site). Its live indicator checks the age of the captured game frame; an offline frame is not presented as live play.

The repository contains the local runner, authenticated live-site worker, recurrent behavior-cloning/PPO pilot, tests, selected research notes, and immutable release bundles for completed recordings. Credentials, private host paths, preservation archives, and runtime launch receipts are intentionally excluded.

## Research status

The included pilots are development experiments. They are negative or inconclusive: no ascension or state-of-the-art result is claimed. The benchmark plan describes the controls and evaluation contract required before making a stronger claim. The initial training sample is a small schema-preserving example, not the complete corpus. The deployed training archive API accepts verified complete observation/action transition packs; continuous runtime and publishing integration are still being validated.

## Layout

- `runner/`: local game runner and broadcast client
- `site/`: public read-only viewer and authenticated ingest Worker
- `research/`: bounded recurrent BC/PPO pilot
- `data/training/`: sanitized sample and schema/provenance
- `docs/`: research and operations notes
- `release-bundles/`: completed recording MP4/NDJSON artifacts and per-segment manifests, with a SHA-256 manifest

Run Python tests from `runner/` with the dependencies in `runner/requirements.txt`. Run site tests with `npm test` from `site/`. To build, copy `site/.openai/hosting.example.json` to the ignored `site/.openai/hosting.json`, set your own hosting project and bucket binding, and run `npm run build` from `site/`. Placeholder values are sufficient for a local build check; deployment needs real project configuration.

## Data and privacy

The sample removes wall-clock fields, local paths, stream identifiers, hosted URLs, and credentials while retaining observation/action/reward structure. The old dataless pilot checkpoints and result files remain unrecovered from iCloud and are not regenerated here. See `data/training/README.md`, `release-bundles/README.md`, and `PUBLIC_MANIFEST.json`.

## Viewer and evidence

The viewer displays Jev’s returned probabilities for candidate choices and labels the selected action. These probabilities and model confidence are not empirical action accuracy or a calibrated chance of winning. Current-episode score, depth, health, and game turns have separate charts. Completed-episode final scores and the ascension rate use reported outcomes and an explicit denominator; interrupted games are kept separate.

Site v4 source commit: `6d6af33eefaeac8a0ff8fe9132705c86f6fdc3df`. Deployed and anonymously verified on September 18, 2026. Site tests cover old-frame freshness, archive integrity, multi-frame NPZ/JSONL training upload validation, probabilities, and episode counters.

Completed footage is available in [GitHub Releases](https://github.com/integrate-your-mind/jev-nethack/releases) and the public viewer archive. Release checksums verify copied bytes; they do not guarantee indefinite availability from a hosting provider.
