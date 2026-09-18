# Jev × NetHack

Public code and research snapshot for a bounded Jev-controlled NetHack pilot.

The repository contains the local runner, authenticated live-site worker, recurrent behavior-cloning/PPO pilot, tests, selected research notes, and immutable release bundles for completed recordings. Credentials, private host paths, preservation archives, and runtime launch receipts are intentionally excluded.

## Research status

The included pilots are development experiments. They are negative or inconclusive: no ascension or state-of-the-art result is claimed. The benchmark plan describes the controls and evaluation contract required before making a stronger claim. Training data is a small schema-preserving sample derived from local recorded observations; it is provided for format and provenance demonstration, not as a complete training corpus.

## Layout

- `runner/`: local game runner and broadcast client
- `site/`: public read-only viewer and authenticated ingest Worker
- `research/`: bounded recurrent BC/PPO pilot
- `data/training/`: sanitized sample and schema/provenance
- `docs/`: research and operations notes
- `release-bundles/`: completed recording MP4/NDJSON artifacts and per-segment manifests, with a SHA-256 manifest

Run Python tests from `runner/` with the dependencies in `runner/requirements.txt`. Run site tests with `npm test` from `site/`.

## Data and privacy

The sample removes wall-clock fields, local paths, stream identifiers, hosted URLs, and credentials while retaining observation/action/reward structure. The old dataless pilot checkpoints and result files remain unrecovered from iCloud and are not regenerated here. See `data/training/README.md`, `release-bundles/README.md`, and `PUBLIC_MANIFEST.json`.
