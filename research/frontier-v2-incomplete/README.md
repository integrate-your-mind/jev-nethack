# Frontier-v2 incomplete release package

This package preserves the original failed `frontier-v2-paired-20260918T154833Z` attempt and its frozen candidate/harness/plan/evidence. It is an incomplete negative record, not a successful experiment release.

The attempt made **0 provider calls** and produced **0 future actions**. It stopped before gameplay because frozen action-map validation compared in-memory tuple pairs with JSON-frozen list pairs. Canonical JSON and SHA-256 matched, but direct Python equality failed (`ExperimentError`). Therefore no gameplay benefit, policy comparison, or learning conclusion is supported.

Binary replay and raw training artifacts are byte-preserved. UTF-8 metadata copied into this package has only private absolute data-root text paths replaced with `JEV_NETHACK_DATA_ROOT`; `manifest/path-redactions.json` records that mapping.

The successor validation-fix candidate is outside this package. The full evidence bundle is attached to release v0.4.0-frontier-v2-incomplete. This public source directory contains the release description and complete bundle member hashes.
