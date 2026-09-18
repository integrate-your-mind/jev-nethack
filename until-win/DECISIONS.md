# Jev decision receipts

- `da47e6a0-dc87-47b9-adfd-b2e6aa6be0f8` — Jev selected an integrated
  supervisor over wrapping the old bounded broadcast runner (confidence 0.66,
  rubric `decision-2026-09-18.2`, model `jev-1.13.0`).
- `9244cb2c-2fd5-4f07-8cf6-3b1b963eb3a0` — Jev abstained on the exact
  accounting-chunk size. Ordinary bounded reasoning selected 1,000 attempts / 24
  MiB defaults; launch arguments can override them without changing episode
  lifetime.
- `cbcd586a-a12e-4247-bd23-ccf100697ecb` — Jev selected independently framed,
  checksummed compressed NPZ records over individual files, memmaps, or JSON-only
  data (confidence 0.99).
- `75a8c0c8-da22-4e71-966d-d801152adc9f` — Jev selected independent reliability
  review before handoff and explicitly rejected activating a second smoke while
  the existing game was active (confidence 1.0).
- `9c07065b-e069-4c3a-8044-56c6aa2e4dd9` — Jev selected the latest durable
  bridge (`bridge-20260918T054907Z`, episode 0, seed 103) as the activation
  recovery source after its 1,967-action replay verification.
- `f2e41c0e-3a25-4ddb-b918-3021ee3ccb5e` — Jev abstained at low confidence on
  which recovery invariant to audit first; its highest-probability candidate was
  pending-intent recovery. Engineering review covered pending intent, restore
  identity, pack durability, and public recovery telemetry together.
- `8f124211-77f4-48d5-9bb9-f87452264936` — after the first native continuation
  exposed a legacy-prefix/native-suffix migration boundary, Jev selected a
  stopped, auditable repair and migration over either abandoning those actions
  or allocating a fresh game (confidence 1.0).

These are advisory design receipts. Passing tests and artifact inspection are
the implementation evidence; the receipts are not execution proof.
