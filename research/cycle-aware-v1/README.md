# Cycle-aware Jev question candidate v1

This is an offline policy-question candidate. It has never controlled the persistent public game. It was evaluated in the
isolated actual-game branch experiment documented below.

The candidate makes two small changes while Jev remains the sole chooser:

1. `augment_state` adds `cycle_context`, a deterministic summary of recent
   positions, two-step reversals, action labels, adjacent visit counts, and
   observed prompt/combat/survival exception signals.
2. The existing instructions gain one cycle paragraph. It asks Jev to prefer a
   plausible cycle-breaking action, while explicitly allowing reversal for
   combat, survival, retreat, prompts, menus, `--More--`, and constrained safe
   play. It tells Jev not to choose blank/wall/stone directions merely to differ.

`prepare_candidate` copies the supplied criteria without filtering, forcing,
blocking, or reweighting anything. Every evaluation request retained the exact
121 action IDs and labels. Both baseline and candidate results retain all 121
raw probabilities.

## Evidence

Sixteen saved states were deliberately selected from the same episode:

- 6 demonstrated stagnant-cycle states
- 4 non-cycling normal-play controls
- 3 menu/prompt controls
- 3 combat/low-HP survival controls

These are selected controls, not a held-out accuracy set. One baseline and one
candidate request were made per state. The first 32-call iteration showed a
real flaw: although it removed all selected reversals, it often chose blank
adjacent cells. That full result is preserved under `evidence/iteration1/`.
The final instruction added the traversability clause and used the remaining 32
calls. The two iterations consumed exactly the authorized approximate 64-call
cap; see `api-call-ledger.json`.

Final observed selection effects:

- Stagnant states: the selected immediate reversal count fell from 5/6 to 2/6;
  reversal probability mass averaged 0.317 baseline versus 0.128 candidate.
- Candidate stagnant choices included two moves toward a lower-visit lichen
  corpse cell, one `SEARCH`, one blank-cell movement, and two remaining
  immediate reversals. Thus the result is improved selection pressure, not
  proof of useful progress.
- Normal controls: choices were unchanged in 4/4 states.
- Menu controls: the same correct contextual handler remained selected in 3/3
  states (potion, food, and `--More--`). These controls were selected and must
  not be described as held-out success.
- Survival controls: choices were unchanged in 2/3 states. At HP 1 after a
  jackal bite, the candidate changed movement to `QUAFF`; no action was
  executed, so survival benefit is not established.

Confidence is recorded as distribution concentration only. It is never treated
as accuracy. The saved-state result does not establish higher score, greater
depth, survival, exploration, or ascension probability.

The paired 64+64 actual NLE branch experiment was subsequently authorized and
completed under `branch-runs/paired-64x2-20260918T150538Z/`. Both arms exactly
replayed seed 103 through step 3029 with zero provider calls during replay and
reached the same raw and compact branch state. The baseline then made 63
two-step reversals across two positions. The candidate reduced that to 20
reversals across four positions, but reached no cell absent from the historical
prefix and settled into a movement plus real `--More--` item-pile prompt cycle.
Neither arm changed score, depth, experience level, HP, or hunger. This is
negative evidence for useful progress, and cycle-aware-v1 was rejected for live
migration. `paired-branch-receipt.json` is the frozen top-level receipt (SHA-256
`2fb273733b74a4b5cec944ea282e1a08b9879a444008deddf4e6563dc73ae11e`);
the comparison SHA-256 is
`18f9f3db96bd56fc86012a314590468b8b87b690a91e161f31416fdf345befcc`.

## Files

- `candidate.py`: pure state/instruction transformation
- `test_candidate.py`: deterministic transformation and exception tests
- `fixtures.jsonl`, `fixture-receipt.json`: immutable compact saved states and
  full criteria
- `evaluate.py`: bounded paired evaluator with sanitized provenance
- `evaluation-results.jsonl`: final complete probability distributions
- `evaluation-summary.json`, `evaluation-receipt.json`: final aggregate and hashes
- `observed-selection-effects.json`: conservative target/action audit
- `evidence/iteration1/`: preserved counterexample-producing first iteration
- `api-call-ledger.json`: exact 64-call total
- `branch_experiment.py`, `test_branch_experiment.py`: exact-prefix paired runner and deterministic checks
- `branch-runs/paired-64x2-20260918T150538Z/`: preserved 128-action negative result with raw observations and judgments
- `paired-branch-receipt.json`: frozen paired-result receipt and hashes

## Verification

```sh
PYTHONPATH=/path/to/jev-nethack/until-win python -m unittest -v \
  test_candidate test_evidence test_branch_experiment
```

Live TypeSafe guidance consulted during design:

- `https://docs.typesafe.ai/concepts/state.md`
- `https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md`
- `https://docs.typesafe.ai/primitives/advanced.md`
- `https://docs.typesafe.ai/primitives/choice.md`
- `https://docs.typesafe.ai/confidence.md`

The complete `branch-runs/` tree, including both raw transition packs and request/response records, is in the [release bundle](https://github.com/integrate-your-mind/jev-nethack/releases/tag/v0.3.0-cycle-aware-v1-negative). This source directory includes the portable code and summarized receipts. `source-public-hashes.tsv` maps the original frozen files to the sanitized full bundle; this repository adds this distribution note.
