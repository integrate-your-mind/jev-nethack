# Bounded Jev-guided recurrent RL pilot

`train.py` is a small CPU experiment with three modes:

* `bc` trains a GRU actor-critic on Jev `steps.jsonl` actions.
* `ppo` starts from a fresh policy and updates on native `NetHackScore-v0`
  reward.
* `bc-ppo` runs BC first, writes its checkpoint, then performs one bounded PPO
  rollout/update from that policy.

The policy uses one parameter-free `StateEncoder` for logged states, live NLE
states, and evaluation. It carries a GRU hidden state through each episode and
resets it at every terminal or truncated boundary. Teacher episodes are kept
sequential, with hidden state detached at BPTT windows (`--bptt 32`). PPO
replays the collected rollout in its original order; it stores old log
probabilities and values and uses clipped ratios, GAE, entropy regularization,
and value loss.

The GAE convention is explicit: a true terminal transition has no value
bootstrap; a truncation bootstraps from the next state but stops the eligibility
trace at the boundary. A rollout ending only because `--steps` was reached is
bootstrapped as a continuing state. Evaluation always runs three separate
seeded capped episodes and records `terminated`, `truncated`, `censored_score`,
and `stop_reason` in `eval.json`; the local evaluation cap is reported as
`evaluation_cap` separately from an environment truncation. Each rollout and
evaluation also records an initial-state digest, while `config.json` records
the explicit NLE seed scheme, policy initialization/final hashes, and
teacher/evaluation seed-overlap validation. A score is a native NetHack score
only; it is not evidence of ascension.

## Commands

Use the rebuilt local runtime supplied by the parent task (the old cloud-backed
`work/nle-venv` is not a reliable runtime):

```sh
PYTHONDONTWRITEBYTECODE=1 \
  isolated-venv/bin/python \
  outputs/jev-nethack/test_train.py

PYTHONDONTWRITEBYTECODE=1 \
  isolated-venv/bin/python \
  outputs/jev-nethack/train.py \
  --teacher-data outputs/jev-nethack/results/jev-pilot-v2/steps.jsonl \
  --mode bc-ppo --steps 8192 --seed 1 --eval-steps 128 \
  --output outputs/jev-nethack/results/rl-pilot-bc-ppo-seed1
```

`--mode ppo` does not require teacher data. `--mode bc` writes a BC checkpoint
without native-environment training. PPO consumes `--steps` as a total budget in
bounded `--rollout-steps` (128 by default) recurrent updates and keeps one Adam
optimizer across those updates. The NLE environment and recurrent collector
persist across update boundaries; only true environment terminal/truncated
events reset the episode. At a nonterminal update boundary the collector carries
the hidden state forward, and replay receives that exact chunk-start hidden
state. This is the usual small-policy-staleness approximation for recurrent
PPO and is recorded here explicitly. The output path must be new: the runner
uses `exist_ok=False` to prevent replacing receipts. Outputs include
`config.json` with source/data hashes and `api_calls: 0`, checkpoints,
`learning.jsonl`, and the three-episode `eval.json`.

## Scope and limitations

This is a bounded pilot for checking whether Jev demonstrations provide a
useful initialization signal. It is not a SOTA benchmark, an ascension claim,
or evidence that the policy learned general NetHack strategy. The hashed text
features are intentionally small and lossy, one CPU thread is used, and the
default PPO run performs bounded rollout updates and a small number of replay
epochs. No
reward shaping or API calls are made during training. Teacher action quality and
teaching specificity remain empirical questions for the parent Jev experiment.

The implementation currently uses one environment and one contiguous replay
sequence, so memory and update throughput are bounded but not representative of
large recurrent PPO systems. `--steps` caps collection; evaluation has its own
cap and reports cap truncation explicitly.
