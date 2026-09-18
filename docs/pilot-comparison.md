# Jev NetHack pilot comparison

Checked 2026-09-18 from the completed receipts. This is a capped feasibility comparison, not a SOTA evaluation. The five evaluation arms are direct Jev, random control, BC-only, PPO-only, and BC→PPO. Development `jev-pilot-v2` is reported separately because its episodes supplied teacher data.

## Common evaluation contract

All five arms used NetHack 3.6.7 through NLE 1.3.0, Gymnasium 1.2.0, the Human Monk role `mon-hum-neu-mal`, NLE’s 121 keyboard/menu actions, seeds **10001, 10002, 10003**, and a **128-action cap**. The direct and random runs stopped with `stop_reason=action_cap`; the RL runs stopped with `stop_reason=evaluation_cap`. None completed an episode. Every arm stayed at dungeon depth 1 and produced zero ascensions.

The score below is the native NetHackScore observed at the artificial cap. It is censored: it is not a completed-episode return. Direct Jev and random receipts expose `RUNNING` at the cap; RL eval receipts expose `truncated=true`, `terminated=false`, and `censored_score=true`.

| Arm | Seed 10001 | Seed 10002 | Seed 10003 | Mean observed score | Median | Steps each | Max depth | Ascensions |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Direct Jev | 0 | 0 | **3** | 1.0 | 0 | 128 | 1, 1, 1 | 0 |
| Random | 0 | 0 | 0 | 0.0 | 0 | 128 | 1, 1, 1 | 0 |
| BC-only | 0 | 0 | 0 | 0.0 | 0 | 128 | 1, 1, 1 | 0 |
| PPO-only | 0 | 0 | 0 | 0.0 | 0 | 128 | 1, 1, 1 | 0 |
| BC→PPO | 0 | 0 | 0 | 0.0 | 0 | 128 | 1, 1, 1 | 0 |

Receipts: [direct Jev summary](outputs/results/jev-heldout/summary.json), [random summary](outputs/results/random-heldout/summary.json), [BC eval](outputs/results/bc-pilot/eval.json), [PPO eval](outputs/results/ppo-pilot/eval.json), and [BC→PPO eval](outputs/results/bc-ppo-pilot/eval.json). The machine-readable comparison is [pilot-comparison.json](outputs/pilot-comparison.json).

## Training receipts

BC used the development Jev data from seeds 101–103: 384 teacher records and three BC epochs. Its action loss fell from 4.6687 with 20.3% accuracy at epoch 1 to 3.6802 with 29.7% accuracy at epoch 3. That is teacher-record fit, not evidence of game learning; its capped evaluation score was zero.

PPO-only and BC→PPO each ran **64 PPO updates × 128 rollout steps = 8,192 environment interactions**, with the native NetHackScore reward only. Both logged mean native reward 0.0 and mean native score 0.0 per rollout. The PPO-only collector recorded three training terminations and BC→PPO four, but neither arm obtained native reward or evaluation progress. BC→PPO performed the three BC epochs before its PPO budget.

The initial and final policy hashes differ for all three trained arms, so weights changed. The zero-reward receipts do not show that the changed weights learned NetHack strategy.

Training artifacts: [BC config/learning](outputs/results/bc-pilot), [PPO config/learning](outputs/results/ppo-pilot), and [BC→PPO config/learning](outputs/results/bc-ppo-pilot).

## Initial-state digest verification

The canonical digest is exactly:

```python
sha256(json.dumps(state, sort_keys=True, ensure_ascii=True,
                  separators=(",", ":")).encode()).hexdigest()
```

I recomputed each evaluation seed by making a fresh NLE reset with the recorded seed scheme (`core=seed`, `disp=seed+100000`, `lgen=seed+200000`, `reseed=False`, moon phase fixed). The result matches every RL `eval.json` and the first state in both direct and random `steps.jsonl`:

| Evaluation seed | Canonical initial-state digest |
|---:|---|
| 10001 | `bc729abdf018fb8a0874f0ec35a5b1c38c4e8378f15a083b11624890bef7f67e` |
| 10002 | `63224aa91ab13642d4d3141c635036e41b92cb837057b0643ebe0854190c5dd4` |
| 10003 | `ba7ef55e79853ef224b709ed560b92ed377310bc585377737737575954aaa2fb` |

This verifies the matched initial states and serialization contract. It does not turn the capped runs into completed episodes.

## Development teacher run

`jev-pilot-v2` used seeds 101–103 and the same 128-action cap. Its observed scores were **[4, 4, 18]**, with native rewards **[4.0, 4.0, 18.0]**, depth 1 throughout, and zero ascensions. These are development teacher traces used by BC and BC→PPO; they are not held-out evaluation results or completed-game scores. See the [development summary](outputs/results/jev-pilot-v2/summary.json) and [teacher steps](outputs/results/jev-pilot-v2/steps.jsonl).

## Limitations and interpretation

- The 128-action cap censors all five evaluations. Mean and median here summarize observed pre-cap score, not completed-episode score.
- The experiment has one fixed role, one game version, three evaluation seeds, and 121 actions. It is not the 2021 multi-role 4,096-game challenge and cannot establish SOTA, an ascension rate, or a general performance advantage.
- Direct Jev’s score 3 is a tiny observed difference against these controls, not a win. All arms remained at depth 1.
- BC’s lower action loss and all changed policy hashes establish optimization activity only. PPO received zero native reward; no arm shows evidence of learned general NetHack progress.
- The pilot’s observed nonblank tiles and capped score are diagnostics. They are not the official scout/depth/ascension metrics used by historical benchmarks.

## Next Jev decisions and architecture broadcast

The completed [menu-label decision](outputs/menu-decision.json) selected `relabel_full_actions` with confidence 1.0. The trigger was a prompt asking for `read(r)` while the full action labels described `r` as normal movement, causing repeated ineffective actions. This is a next-variant decision only; no new variant result is included here.

The separate [architecture decision](outputs/live-architecture-decision.json) addressed the new request to publish a public live Jev/NetHack site with complete recordings. It selected `hosted_ingest` at confidence 0.99. That is an architecture choice, not evidence of a public deployment, live stream, or verified permanent backup.
