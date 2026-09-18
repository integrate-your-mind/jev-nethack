# Jev × NetHack: experiment and benchmark contract

Prepared 2026-09-17. This is an implementation and research plan, not a SOTA claim.

## The two tracks

1. **Direct Jev:** a frozen `jev-1.13.0` chooses an action from the environment's keyboard interface. The adapter supplies only player-visible observations and memory of earlier observations/actions. No simulator snapshots, future observations, hidden dungeon state, or seed values go to Jev. The controller records every action and actual environment outcome.
2. **Jev-guided RL:** a separate local recurrent policy learns from Jev's recorded choices, then improves through PPO on native game reward. Imitation-only and PPO-only controls determine whether the teacher helps. This is foundation-model-assisted RL; Jev's unknown pretraining cost/data cannot be represented as zero-cost, from-scratch RL.

The live Jev strategy call selected teacher → behavior cloning → PPO over the other feasible proposals, a learned preference reward and a hierarchical skill selector. Its recommendation is saved in `strategy-response.json`. Subsequent Jev decisions selected the bounded experiment and completion of its controls. Those choices are advisory hypotheses, not empirical proof that the method is best.

[TypeSafe's current model documentation](https://docs.typesafe.ai/models) says customer fine-tuning and LoRA are unavailable. Training changes our student policy; it does not change Jev. For a competitive implementation, use Puffer's trainer/fast environment as infrastructure and compare its unchanged baseline against the same trainer with Jev guidance.

## What the initial implementation tests

The direct pilot uses NLE 1.3.0 / NetHack 3.6.7, 121 keyboard actions, full menus, a Human Monk, deterministic core/display/level-generator seeds, and an external 128-action cap. PPO uses the same environment constructor and observations. Arithmetic and event accounting happen in code. The initial student has small hashed text features and a GRU; it is a prototype for testing the learning loop, not a competitive NetHack architecture.

Teacher/development games: seeds 101–103. Evaluation games: 10001–10003, separated from demonstrations and PPO training. All five policies receive the same evaluation cap: uniform random, direct Jev, imitation-only, PPO-only, and imitation-plus-PPO. Both PPO arms get 8,192 training interactions with matched initialization, configuration, and training seed. Every result must disclose artificial stops, actual game termination, action count, in-game turn count, score, progress, and ascension status. Three short censored episodes cannot establish an ascension rate or a general performance advantage.

This pilot is not the historical challenge protocol: its NLE/game version, fixed role and action budget differ. The [maintained NLE README](https://github.com/NetHack-LE/nle) identifies its current game as 3.6.7. Historical and Puffer comparisons require a separately pinned 3.6.6 benchmark environment.

## Baselines to reproduce

See [the source ledger](../nethack-sota-sources.md) for exact scenarios, citations and missing evidence.

| Track | Reference to reproduce | Comparison boundary |
|---|---|---|
| Multi-role full game | [2021 challenge / AutoAscend](https://proceedings.mlr.press/v176/hambro22a.html) | Ascensions, then median score, then mean score; the final challenge used 4,096 games. Symbolic agent, separate from pure RL. |
| Human Monk neural score | [Wołczyk et al., ICML 2024](https://arxiv.org/abs/2402.02868) | Reports over 10K in its Human Monk setting; retain its actions, demonstrations and evaluation settings. |
| Scaled online hierarchical RL | [SOL, revised May 2026](https://arxiv.org/abs/2509.00338) | 30B training frames; preserve intrinsic reward and training budget. A score-learning curve is not a full-game ascension result. |
| AI-guided skills | [MaestroMotif](https://arxiv.org/abs/2412.08542) | Compare its explicit skill/task success metrics as well as native score. |
| Puffer score/depth | [Puffer NetHack source](https://github.com/PufferAI/PufferLib/tree/5.0/ocean/nethack) | Pin fast-nle, factored actions, masks, reward components, checkpoint and evaluation script; a PR title or training recipe alone is not a verified leaderboard result. |

The primary Puffer sources located so far do not supply a protocol-backed recent ascension/win receipt. The July score claim and September challenge code are useful reproduction leads. Do not assume “Puffer recently won” defines one universal threshold; identify the exact competition/result before declaring it beaten.

## Meaning of “across the board”

Freeze a separate comparison contract for each task/interface rather than merging incompatible score tables. For every contract, report:

- Ascension count and rate with binomial uncertainty, using actual terminal events and replay evidence.
- Median, mean and lower-tail native score, with confidence intervals and per-episode receipts.
- Official scout/progress metrics, dungeon/branch milestones, turns survived and deaths. Our pilot's observed-nonblank-tile counter is only a diagnostic, not the canonical scout metric.
- All 13 roles, eligible race/alignment combinations and the same role sampler; include per-role and worst-role results.
- Training frames, training seeds, GPU hours, environment throughput, inference latency, API requests/tokens/cost and peak memory.
- Game/NLE/fast-nle versions, action interface and masks, reward shaping, timeouts, known text/pretraining/demonstration access, model/checkpoint hashes and test seeds.

Use the same untouched test games for matched candidate/baseline evaluation where the protocol permits seeds. Keep tuning games separate. Preserve the official challenge's prohibition on controlled test seeds when using that protocol. Bootstrap paired game differences where appropriate; use multiple independent training seeds, correct for the multiple primary comparisons, and freeze sample sizes/stopping rules before the final test. Increase evaluation from development pilots to a meaningful intermediate batch and then the official-size run. A broad win requires improvement in every predeclared primary comparison with uncertainty accounted for; Pareto tradeoffs must be reported as such.

## Scaling path

1. Verify the two working loops and measure the teacher's actual failure modes. Keep failure traces.
2. Improve direct control using compact semantic observations, persistent spatial memory and explicit handling of prompts, menus and inventory. Compare each change on development games. Then collect better demonstrations, including corrective labels on the student's visited states.
3. Replace the lossy pilot encoder with the competitive baseline's spatial/message/inventory encoder and recurrent memory. Preserve the same observation access and expose any expert rules or action macros as separate ablations.
4. Train baseline and Jev-assisted policy with matched environment budgets and several training seeds. Measure the added value of Jev against imitation-only and equal-cost alternative teachers. Use distillation to keep API calls out of the high-throughput training/inference loop.
5. Reproduce Puffer's exact checkpoint and evaluation, or retrain its pinned recipe when the matching public checkpoint is unavailable. Run the multi-role/native-score/progress contracts independently before making a combined claim.

The measured first pilot used 2,301,355 input tokens for 384 actions, approximately $0.0967 at the current documented $0.042/M input-token rate. That is about 5,993 input tokens and 0.50 seconds per API decision in this setup. A linear extrapolation is roughly $25 per 100,000 direct decisions, excluding failed calls and future changes. This is why a local student and infrequent teacher calls matter for large evaluations. Actual provider billing remains separate from this token-based estimate.

Large GPU training and a publication-quality benchmark require a concrete compute allocation; neither has been launched by this pilot. The next allocation should be based on measured learning curves and an exact reproducible baseline, not Jev's confidence alone.
