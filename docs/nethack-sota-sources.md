# NetHack RL SOTA evidence (literature review: 2026-09-17)

Puffer-specific refresh on 2026-09-18 rechecked PRs #624 and #690, the current 5.0 config/depth recipe/README, and checkpoint removal. The missing protocol-backed win receipt and current checkpoint links remain unresolved. Other literature notes below retain their original review date.

## Executive finding

I found no corroborating source for the exact claim that PufferAI has produced an ascension. The strongest current Puffer claim I found is a July 2026 pull-request title, **“Nethack head gating, 5-6k score”**. The PR was merged, but its description is empty and it does not publish a median/mean, episode count, seed set, role breakdown, ascension count, or a downloadable checkpoint. Treat “5-6k” as an unstandardized score claim, not as a win or a new NetHack record.

For a directly comparable benchmark, the NeurIPS 2021 challenge defined the ordering as (1) ascensions, (2) median in-game score, (3) mean in-game score. Its final test used 4,096 episodes. The winning agent had median score 5,300 and **no reported ascension**; the paper explicitly says this was far short of ascension. HiHack (2023) is a useful open neural baseline, but it is not the current neural SOTA: its best fully data-driven policy reported mean 1,551 and median 972 on that paper’s standard evaluation, while symbolic AutoAscend reported mean 8,556 and median 4,918 under the same study. Later work reports substantially higher **score-task** numbers: the 2024 retention paper reports 10,588 ± 672 mean score for Human Monk, and SOL+Motif (2025/2026) reports a new best on the NLE NetHackScore task. Those are score-task results, not evidence of ascension or of winning the 2021 full-game challenge tuple.

## Evidence ledger

| Date | Source | What it establishes | What it does not establish |
|---|---|---|---|
| 2026-07-25 | [PufferLib PR #624](https://github.com/PufferAI/PufferLib/pull/624) | Merged PR title: “Nethack head gating, 5-6k score”; one commit (`4405b25`) was merged into the `4.0` branch. | No protocol, N, mean/median, seed list, role mix, max turns, ascension count, or checkpoint is supplied. The GitHub API reports an empty PR body. |
| 2026-07-25 | [Puffer head-gating commit](https://github.com/PufferAI/PufferLib/commit/4405b25d6616d19cec8f11cb2399066180ba7dfa) | Code adds opt-in `PUFFER_HEAD_GATING` for consumed action-head log-prob/entropy/gradient handling. | The code diff contains no evaluation receipt and cannot by itself prove the title’s 5–6k score. |
| 2026-09-13 | [PufferLib PR #690](https://github.com/PufferAI/PufferLib/pull/690) and [merge commit](https://github.com/PufferAI/PufferLib/commit/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2) | Current `5.0` branch received a “nethack challenge” change. | PR body is empty; no result table or ascension evidence is published. |
| current `5.0` | [Puffer challenge config](https://raw.githubusercontent.com/PufferAI/PufferLib/5.0/config/nethack.ini) | `multi_role = 1`; 512 agents, 32 threads; 1,000,000,000 training timesteps; horizon 512; 1024-wide, 4-layer policy; sweep metric `score` over 200M–1B timesteps. | These are training settings and search bounds, not measured benchmark outcomes. |
| current `5.0` | [Puffer depth recipe](https://raw.githubusercontent.com/PufferAI/PufferLib/5.0/ocean/nethack/depth.ini) | The file comments call the recipe a `sweep-0306 winner`, with “tail depth 11.0”, asynchronous rollouts, RNN carry, challenge multi-role, and about 366M steps; Search20 and Run are masked. | “Tail depth 11.0” is a depth-tail metric, not an ascension and not a median/mean score. The file does not define the tail quantile or publish episode-level results. |
| 2026-09-06 | [Puffer cleanup commit](https://github.com/PufferAI/PufferLib/commit/713733549ab85a200a24051464630a60c6794511) | The public history shows the previously tracked `nethack_score_weights.bin` and `nethack_depth_weights.bin` were removed in the “Minor cleanup” commit. | The current branch therefore does not provide those trained weights for direct reproduction. The pre-cleanup [2da3f7b tree](https://github.com/PufferAI/PufferLib/tree/2da3f7bffcd2d579aa3cf29d86586b90a7aea64f/resources/nethack) still records the binaries, but reproducing them requires pinning that older tree. |
| current `5.0` | [Puffer NetHack README](https://github.com/PufferAI/PufferLib/blob/5.0/ocean/nethack/README.md) | Public implementation targets NetHack 3.6.6 via fast-nle, with a 26-verb factored action space, legality masking, decomposed-score reward, CUDA encoder/decoder, and a GPU `puffer eval nethack` command. | The README’s default `resources/nethack/nethack_score_weights.bin` reference is not a current downloadable checkpoint after the cleanup above. |
| 2026-07-13 | [Puffer MinGRU article](https://puffer.ai/blog/mingru/) | PufferNet/MinGRU + highway connections are reported as roughly 2× faster than LSTM and tested to horizon 1024 at 5M steps/s. | This is an architecture/throughput report across PufferLib tasks, not a NetHack score or ascension result. |
| 2026-09-17 check | [PufferLib 4.0 overview](https://puffer.ai/blog/4.0/) and [engineering article](https://puffer.ai/blog/engineering-4.0/) | Puffer 4.0 reports generic RL throughput: 15M steps/s for the standard model and >20M steps/s for smaller models on one RTX 5090; the engineering article says its topline speed uses Breakout and selects settings that solve environments fastest in wall-clock time. | Neither article reports a NetHack score, NetHack ascension, NetHack evaluation budget, or a checkpoint. The 15–20M steps/s claim must not be presented as a NetHack SOTA result. |

**Checkpoint note:** at the current `5.0` tip, the public `resources/nethack` directory contains `msgcode128.bin` but no score/depth weight file. The [removal commit](https://github.com/PufferAI/PufferLib/commit/713733549ab85a200a24051464630a60c6794511) shows both binary files being deleted.

## Standard benchmark baselines

### NeurIPS 2021 challenge (the closest accepted full-game protocol)

The [challenge paper](https://proceedings.mlr.press/v176/hambro22a.html) says the `challenge` task exposed full NetHack 3.6.6, rotated race, role, alignment, and gender, and was frozen at NLE v0.7.3 for evaluation. The competition metrics were lexicographic: ascension count first, then median score, then mean score. Development evaluations used 512 episodes in a two-hour window; final test evaluations used 4,096 episodes over a day.

The final winner achieved a median score of 5,300. The same paper states that this was only just above a Beginner score and “still far short of ascension”; a 1-in-20 Valkyrie episode exceeded 30,000, illustrating why a top episode or max score must not be substituted for median performance. The paper reports an ascension minimum of 12,200 points and an average human ascending-run score of 6.98 million, while warning that score is an imperfect proxy for progress.

Primary implementation references:

* [NLE repository](https://github.com/facebookresearch/nle) — NLE is based on NetHack 3.6.6 and supplies the standard Gym interface.
* [AutoAscend repository](https://github.com/maciej-sypetkowski/autoascend) — first-place 2021 symbolic agent; includes Docker setup and seeded multi-episode simulation with JSON results.

### 2023 HiHack neural benchmark

The [NeurIPS 2023 paper](https://papers.nips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf) is the clearest reproducible neural baseline after the 2021 challenge. Its standard evaluation used 1,024 withheld NLE games; it also ran an in-depth evaluation on the same 3,402 withheld games for every policy. Training used six random seeds and exactly 48 hours on one GPU per policy. The paper distinguishes a rolling score proxy from the large-batch standard evaluation and explicitly says they are not interchangeable.

Table 2 reports the following standard score results (mean ± standard error; median):

* APPO + BC hierarchical LSTM: **1,551 ± 73 mean; 972 median** — best fully data-driven neural policy in that study.
* APPO + BC non-hierarchical LSTM: 1,204 ± 138 mean; 779 median.
* Symbolic AutoAscend: **8,556 ± 187 mean; 4,918 median** under the same study’s evaluation.

The paper’s headline “25% online improvement” refers to median score, not ascension. It reports no learned-policy ascension. The [official HiHack repository](https://github.com/upiterbarg/hihack) provides source, a 1.24GB pretrained-checkpoint archive, an evaluation command, and the approximately 97GB HiHack dataset. This is the most practical public neural baseline to reproduce before claiming SOTA.

### 2024 knowledge-retention fine-tuning (Memento / Wołczyk et al.)

The ICML 2024 Spotlight paper [Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem](https://arxiv.org/abs/2402.02868) is the first later result that materially changes the neural score comparison. It fine-tunes a Scaled-BC model (a 33M-parameter Sample Factory/APPO-style LSTM policy) trained from AutoAscend demonstrations, then protects the pretrained behavior with knowledge-retention losses. The NetHack setup is the **Human Monk score task only**, with NLE’s 120-action interface and reward equal to the change in in-game score; it is not the 2021 all-role challenge.

The paper’s full evaluation uses the last checkpoint of each run and **1,000 trajectories**. Its best method, Scaled-BC + online fine-tuning + Kickstarting (KS), reports **mean score 10,588 ± 672** in Human Monk (Table 5). Table 4 gives the same method’s 1,000-episode aggregate as score 10,588, turns 24,436, steps 38,635, dungeon depth 2.66, and experience level 7.73. The study uses about 8,000 Human Monk games from NLD-AA (over 3B state/action/score transitions, drawn from 100,000 AutoAscend games); the pretrained Scaled-BC checkpoint itself was trained on over 115B transitions. The paper also shows large return variance, including runs near 50,000 and unlucky runs around 1,000.

This is a **mean score result**, with a single role and no published ascension count, median, 4,096-game challenge receipt, or role-balanced result. It is therefore a stronger neural score baseline than HiHack for Human Monk, but it does not establish a full-game win. The authors publish [training/evaluation code](https://github.com/BartekCupial/finetuning-RL-as-CL), including a 500M-step reproduction command and a `fast_eval_nethack` command; the command references an author-local pretrained checkpoint path, so code is public but the exact pretrained artifact is not bundled in the repository.

### 2023–2026 AI-guided and hierarchical neural results

**Motif** ([paper](https://arxiv.org/abs/2310.00166)) uses LLM preferences over pairs of NetHack observations to learn an intrinsic reward, then trains RL agents against that reward. It is relevant to “Jev-guided RL” because it injects language-derived behavioral preference into training, but its claims are still NetHackScore/task-score claims, not ascensions.

**MaestroMotif** ([paper](https://arxiv.org/abs/2412.08542), [official code/checkpoints](https://github.com/mklissa/maestromotif)) extends Motif into five LLM-defined skills (discoverer, descender, ascender, merchant, worshipper), trains a shared skill-conditioned PPO policy, and lets an LLM generate code that recomposes skills at deployment. The evaluation is a separate **early-game, language-conditioned benchmark**: nine seeds (three skill-training repetitions × three policy-generation repetitions), success rates for navigation/composite tasks, and object counts for interaction tasks. Reported composite success rates are Golden Exit **24.80% ± 1.18%**, Level Up & Sell **7.09% ± 0.99%**, and Discovery Hunger **7.91% ± 1.47%**; navigation success is 46% Gnomish Mines, 29% Delphi, and 7.2% Minetown. The repository includes pretrained skill-policy weights and an evaluation recipe with `stats_avg 1000`; its training command uses 5B environment steps. These results show useful LLM-guided hierarchical control, but they do not report full-game ascensions or a challenge-comparable final score.

**Scalable Option Learning (SOL)** ([arXiv v3, revised May 8 2026](https://arxiv.org/abs/2509.00338), [official implementation](https://github.com/mbhenaff/sol)) is the strongest newer large-scale score-task result located here. The paper trains on the NLE `NetHackScore` task for **30 billion frames**, compares means with shaded **two-standard-error bands over five seeds**, and reports that SOL+Motif (SOL’s learned hierarchy plus Motif’s LLM-generated intrinsic reward) significantly improves over Motif and sets a “new state of the art on NetHackScore.” The paper does not give a scalar final score table in the text; the result is shown in Figure 3’s learning curves. Training continued to improve at 30B frames, taking about **14 days on one V100-SXM2-32GB with 48 CPUs per run**. The default experiment is Monk, with additional Ranger and Archaeologist curves showing the same trend but lower scores; the repo config pins `nethack_score_fixed_eat`, `score,health` option rewards, adaptive option length, and the three character settings.

SOL’s metric and protocol matter: `NetHackScore` is incremental game score (with a modified EAT action), and the reported figures are training curves over five seeds. The paper reports no Amulet acquisition, successful ascension, 4,096-game full-challenge receipt, median, or role-balanced challenge tuple. The official repository is reproducible at the code/config level and includes a patched NLE, but its example evaluation script points to an author-local checkpoint directory; a public trained SOL checkpoint was not found. The current repository also includes the 2026 [Hierarchical Behaviour Spaces](https://arxiv.org/abs/2604.24558) code path, but that later method does not supply a full-game ascension result in the sources checked here.

**Practical interpretation:** HiHack is an open historical neural baseline; Memento is the clearest published Human Monk mean-score record; SOL+Motif is the later published NetHackScore SOTA claim; MaestroMotif is a distinct language-conditioned early-game skill benchmark. None should be called the current full-game ascension SOTA without a separately published ascension receipt under the 2021 challenge contract.

**Implication for the two Jev tracks (recommendation, not a claim from these papers):** direct Jev should first be treated as a fixed policy and measured on the frozen challenge tuple, with no credit for training reward or LLM self-reports. Jev-guided RL can borrow the evidence-backed ingredients separately: Memento’s retention/distillation to prevent a pretrained policy from forgetting deep-game behavior, and SOL/Motif/MaestroMotif’s options or language-derived rewards for long-horizon credit assignment. The training loop should log those auxiliary rewards, but the acceptance gate should remain ascension count plus fixed-N final score/depth receipts under a pinned engine and action interface.

## What “beat SOTA” should mean for Jev

Use the 2021 challenge tuple as the acceptance contract, and report every component separately:

1. **Ascension:** count completed games that obtain the Amulet and successfully ascend; report count/N and an episode receipt. A score of 5–6k is not an ascension claim.
2. **Score:** report median and mean final in-game score over a fixed, predeclared N (ideally 4,096 for challenge comparability), not only max score or training reward.
3. **Progress:** report max dungeon level, turns survived, and per-role distributions. These help diagnose a policy that farms early score or succeeds only on easy roles.
4. **Reproducibility:** pin the NetHack/NLE or fast-nle version, action-space variant, reward shaping, masks, role/race/alignment/gender sampler, timeout/max turns, seed list, model/config commit, and checkpoint hash. Puffer’s `5.0` code currently exposes recipes but not the formerly tracked trained weights, so a Jev reproduction would require retraining or obtaining the exact checkpoint from the authors.

The immediate research target should therefore be “reproduce a standard multi-role score/depth baseline with receipts, then test ascension explicitly,” rather than treating Puffer’s PR title or its training reward as a full-game win.

## Unresolved evidence gaps

* I found no public Puffer episode log, 4,096-game evaluation receipt, ascension count, role-by-role table, or checkpoint hash tied to the “5-6k” PR title.
* Puffer’s current `5.0` configs and code are public, but the trained weight files referenced by the README were removed from the branch tip on Sep 6, 2026. Historical pre-cleanup commits still exist, so a checkpoint may be recoverable only by pinning that older tree and its matching engine/build; the current tip has no direct downloadable score/depth checkpoint.
* Puffer’s 5.0 challenge recipe uses fast-nle and a 26-verb factored action interface, whereas the 2021 challenge used the full 113-action keyboard. A raw score comparison across those action spaces is not valid until the evaluation contract is shown to match.
* Memento and SOL+Motif materially exceed older neural score numbers, but no source checked here establishes that either has beaten AutoAscend on the standard full-game tuple, much less achieved a reliable ascension rate. MaestroMotif’s success rates measure early-game language-conditioned tasks and are not a full-game score substitute.
