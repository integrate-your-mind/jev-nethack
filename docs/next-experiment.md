# Jev-selected next experiment

Decision receipt: `8ce1d461-acc5-42a4-a6af-66d537d91c16`, Jev 1.13.0, staged_signal_gate, probability 0.82, confidence 0.77.

The existing 128-action, three-seed results are censored diagnostic pilots. Direct Jev scores were 0, 0, 3 on seeds 10001–10003; random, BC, PPO, and BC→PPO were all zero. All stayed at depth 1 without ascension. Optimizer weight changes do not establish learned game skill. Those seeds are no longer untouched evaluation seeds.

Before scaling training, compare frozen direct Jev, random, untrained initial, BC-only, PPO-only, and BC→PPO on twelve new seeds at 512 actions each. Match the character, core/display/level seeds, action set, horizon, and initial observation digest. Report raw score as primary, with depth, unique positions, and censored survival separately, including every per-seed row. This remains a pilot, not a SOTA evaluation.

The teacher-adequacy gate is a paired win over both random and untrained controls on at least eight of twelve seeds, using the preregistered lexicographic tuple (maximum depth, raw score, unique positions, survival). Stop immediately on any initial-digest/action-set mismatch or invalid episode; do not replace failed seeds after viewing results. If the gate fails, redesign the teacher/reward before additional PPO.

If the gate passes, collect separate 32 training and four validation teacher seeds at 512 actions. Require BC validation negative log likelihood to improve at least 20% against the untrained checkpoint. Then give PPO-only and BC→PPO identical raw rewards, training seeds, and a maximum 65,536 environment steps. Stop both at 32,768 if all observed training rewards remain zero. Do not extend the budget after looking at results.

Use another untouched twelve-seed set for the final comparison. Pilot success requires BC→PPO paired wins over BC-only and PPO-only on at least nine of twelve seeds, without worse median raw score. Otherwise retain the negative or inconclusive result.

This plan was selected from previously verified pilot receipts while the original planning files were temporarily online-only. Exact seed IDs and frozen checkpoint hashes must be registered before running; no run or successful gate is claimed by this document.
