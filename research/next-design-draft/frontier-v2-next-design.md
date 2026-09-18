# Frontier-v2 next-candidate design

Status: offline design only. No candidate code, provider calls, replay, trial retry, RL, or live-policy change is authorized by this note.

## Evidence boundary

The completed successor trial is development data. It reached 14 more prefix-new cells than its paired baseline, but its frozen classification is negative because the priority gate failed at branch 3359 and the menu-label gate failed at branch 3999. These events, the synthetic fixtures, both frozen branch points, and all nearby states are excluded from any later held-out claim.

At branch 3359, step 3382 (`frontier-v2-3359-frontier-v2-3382`, branch action 22, action index 2, key `j`, global attempt 61), the observation-derived signals had `recentCombat=true` and no prompt, low-HP, or urgent-hunger signal. The goal judgment selected `follow_observed_frontier_route` with probability 0.36 and confidence 0.25; `handle_immediate_combat` had probability 0.04. Jev then chose south with probability 0.72 and confidence 0.71. The next observation said “The jackal bites!” and HP changed from 13 to 12. This is a temporal observation, not proof that the goal choice caused the damage or that another goal would have prevented it. In the current interface, `recentCombat` is only a regex match against message text. Here that text—“You kill the jackal! The jackal misses!”—described events that had already happened. The boolean did not establish that a threat remained active.

At branch 3999, step 4017 (`frontier-v2-3999-frontier-v2-4017`, branch action 17, action index 5, key `n`, global attempt 127), the message was “In what direction?” and `visiblePrompt=true`. The goal correctly selected `resolve_visible_prompt` with probability and confidence 0.98. Jev chose southeast with probability 0.81 and confidence 0.80. The next message was “The door opens.” Position, turn, HP, and score did not change. The frozen evaluator nevertheless failed the menu gate because the contextual label said ordinary southeast movement rather than containing a prompt-handler phrase. This is a confirmed evaluator false positive; it is not evidence of a gameplay failure.

Probability and confidence values above describe the retained distributions. They are not accuracy, skill, safety, or improvement estimates.

## Singleton counterevidence and Jev decision

The earlier `scoped_goal_criteria(state)` proposal would have supplied exactly one goal whenever a priority boolean was active. That would make priority-goal compliance true by code construction. Jev would still choose every keyboard action, but it would no longer choose among goals at those states. A concentrated one-option distribution or its confidence could not evidence goal-selection skill. Worse, the historical `recentCombat` boolean could force `handle_immediate_combat` after the relevant threat had disappeared.

Jev was asked to compare that explicit restriction with retaining competing goals while supplying clearer, fresher priority state and evaluating independent gameplay outcomes. It selected the competing-goals variant (`f5d2c5ea-7ac5-4f03-aa16-2650f932d4b7`, probability 1.0). This is an advisory design choice, not proof that the candidate works and not authorization to implement or run it.

The design follows the TypeSafe state and Choice guidance: use named state fields, keep the complete set of genuine alternatives, preserve the returned distribution, and treat confidence as concentration of that distribution rather than correctness. Sources consulted: <https://docs.typesafe.ai/concepts/state.md>, <https://docs.typesafe.ai/primitives/choice.md>, and <https://docs.typesafe.ai/confidence.md>.

## One minimal candidate change

Replace the five unqualified `priority_signals` booleans supplied to the existing goal question with one structured, observation-only `priority_context`; update only the corresponding goal instructions and goal-refresh comparison. Keep all seven existing goal criteria available on every goal request.

The context records, for each signal, its value, source, and freshness semantics:

- `visiblePrompt`: derived from the current message/terminal and labeled `current_observation`;
- `lowHp`, `urgentHunger`, and `conditionImpairment`: derived from current player fields and labeled `current_status`;
- `combatMessageEvidence`: the existing combat regex result, relabeled `historical_message_only`, with an explicit `activeThreat: unknown` rather than calling it current combat.

The revised instruction keeps visible prompt and directly observed survival status ahead of exploration. It tells Jev that combat-message evidence warrants re-evaluation but does not itself prove a continuing threat; Jev must use the rest of the current observation to choose among the unchanged goals. Refresh occurs when one of these structured values appears, clears, or changes freshness, so a six-action goal cannot remain solely because a prior boolean silently disappeared. This is one goal-interface change: clearer observed state plus matching refresh semantics. It does not add a threat detector, infer unseen monsters, force a goal, or change route/graph facts.

Everything else remains unchanged: observation graph, memory, six-action maximum horizon, goal descriptions, action instructions, raw judgment retention, and the ordered 121 action candidates. Jev chooses among all seven goals and chooses every actual keyboard action. Code does not select, remove, mask, reweight, or override a goal or action. Goal-choice distributions are retained as a manipulation check only; they cannot by themselves satisfy an outcome criterion or support a learned-skill claim.

## Corrected evaluator contract

The evaluator separately recomputes `priority_context` from each retained pre-action state and verifies that the candidate serialized every field truthfully and offered the complete unchanged goal set. It records whether Jev's selected goal is consistent with direct current safety evidence, but a mismatch against `combatMessageEvidence` alone is not a hard failure because that evidence is retrospective. Goal compliance, probability, and confidence remain descriptive.

Replace the menu-label substring gate with a versioned prompt/action contract. Before an action, classify the visible prompt and derive allowed action IDs from action identity and prompt semantics:

- Direction prompts such as “In what direction?” allow the eight single-step `CompassDirection` action IDs and Escape.
- `--More--` allows Space and Enter/MORE.
- Yes/no prompts allow the represented answer keys and Escape.
- Inventory-letter prompts allow only the visible inventory letters and Escape.

A prompt action passes when its action ID is in that pre-action allowed set and the next raw observation shows progress: the prompt clears, changes to a new prompt stage, or the raw/message state changes consistently with a multi-page prompt. An allowed action with an identical unresolved prompt and identical raw observation fails as `prompt_unresolved`. Label prose is presentation data and cannot independently pass or fail. Unknown prompt classes fail closed as `unsupported_prompt_class`.

Apply this evaluator symmetrically to comparator and candidate. Under it, development step 4017 passes as a valid direction answer whose next observation opened the door. Development step 3382 is retained as counterevidence about historical combat state; it is not causally attributed to its goal and does not become a candidate pass. This reclassification validates evaluator behavior only and does not change the frozen trial result.

## Separately preregistered held-out evaluation

Freeze and publicly anchor the candidate source, evaluator, provider model, environment/source hashes, ordered 121-action contract, exact attempt ledger, analysis code, and this plan before selecting a branch point or making a provider call.

An independent deterministic selector chooses six exact-prefix replayable contexts from preserved raw observations that were not used to design the change:

1. one direction-prompt context;
2. one non-direction prompt context;
3. one message-only recent-combat context with no predeclared current-threat evidence;
4. one recent-combat context with predeclared, observation-only current-threat evidence, or end as incomplete if no reliable predicate and eligible state were frozen;
5. one low-HP, urgent-hunger, or condition-impairment context;
6. one neutral context with none of those signals.

Exclude all synthetic fixtures, every prior v1/v2 evaluation point, and at least 64 actions on either side of steps 3359 and 3999. The current-threat predicate must be defined and tested before eligible-set construction; it cannot be inferred manually after selection. Within each stratum, choose the lowest SHA-256 of `episode identity || step || raw-observation digest`. Publish the eligible-set digest, selected identities, exact prefix digests, and exclusion audit before calls. If any stratum lacks a replayable eligible context, end as `incomplete`; do not substitute a point or relax the stratum.

At each selected context, replay the exact prefix with zero provider calls and run two isolated eight-action arms:

- comparator: frozen frontier-v2;
- candidate: frontier-v2 plus only the structured priority-context/interface change above.

This yields 96 actual future actions. Reserve at most two calls per action (goal plus action), for a global ceiling of 192 attempted calls across both arms and all six pairs. Failed, timed-out, malformed, cancelled, and crash-ambiguous calls count. There is no retry, resume, replacement branch, extension, or favorable-outcome rerun. Retain terminal arms. Preserve complete requests, responses, probability arrays, raw before/after observations, transition packs, event journals, and isolated ttyrec footage.

The development events and fixtures may test serialization and evaluator logic, but they cannot enter the eligible set, thresholds, result classification, or held-out claim. The selected contexts are held out only from this narrow design process; six short contexts do not establish generalization.

## Preregistered outcomes

Every arm must either retain all eight actions or retain its actual terminal transition. Replay mismatch, provider/preservation failure, missing stratum, or unevaluable metric makes the entire result `incomplete`.

Hard safety gates use the frozen arm-specific contracts and compare the arms symmetrically where the contracts differ:

- `state_contract`: each arm matches its frozen interface; the candidate's structured priority fields are exactly reproducible from the retained observation, and both arms preserve their full seven-goal and ordered 121-action contracts;
- `prompt_validity`: every visible-prompt action satisfies the corrected prompt/action contract;
- `terminal`: candidate has no new non-ascended terminal relative to its comparator;
- `critical_survival`: in the two preregistered combat/status strata, candidate minimum HP is not lower than comparator minimum HP and candidate acquires no condition bit absent from comparator.

Actual outcome metrics are frozen before calls. Primary progress metrics are aggregate prefix-new positions and aggregate score delta, with both values also retained per pair. Secondary safety numerics include minimum HP, ending HP, hunger-code worsening, condition-bit transitions, and non-ascended terminal count. Goal choice, probability, confidence, priority consistency, and refresh count are reported separately and cannot make a result positive.

Apply the following ordered and exhaustive taxonomy:

1. `incomplete`: any required evidence, context, metric, or arm is incomplete.
2. `negative`: any hard safety gate fails.
3. `positive_development`: every actual outcome metric is at least as good as comparator under its frozen direction, no pair has both primary progress values lower, and at least one primary progress metric is strictly better.
4. `secondary_safety_only`: every primary progress value is equal, every secondary safety numeric is at least as good as comparator, and at least one secondary safety numeric is strictly better.
5. `tie`: every primary progress and secondary safety numeric is exactly equal.
6. `hard_gates_pass_tradeoff`: every remaining result for which all hard safety gates pass, including equal progress with different opposing or worse secondary safety numerics, a progress/safety tradeoff, or mixed pair-level results.

This ordering closes the former gap where all hard gates passed and progress was equal but secondary safety numerics differed. Directionality and aggregation for every numeric must be frozen in code; missing or incomparable values produce `incomplete`, not an inferred tie.

Report depth, HP, hunger, score, turns, distinct positions/raw observations, exact raw no-ops, unchanged-position actions, reversals, blank/wall attempts, goal refresh count, priority-context values, goal distributions, and prompt classes as secondary evidence. Do not infer causation from one post-action observation, treat confidence as accuracy, label code-enforced compliance as skill, tune after seeing held-out results, claim generalization from six contexts, or authorize live migration from this bounded evaluation.

## Evidence pointers

- Jev design classification: receipt `f5d2c5ea-7ac5-4f03-aa16-2650f932d4b7`, input digest `5b59bb6dcf0a23ebf0e182c46409521ace4ec74aa36898b076ad3ead290c1f85`, choice `B_competing_goals_fresher_context`.
- Successor verification receipt: `policy-candidates/frontier-v2-validation-fix/trial-runs/frontier-v2-validation-fix-20260918T160446Z/verification-receipt.json`, SHA-256 `e679a09150255eff146d2d4f65368e52db36dc43da74311993c4b03351ae9fd5`.
- Frozen negative analysis: SHA-256 `f2b4fd15dff1f5893e8918be6a7aaa245e27a367dd94ed78b78c3b488b0ddddc`.
- Retained decision ledger: SHA-256 `a0fef7655ff0580e2a98b38c472a288be4e71316ee0d0912f32aa192d28e9cd8`.
- Candidate interface reviewed: `policy-candidates/frontier-v2-validation-fix/candidate.py`, SHA-256 `b29743bfc43e34b9771a143623e4d7026aef96b80039d11848af6b7751f1bcbb`.
