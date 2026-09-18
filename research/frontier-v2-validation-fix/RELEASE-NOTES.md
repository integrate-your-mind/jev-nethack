# Frontier-v2 validation-fix release notes

Status: completed development trial; preregistered result NEGATIVE.

- Four 32-action arms completed; 143 provider attempts; 0 replay provider calls.
- Aggregate: +14 new positions versus exact prefixes (+8 at 3359, +6 at 3999).
- Actual priority lapse: branch 3359 selected frontier route 0.36 over immediate combat 0.04; south action then jackal damage reduced HP 13 to 12.
- Menu4017 gate: evaluator false positive. Visible prompt “In what direction?” was correctly handled by `resolve_visible_prompt` at 0.98 and SE opened the door, though the frozen classifier recorded a menu mismatch.
- Negative classification and no-migration decision remain unchanged. No tuning or rerun.
