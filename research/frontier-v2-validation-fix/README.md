# Frontier-v2 validation-fix successor result

Final validated development result. Four 32-action arms completed with 143 attempted provider calls and +14 new cells, but the preregistered classification remains **negative**.

The final receipt records two frozen safety failures. At priority branch 3359, the recent-combat priority gate was violated: the candidate selected a frontier route (0.36) over immediate combat (0.04), then took south and jackal damage (HP 13 to 12). At menu step 4017, the frozen classifier flagged a menu mismatch, but this is an evaluator false positive: the visible prompt was “In what direction?”, Jev selected `resolve_visible_prompt` at 0.98, and SE successfully opened the door. The negative result is preserved while this limitation is disclosed.

Development-only: no live migration, training, ascension, SOTA, tuning, or rerun is authorized. The predecessor is linked by archive hash only.

The complete preserved harness, frozen contracts, ledger, all request/response distributions, binary transition packs, and ttyrec footage are in [release v0.5.0-frontier-v2-negative](https://github.com/integrate-your-mind/jev-nethack/releases/tag/v0.5.0-frontier-v2-negative). Archive SHA-256: `47c358d2310d037dbe6c5d6ee5c2d005e42c520af8e75edb21f2779fa01c9b4c`.

`MEMBERS.sha256.json` describes files in the full archive, not just this source subset. Private absolute text paths in archived metadata/code are replaced with `JEV_NETHACK_DATA_ROOT` and recorded in the archive's original/public hash mapping. This is an evidence archive; those path tokens are not executable environment-variable substitutions, and the frozen one-use trial must not be rerun. The local original verification receipt remains hash-bound separately.

The pre-call [public preregistration](../frontier-v2-validation-fix-preregistration/README.md) remains unchanged. The post-trial Jev assessment selected preserving the negative result with the evaluator limitation disclosed (receipt `9ef3a4db-5f4a-4451-bfa3-6ba179111cb1`). That interpretation label does not change the preregistered classification `negative`.
