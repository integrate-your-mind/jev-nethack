# TypeSafe skill and decision support

This project uses the official [TypeSafe skill](https://github.com/typesafe-ai/skills/tree/main/skills/typesafe-ai) and the Jev Workflows plugin for project classifications.

Install it for Codex with one method:

```sh
npx skills add typesafe-ai/skills --skill typesafe-ai --agent codex
```

The skill keeps deterministic execution and observed outcomes in code, with bounded semantic judgments supplied by Jev. The direct game policy sends the observed state and contextual keyboard choices to Jev. Its probabilities describe those choices; they are not a measured NetHack win probability. We retain judgments and observed outcomes for later policy training.

The current plan is direct Jev live play with recovery and data capture. Training a separate policy remains a later stage, with held-out evaluation before performance claims.
