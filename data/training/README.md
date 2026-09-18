# Training-data sample

`sample.jsonl` contains a small sanitized sample of recorded NetHack observations and action outcomes. It is derived from the local `smoke-v2` recording, whose source is the local Jev/NLE runner; the complete recording is intentionally excluded.

Each line has:

```json
{"episode": 0, "step": 0, "phase": "before_action", "state": {...}, "action": null, "reward": null, "terminated": false, "truncated": false}
```

State fields retain the game observation contract (player, message, terminal, adjacent cells, inventory, and recent actions). Timing, stream IDs, API decisions/probabilities, local paths, and network metadata are removed. This sample is suitable for parser and encoder tests; it is not a complete or independently representative training set.
