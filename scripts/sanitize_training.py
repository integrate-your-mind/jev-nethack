"""Create the deliberately small public JSONL training sample."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def sanitize(record: dict) -> dict:
    state = record.get("state")
    if not isinstance(state, dict):
        return {}
    action = record.get("action")
    if isinstance(action, dict):
        action = {key: action[key] for key in ("index", "keycode", "label") if key in action}
    return {
        "episode": record.get("episode", 0),
        "step": record.get("step", 0),
        "phase": record.get("phase", record.get("event_type", "observation")),
        "state": state,
        "action": action,
        "reward": record.get("reward"),
        "terminated": bool(record.get("terminated", False)),
        "truncated": bool(record.get("truncated", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source
    destination = args.destination
    rows = []
    for line in source.read_text().splitlines():
        if line.strip():
            row = sanitize(json.loads(line))
            if row:
                rows.append(row)
        if len(rows) >= 24:
            break
    destination.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
