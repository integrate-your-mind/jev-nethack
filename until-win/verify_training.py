"""Offline integrity scan for Jev NetHack raw transition packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_pack import TransitionPackError, validate_shard


def verify(data_root: Path) -> dict:
    packs = sorted(data_root.glob("fragments/*/training/*.npzpack"))
    result = {
        "dataRoot": str(data_root),
        "packs": [],
        "packCount": len(packs),
        "transitionCount": 0,
        "valid": True,
        "complete": bool(packs),
    }
    for pack in packs:
        item = {"path": str(pack.relative_to(data_root)), "records": 0, "valid": False, "complete": False}
        try:
            validation = validate_shard(pack)
            frames = validation["frames"]
            item["records"] = validation["records"]
            item["indexedRecords"] = validation["indexedRecords"]
            item["partialTail"] = validation["partialTail"]
            item["complete"] = validation["completed"]
            if frames:
                item["lastEventId"] = frames[-1].metadata.get("eventId")
            item["valid"] = True
            result["transitionCount"] += item["records"]
            result["complete"] = result["complete"] and item["complete"]
        except TransitionPackError as exc:
            item["error"] = str(exc)
            result["valid"] = False
            result["complete"] = False
        result["packs"].append(item)

    manifests = sorted(data_root.glob("fragments/*/training/training-manifest.json"))
    result["manifests"] = []
    for manifest_path in manifests:
        item = {"path": str(manifest_path.relative_to(data_root)), "valid": False, "complete": False}
        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError
            item["records"] = manifest.get("records")
            item["complete"] = manifest.get("completed") is True
            item["valid"] = isinstance(manifest.get("records"), int) and isinstance(manifest.get("artifacts"), list)
            if not item["valid"]:
                result["valid"] = False
            result["complete"] = result["complete"] and item["complete"]
        except (OSError, ValueError):
            item["error"] = "training manifest is invalid"
            result["valid"] = False
            result["complete"] = False
        result["manifests"].append(item)
    if packs and not manifests:
        result["complete"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.data_root.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
