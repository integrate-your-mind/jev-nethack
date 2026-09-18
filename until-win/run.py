"""Capped, observation-only Jev/NetHack feasibility experiment; not an RL trainer."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time

import gymnasium as gym
import nle
from nle import nethack


INSTRUCTIONS = (
    "Choose exactly one next keyboard action to play NetHack 3.6.7. "
    "Your objective is to survive, explore the dungeon, obtain the Amulet of Yendor, "
    "and ascend. The state contains only player observations and past actions. "
    "Use the named adjacent cells for immediate movement; directions are relative to the player. "
    "A movement into a monster attacks it. Avoid repeatedly walking into a wall or repeating "
    "an ineffective action. Handle any visible menu or question before normal movement. "
    "In an inventory prompt, the same keyboard letter selects that inventory item. "
    "Use the visible terminal text to distinguish movement from menu selection. "
    "Inspect HP, hunger, pets, and dangerous terrain when deciding."
)
DIRECTIONS = {"N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
              "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1)}


def text_line(value):
    return bytes(value).split(b"\0", 1)[0].decode("latin1").rstrip()


def stats(obs):
    b = obs["blstats"]
    names = {"x": "X", "y": "Y", "hp": "HP", "max_hp": "HPMAX", "score": "SCORE",
             "depth": "DEPTH", "dungeon": "DNUM", "level": "DLEVEL", "turn": "TIME",
             "gold": "GOLD", "armor_class": "AC", "experience_level": "XP",
             "hunger_code": "HUNGER", "condition_bits": "CONDITION"}
    return {k: int(b[getattr(nethack, "NLE_BL_" + v)]) for k, v in names.items()}


def make_state(obs, visits, history):
    s = stats(obs)
    s["hunger"] = {0: "satiated", 1: "not hungry", 2: "hungry", 3: "weak",
                   4: "fainting", 5: "fainted", 6: "starved"}.get(s["hunger_code"], "unknown")
    adjacent = {}
    for direction, (dx, dy) in {"HERE": (0, 0), **DIRECTIONS}.items():
        x, y = s["x"] + dx, s["y"] + dy
        if 0 <= y < obs["chars"].shape[0] and 0 <= x < obs["chars"].shape[1]:
            cell = {"symbol": chr(int(obs["chars"][y, x])),
                    "visits": visits[(s["dungeon"], s["level"], x, y)]}
            if "screen_descriptions" in obs:
                cell["description"] = text_line(obs["screen_descriptions"][y, x])
            adjacent[direction] = cell
        else:
            adjacent[direction] = {"description": "outside map boundary"}
    return {"player": s, "adjacent_cells": adjacent,
            "message": text_line(obs["message"]),
            "terminal": "\n".join(text_line(row) for row in obs["tty_chars"]),
            "inventory": [{"letter": chr(int(letter)), "item": text_line(row)}
                          for letter, row in zip(obs["inv_letters"], obs["inv_strs"]) if letter],
            "recent_actions": list(history)}


def action_choices(actions):
    result = {}
    for i, action in enumerate(actions):
        code = int(action)
        key = chr(code) if 32 <= code <= 126 else f"ASCII {code}"
        name = getattr(action, "name", key)
        kind = action.__class__.__name__
        description = f"Press {key!r}: {kind}.{name}"
        if kind == "CompassDirection":
            description += f"; move/attack one step {name} (or choose this letter in a menu)"
        if kind == "CompassDirectionLonger":
            description += f"; run several steps {name}"
        result[f"a{i}"] = description
    return result


def make_env(character):
    # Match the complete keyboard/menu interface, while retaining reproducible
    # development seeds. This is explicitly not the official Challenge class.
    return gym.make("NetHackScore-v0", actions=nethack.ACTIONS,
                    character=character, max_episode_steps=1_000_000,
                    allow_all_modes=True, allow_all_yn_questions=True,
                    penalty_step=0.0, penalty_time=0.0, fix_moon_phase=True)


def fingerprint():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "run.py", root / "jev_client.py") if p.exists()}


def play(args):
    # Exclusive output creation prevents accidental replacement of receipts.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    client = None
    if args.policy == "jev":
        from jev_client import JevClient
        client = JevClient(max_calls=args.episodes * args.steps,
                           max_input_bytes=args.max_input_bytes)
    config = vars(args).copy()
    config.update({"started_utc": datetime.now(timezone.utc).isoformat(),
                   "versions": {x: importlib.metadata.version(x) for x in ("nle", "gymnasium", "numpy")},
                   "source_sha256": fingerprint(), "protocol": "capped feasibility, not SOTA evaluation",
                   "game_version": "NetHack 3.6.7", "observations": "NLE player observations only",
                   "rng": "core=seed; disp=seed+100000; lgen=seed+200000; reseed=False; fix_moon_phase=True",
                   "role_setting": args.character})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summaries, input_tokens, latencies = [], 0, []
    try:
        with (output / "steps.jsonl").open("x") as log:
            for episode in range(args.episodes):
                env = make_env(args.character)
                seed = args.seed + episode
                rng = random.Random(seed)
                visits, discovered = Counter(), set()
                history = deque(maxlen=8)
                total_reward, max_depth, no_turn_steps = 0.0, 0, 0
                terminated = truncated = False
                end_status = None
                reason = "action_cap"
                ascended = False
                started = time.monotonic()
                steps = 0
                obs = None
                try:
                    env.unwrapped.seed(core=seed, disp=seed + 100000, lgen=seed + 200000, reseed=False)
                    obs, info = env.reset()
                    criteria = action_choices(env.unwrapped.actions)
                    for step in range(args.steps):
                        before = stats(obs)
                        visits[(before["dungeon"], before["level"], before["x"], before["y"])] += 1
                        for y, row in enumerate(obs["chars"]):
                            for x, c in enumerate(row):
                                if int(c) != 32:
                                    discovered.add((before["dungeon"], before["level"], x, y))
                        state = make_state(obs, visits, history)
                        if client:
                            decision = client.choose(state, criteria, INSTRUCTIONS)
                            chosen = decision["choice"]
                            input_tokens += decision["usage"]["input_tokens"]
                            latencies.append(decision["latency_seconds"])
                        else:
                            chosen = rng.choice(list(criteria))
                            decision = {"choice": chosen, "policy": "uniform_random"}
                        action = int(chosen[1:])
                        # State is materialized before step: NLE reuses array buffers.
                        obs, reward, terminated, truncated, info = env.step(action)
                        after = stats(obs)
                        ascended = bool(info.get("is_ascended", False))
                        end_status = getattr(info.get("end_status"), "name", str(info.get("end_status")))
                        no_turn_steps += after["turn"] == before["turn"]
                        max_depth = max(max_depth, before["depth"], after["depth"])
                        total_reward += reward
                        steps += 1
                        event = {"episode": episode, "seed": seed, "step": step,
                                 "state": state, "decision": decision, "action_index": action,
                                 "keycode": int(env.unwrapped.actions[action]), "after": after,
                                 "reward": reward, "terminated": bool(terminated),
                                 "truncated": bool(truncated), "is_ascended": ascended,
                                 "end_status": end_status}
                        log.write(json.dumps(event, ensure_ascii=True) + "\n")
                        log.flush()
                        history.append({"action": criteria[chosen], "before": before,
                                        "after": after, "message_after": text_line(obs["message"])})
                        if terminated or truncated:
                            reason = "environment_truncation" if truncated else "game_end"
                            break
                except Exception as exc:
                    reason = "runner_error"
                    # Sanitized client errors contain no provider body or key.
                    (output / "error.json").write_text(json.dumps({"type": type(exc).__name__,
                                                                 "message": str(exc)}) + "\n")
                    raise
                finally:
                    summary = {"episode": episode, "seed": seed, "steps": steps,
                               "score": stats(obs)["score"] if obs is not None else None,
                               "max_depth": max_depth, "observed_nonblank_tiles": len(discovered),
                               "no_turn_steps": no_turn_steps, "reward": total_reward,
                               "is_ascended": ascended, "terminated": bool(terminated),
                               "truncated": bool(truncated), "stop_reason": reason,
                               "end_status": end_status, "wall_seconds": time.monotonic() - started}
                    summaries.append(summary)
                    env.close()
                    print(json.dumps(summary), flush=True)
    finally:
        report = {"protocol": "capped feasibility; scores are not full-episode benchmark results",
                  "episodes": summaries, "input_tokens": input_tokens,
                  "estimated_api_usd": input_tokens * 0.042 / 1_000_000,
                  "price_source": "https://docs.typesafe.ai/models (2026-09-17)",
                  "mean_api_latency_seconds": statistics.mean(latencies) if latencies else None,
                  "successful_decisions": len(latencies), "source_sha256": fingerprint()}
        if client:
            report["attempted_api_calls"] = client.calls_used
            report["reserved_request_bytes"] = client.input_bytes_used
            report["api_cost_note"] = "Estimate for validated responses only; failed/unvalidated calls may also be billed."
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("random", "jev"), default="random")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--character", default="mon-hum-neu-mal")
    parser.add_argument("--max-input-bytes", type=int, default=8_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.max_input_bytes <= 0:
        parser.error("episodes, steps and max-input-bytes must be positive")
    play(args)
