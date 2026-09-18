"""Small, local recurrent behavior-cloning and PPO pilot for the Jev/NLE log.

This is deliberately a bounded research runner.  It makes no network calls and
does not add a reward signal: PPO optimizes the native ``NetHackScore-v0``
reward.  ``run.make_state`` is used for both demonstrations and live rollouts,
and :class:`StateEncoder` is shared by BC, PPO, and evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_DIM = 128
GRU_DIM = 128
DEFAULT_GAMMA = 0.99
DEFAULT_LAMBDA = 0.95
TOKEN_RE = re.compile(r"[A-Za-z0-9_@+#!?=./:-]+")
PRESS_KEY_RE = re.compile(r"Press '([^']+)'", re.IGNORECASE)


def _bucket(token: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little") % buckets


def _state_tokens(state: dict[str, Any]) -> Iterable[str]:
    """Yield stable, compact tokens from the JSON state representation."""
    player = state.get("player", {})
    if isinstance(player, dict):
        for key in ("depth", "dlevel", "level", "hunger", "condition", "ac"):
            if key in player:
                yield f"player_{key}={player[key]}"
    for key in ("message", "terminal"):
        value = state.get(key, "")
        if isinstance(value, str):
            yield from (f"{key}:{token}" for token in TOKEN_RE.findall(value.lower()))
    adjacent = state.get("adjacent_cells", {})
    if isinstance(adjacent, dict):
        for direction in sorted(adjacent):
            cell = adjacent[direction]
            if isinstance(cell, dict):
                yield f"adj_{direction}_symbol={cell.get('symbol', '?')}"
                yield f"adj_{direction}_desc={cell.get('description', '')}"
    inventory = state.get("inventory", [])
    if isinstance(inventory, list):
        for item in inventory:
            if isinstance(item, dict):
                yield f"item_{item.get('letter', '?')}=" + str(item.get("item", ""))
    history = state.get("recent_actions", [])
    if isinstance(history, list):
        for item in history[-8:]:
            if isinstance(item, dict):
                action = item.get("action", item.get("action_index", "?"))
            else:
                action = item
            if isinstance(action, str):
                match = PRESS_KEY_RE.search(action)
                action = f"key:{match.group(1)}" if match else "text:" + action[:80]
            else:
                action = f"index:{action}"
            yield "history_action=" + str(action)


class StateEncoder:
    """Parameter-free encoder, so teacher and environment states share a contract.

    The first 16 dimensions carry clipped numeric player features.  The rest are
    a deterministic hashed bag of terminal/message/map/inventory/history tokens.
    Hashing is used only for a compact local pilot; it is not a learned text
    representation and is intentionally recorded in the run configuration.
    """

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        if input_dim < 32:
            raise ValueError("input_dim must be at least 32")
        self.input_dim = input_dim

    def encode(self, state: dict[str, Any]) -> list[float]:
        vector = [0.0] * self.input_dim
        player = state.get("player", {})
        numeric_keys = ("x", "y", "hp", "max_hp", "score", "depth", "dungeon", "level",
                        "turn", "gold", "armor_class", "experience_level", "hunger_code",
                        "condition_bits", "time", "AC")
        if isinstance(player, dict):
            for index, key in enumerate(numeric_keys[:16]):
                value = player.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vector[index] = max(-1.0, min(1.0, float(value) / (1000.0 if key in {"turn", "score", "gold"} else 100.0)))
        buckets = self.input_dim - 16
        for token in _state_tokens(state):
            vector[16 + _bucket(token, buckets)] += 0.1
        scale = max(1.0, max(abs(value) for value in vector[16:]))
        for index in range(16, self.input_dim):
            vector[index] = max(-1.0, min(1.0, vector[index] / scale))
        return vector

    def batch(self, states: list[dict[str, Any]], *, device: torch.device | None = None) -> Tensor:
        return torch.tensor([self.encode(state) for state in states], dtype=torch.float32, device=device)


class RecurrentPolicy(nn.Module):
    """GRU actor-critic with explicit per-step hidden-state reset."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = GRU_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int = 1, *, device: torch.device | None = None) -> Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward_sequence(self, observations: Tensor, hidden: Tensor | None = None,
                         reset_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        if observations.ndim != 3:
            raise ValueError("observations must have shape [T,D] or [B,T,D]")
        batch, steps, _ = observations.shape
        hidden = self.initial_hidden(batch, device=observations.device) if hidden is None else hidden
        if reset_mask is None:
            reset_mask = torch.zeros(batch, steps, dtype=torch.bool, device=observations.device)
        if reset_mask.ndim == 1:
            reset_mask = reset_mask.unsqueeze(0)
        features = self.encoder(observations)
        outputs: list[Tensor] = []
        for step in range(steps):
            mask = reset_mask[:, step].view(1, batch, 1)
            hidden = hidden * (~mask).to(hidden.dtype)
            output, hidden = self.gru(features[:, step:step + 1], hidden)
            outputs.append(output[:, 0])
        sequence = torch.stack(outputs, dim=1)
        return self.actor(sequence), self.critic(sequence).squeeze(-1), hidden

    def step(self, observation: Tensor, hidden: Tensor | None = None, *, reset: bool = False) -> tuple[Categorical, Tensor, Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        reset_mask = torch.tensor([[reset]], dtype=torch.bool, device=observation.device)
        logits, values, hidden_out = self.forward_sequence(observation.unsqueeze(1), hidden, reset_mask)
        return Categorical(logits=logits[:, 0]), values[:, 0], hidden_out


def compute_gae(rewards: Tensor, values: Tensor, next_values: Tensor,
                terminated: Tensor, episode_done: Tensor, gamma: float = DEFAULT_GAMMA,
                gae_lambda: float = DEFAULT_LAMBDA) -> tuple[Tensor, Tensor]:
    """Compute GAE; terminal states do not bootstrap, truncations do.

    ``episode_done`` stops eligibility traces at both terminal and truncated
    boundaries, while ``terminated`` alone controls value bootstrapping.  This
    distinction is the key correctness detail for capped NetHack episodes.
    """
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == episode_done.shape):
        raise ValueError("GAE inputs must have equal shape")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 1.0 - terminated[index].to(rewards.dtype)
        trace = 1.0 - episode_done[index].to(rewards.dtype)
        delta = rewards[index] + gamma * next_values[index] * bootstrap - values[index]
        running = delta + gamma * gae_lambda * trace * running
        advantages[index] = running
    return advantages, advantages + values


def _seed_env(env: Any, seed: int) -> None:
    """Seed the NLE engines explicitly; Gym reset(seed=...) is insufficient."""
    env.unwrapped.seed(core=seed, disp=seed + 100000, lgen=seed + 200000, reseed=False)


def _state_digest(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def load_teacher_data(path: str | Path, action_dim: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid teacher JSON on line {line_number}") from exc
            action = record.get("action_index")
            state = record.get("state")
            if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < action_dim:
                raise ValueError(f"teacher action_index out of range on line {line_number}")
            if not isinstance(state, dict):
                raise ValueError(f"teacher state must be an object on line {line_number}")
            records.append(record)
    if not records:
        raise ValueError("teacher data contains no records")
    return records


def _group_teacher(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[record.get("episode", 0)].append(record)
    return [sorted(items, key=lambda item: item.get("step", 0)) for _, items in sorted(episodes.items(), key=lambda pair: str(pair[0]))]


def behavior_clone(policy: RecurrentPolicy, encoder: StateEncoder, records: list[dict[str, Any]],
                   *, epochs: int, learning_rate: float, bptt: int, device: torch.device,
                   log: list[dict[str, Any]]) -> None:
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    episodes = _group_teacher(records)
    policy.train()
    for epoch in range(epochs):
        losses: list[float] = []
        correct = 0
        count = 0
        for episode in episodes:
            hidden = policy.initial_hidden(device=device)
            for start in range(0, len(episode), bptt):
                chunk = episode[start:start + bptt]
                observations = encoder.batch([item["state"] for item in chunk], device=device).unsqueeze(0)
                targets = torch.tensor([item["action_index"] for item in chunk], dtype=torch.long, device=device)
                logits, _, hidden = policy.forward_sequence(observations, hidden)
                loss = nn.functional.cross_entropy(logits[0], targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                hidden = hidden.detach()
                losses.append(float(loss.detach()))
                correct += int((logits[0].argmax(-1) == targets).sum())
                count += len(chunk)
        log.append({"phase": "bc", "epoch": epoch + 1, "loss": sum(losses) / max(1, len(losses)),
                    "accuracy": correct / max(1, count), "records": count})


class RolloutCollector:
    """Persistent one-environment collector shared by bounded PPO updates."""

    def __init__(self, policy: RecurrentPolicy, encoder: StateEncoder, *, character: str,
                 seed: int, device: torch.device) -> None:
        import run
        self.run = run
        self.policy = policy
        self.encoder = encoder
        self.device = device
        self.seed = seed
        self.episodes_started = 0
        self.env = run.make_env(character)
        _seed_env(self.env, seed)
        self.obs, _ = self.env.reset()
        self.visits, self.history = run.Counter(), run.deque(maxlen=8)
        self.criteria = run.action_choices(self.env.unwrapped.actions)
        self.hidden = policy.initial_hidden(device=device)
        self.first = True
        self.prepared_next = False

    def close(self) -> None:
        self.env.close()

    def collect(self, *, steps: int, gamma: float, gae_lambda: float) -> dict[str, Any]:
        run = self.run
        env = self.env
        policy = self.policy
        observations: list[Tensor] = []
        actions: list[int] = []
        old_log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[float] = []
        next_values: list[Tensor] = []
        terminated: list[bool] = []
        episode_done: list[bool] = []
        resets: list[bool] = []
        scores: list[float] = []
        initial_hidden = self.hidden.detach().clone()
        initial_state_digest: str | None = None
        for _ in range(steps):
            before = run.stats(self.obs)
            if not self.prepared_next:
                self.visits[(before["dungeon"], before["level"], before["x"], before["y"])] += 1
            current = run.make_state(self.obs, self.visits, self.history)
            if initial_state_digest is None:
                initial_state_digest = _state_digest(current)
            encoded = self.encoder.batch([current], device=self.device)[0]
            distribution, value, hidden_after = policy.step(encoded, self.hidden, reset=self.first)
            action = int(distribution.sample())
            log_prob = distribution.log_prob(torch.tensor(action, device=self.device))[0]
            obs_after, reward, is_terminated, is_truncated, _ = env.step(action)
            after = run.stats(obs_after)
            self.visits[(after["dungeon"], after["level"], after["x"], after["y"])] += 1
            self.history.append({"action": self.criteria[f"a{action}"]})
            after_state = run.make_state(obs_after, self.visits, self.history)
            with torch.no_grad():
                if is_terminated:
                    bootstrap = torch.zeros((), device=self.device)
                else:
                    _, bootstrap_values, _ = policy.step(self.encoder.batch([after_state], device=self.device)[0], hidden_after, reset=False)
                    bootstrap = bootstrap_values[0]
            done = bool(is_terminated or is_truncated)
            observations.append(encoded)
            actions.append(action)
            old_log_probs.append(log_prob.detach())
            values.append(value[0].detach())
            next_values.append(bootstrap.detach())
            rewards.append(float(reward))
            terminated.append(bool(is_terminated))
            episode_done.append(done)
            resets.append(self.first)
            scores.append(float(after.get("score", 0)))
            if done:
                self.episodes_started += 1
                next_seed = self.seed + self.episodes_started
                _seed_env(env, next_seed)
                self.obs, _ = env.reset()
                self.visits, self.history = run.Counter(), run.deque(maxlen=8)
                self.hidden = policy.initial_hidden(device=self.device)
                self.first = True
                self.prepared_next = False
            else:
                self.obs = obs_after
                self.hidden = hidden_after.detach()
                self.first = False
                self.prepared_next = True
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        value_tensor = torch.stack(values)
        next_value_tensor = torch.stack(next_values)
        terminal_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        done_tensor = torch.tensor(episode_done, dtype=torch.bool, device=self.device)
        advantages, returns = compute_gae(reward_tensor, value_tensor, next_value_tensor, terminal_tensor, done_tensor, gamma, gae_lambda)
        return {"observations": torch.stack(observations), "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
                "old_log_probs": torch.stack(old_log_probs), "old_values": value_tensor, "advantages": advantages,
                "returns": returns, "resets": torch.tensor(resets, dtype=torch.bool, device=self.device),
                "scores": scores, "rewards": rewards, "terminated": terminated, "truncated": [d and not t for d, t in zip(episode_done, terminated)],
                "initial_state_digest": initial_state_digest, "initial_hidden": initial_hidden}


def collect_rollout(policy: RecurrentPolicy, encoder: StateEncoder, *, character: str, steps: int,
                    seed: int, device: torch.device, gamma: float, gae_lambda: float) -> dict[str, Any]:
    """Compatibility helper for tests and one-off probes; train() persists its collector."""
    collector = RolloutCollector(policy, encoder, character=character, seed=seed, device=device)
    try:
        return collector.collect(steps=steps, gamma=gamma, gae_lambda=gae_lambda)
    finally:
        collector.close()


def ppo_update(policy: RecurrentPolicy, rollout: dict[str, Any], *, epochs: int, learning_rate: float,
               clip_epsilon: float, entropy_coefficient: float, value_coefficient: float,
               log: list[dict[str, Any]], optimizer: torch.optim.Optimizer | None = None) -> None:
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    observations = rollout["observations"]
    advantages = rollout["advantages"]
    if rollout["old_log_probs"].shape != rollout["actions"].shape:
        raise ValueError("old_log_probs and actions must have identical [T] shapes")
    if rollout["old_values"].shape != rollout["returns"].shape:
        raise ValueError("old_values and returns must have identical [T] shapes")
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    policy.train()
    for epoch in range(epochs):
        logits, values, _ = policy.forward_sequence(observations, hidden=rollout.get("initial_hidden"), reset_mask=rollout["resets"])
        distribution = Categorical(logits=logits[0] if logits.ndim == 3 else logits)
        log_probs = distribution.log_prob(rollout["actions"])
        ratio = torch.exp(log_probs - rollout["old_log_probs"])
        policy_loss = -torch.minimum(ratio * advantages, torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages).mean()
        value_loss = 0.5 * (values.squeeze(0) - rollout["returns"]).pow(2).mean() if values.ndim == 2 else 0.5 * (values - rollout["returns"]).pow(2).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        log.append({"phase": "ppo", "epoch": epoch + 1, "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()), "clip_fraction": float((torch.abs(ratio - 1) > clip_epsilon).float().mean())})


@torch.no_grad()
def evaluate(policy: RecurrentPolicy, encoder: StateEncoder, *, character: str, seed: int,
             steps: int, device: torch.device) -> list[dict[str, Any]]:
    import run

    policy.eval()
    results: list[dict[str, Any]] = []
    for episode in range(3):
        env = run.make_env(character)
        episode_seed = seed + 10_000 + episode
        hidden = policy.initial_hidden(device=device)
        visits, history = run.Counter(), run.deque(maxlen=8)
        total_reward = 0.0
        terminated = truncated = False
        step_count = 0
        stop_reason = "unknown"
        is_ascended = False
        end_status = None
        initial_state_digest: str | None = None
        try:
            _seed_env(env, episode_seed)
            obs, _ = env.reset()
            criteria = run.action_choices(env.unwrapped.actions)
            for _ in range(steps):
                stats_before = run.stats(obs)
                visits[(stats_before["dungeon"], stats_before["level"], stats_before["x"], stats_before["y"])] += 1
                state = run.make_state(obs, visits, history)
                if initial_state_digest is None:
                    initial_state_digest = _state_digest(state)
                distribution, _, hidden = policy.step(encoder.batch([state], device=device)[0], hidden, reset=step_count == 0)
                action = int(distribution.probs.argmax())
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                total_reward += float(reward)
                history.append({"action": criteria[f"a{action}"]})
                if terminated or truncated:
                    stop_reason = "environment_truncation" if truncated else "game_end"
                    is_ascended = bool(info.get("is_ascended", False))
                    end_status = getattr(info.get("end_status"), "name", str(info.get("end_status")))
                    break
            else:
                truncated = True
                stop_reason = "evaluation_cap"
        finally:
            env.close()
        final_stats = run.stats(obs)
        results.append({"episode": episode, "seed": episode_seed, "steps": step_count,
                        "score": final_stats.get("score"), "reward": total_reward,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "censored_score": not bool(terminated), "is_ascended": is_ascended,
                        "end_status": end_status, "initial_state_digest": initial_state_digest,
                        "stop_reason": stop_reason})
    return results


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_digest(policy: RecurrentPolicy) -> str:
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def train(args: argparse.Namespace) -> Path:
    if args.mode in {"bc", "bc-ppo"} and not args.teacher_data:
        raise ValueError("--teacher-data is required for bc and bc-ppo")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    import run

    env = run.make_env(args.character)
    try:
        action_dim = len(env.unwrapped.actions)
    finally:
        env.close()
    encoder = StateEncoder(args.input_dim)
    policy = RecurrentPolicy(args.input_dim, action_dim, args.hidden_dim).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = load_teacher_data(args.teacher_data, action_dim) if args.teacher_data else []
    teacher_seeds = sorted({record.get("seed") for record in records if isinstance(record.get("seed"), int)})
    evaluation_seeds = [args.seed + 10_000 + index for index in range(3)]
    overlapping_seeds = sorted(set(teacher_seeds).intersection(evaluation_seeds))
    if overlapping_seeds:
        raise ValueError(f"teacher and evaluation seeds overlap: {overlapping_seeds}")
    config = vars(args).copy()
    config.update({"started_utc": datetime.now(timezone.utc).isoformat(), "action_dim": action_dim,
                  "input_dim": args.input_dim, "hidden_dim": args.hidden_dim, "device": "cpu",
                  "api_calls": 0, "reward": "native NetHackScore-v0 only", "encoder": "StateEncoder-v1",
                  "seed_scheme": "core=seed, disp=seed+100000, lgen=seed+200000, reseed=False; eval=seed+10000..+10002",
                  "teacher_seeds": teacher_seeds, "evaluation_seeds": evaluation_seeds,
                  "teacher_evaluation_seed_overlap": overlapping_seeds,
                  "initial_policy_sha256": _policy_digest(policy),
                  "nle": importlib.metadata.version("nle"), "gymnasium": importlib.metadata.version("gymnasium"),
                  "source_sha256": {name: _hash_file(ROOT / name) for name in ("train.py", "run.py")}})
    if args.teacher_data:
        config["teacher_data_sha256"] = _hash_file(Path(args.teacher_data))
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    log: list[dict[str, Any]] = []
    if args.mode in {"bc", "bc-ppo"}:
        behavior_clone(policy, encoder, records, epochs=args.bc_epochs, learning_rate=args.bc_lr,
                       bptt=args.bptt, device=device, log=log)
        torch.save({"model": policy.state_dict(), "config": config, "phase": "bc",
                    "policy_sha256": _policy_digest(policy)}, output / "checkpoint-bc.pt")
    if args.mode in {"ppo", "bc-ppo"}:
        ppo_optimizer = torch.optim.Adam(policy.parameters(), lr=args.ppo_lr)
        collector = RolloutCollector(policy, encoder, character=args.character, seed=args.seed, device=device)
        remaining = args.steps
        update_index = 0
        try:
            while remaining > 0:
                rollout_steps = min(args.rollout_steps, remaining)
                rollout = collector.collect(steps=rollout_steps, gamma=args.gamma, gae_lambda=args.gae_lambda)
                ppo_update(policy, rollout, epochs=args.ppo_epochs, learning_rate=args.ppo_lr,
                           clip_epsilon=args.clip_epsilon, entropy_coefficient=args.entropy_coef,
                           value_coefficient=args.value_coef, log=log, optimizer=ppo_optimizer)
                log.append({"phase": "rollout", "update": update_index + 1, "steps": rollout_steps,
                            "mean_step_native_score": sum(rollout["scores"]) / len(rollout["scores"]),
                            "mean_reward": sum(rollout["rewards"]) / len(rollout["rewards"]),
                            "terminated": sum(rollout["terminated"]), "truncated": sum(rollout["truncated"]),
                            "initial_state_digest": rollout["initial_state_digest"]})
                remaining -= rollout_steps
                update_index += 1
        finally:
            collector.close()
    config["final_policy_sha256"] = _policy_digest(policy)
    torch.save({"model": policy.state_dict(), "config": config, "phase": "final",
                "policy_sha256": config["final_policy_sha256"]}, output / "checkpoint-final.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "learning.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in log))
    evaluations = evaluate(policy, encoder, character=args.character, seed=args.seed, steps=args.eval_steps, device=device)
    (output / "eval.json").write_text(json.dumps({"episodes": evaluations, "native_score_only": True,
                                                   "api_calls": 0, "cap_steps": args.eval_steps}, indent=2) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-data", help="Jev steps.jsonl (required for bc/bc-ppo)")
    parser.add_argument("--mode", choices=("ppo", "bc", "bc-ppo"), default="bc-ppo")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", required=True)
    parser.add_argument("--character", default="mon-hum-neu-mal")
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=INPUT_DIM)
    parser.add_argument("--hidden-dim", type=int, default=GRU_DIM)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-lr", type=float, default=2.5e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_LAMBDA)
    return parser


if __name__ == "__main__":
    cli = build_parser()
    arguments = cli.parse_args()
    if arguments.steps <= 0 or arguments.eval_steps <= 0 or arguments.rollout_steps <= 0 or arguments.bptt <= 0 or arguments.bc_epochs <= 0 or arguments.ppo_epochs <= 0:
        cli.error("steps, eval-steps, rollout-steps, bptt, and epoch counts must be positive")
    train(arguments)
