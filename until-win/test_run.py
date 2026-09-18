import json
import unittest
from collections import Counter, deque
from run import make_env, make_state, action_choices, stats


class AdapterTests(unittest.TestCase):
    def test_seed_replays_and_state_is_a_snapshot(self):
        traces = []
        for _ in range(2):
            env = make_env("mon-hum-neu-mal")
            try:
                env.unwrapped.seed(core=765, disp=100765, lgen=200765, reseed=False)
                obs, info = env.reset()
                self.assertFalse(info["is_ascended"])
                state = make_state(obs, Counter(), deque())
                initial = json.dumps(state, sort_keys=True)
                self.assertNotIn("seed", state)
                self.assertNotIn("internal", state)
                self.assertEqual(len(action_choices(env.unwrapped.actions)), 121)
                frames = [initial]
                for action in [0, 1, 2, 3, 75]:
                    obs, reward, terminated, truncated, info = env.step(action)
                    frames.append((stats(obs), reward, terminated, truncated))
                self.assertEqual(json.dumps(state, sort_keys=True), initial)
                traces.append(frames)
            finally:
                env.close()
        self.assertEqual(traces[0], traces[1])


if __name__ == "__main__":
    unittest.main()
