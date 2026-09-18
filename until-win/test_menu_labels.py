"""Offline tests for contextual NetHack menu labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from menu_labels import contextual_choices  # noqa: E402


def criteria_for(codes: list[int]) -> tuple[list[int], dict[str, str]]:
    actions = list(codes)
    criteria = {
        f"a{index}": f"Press {repr(chr(code)) if 32 <= code <= 126 else repr('ASCII ' + str(code))}: base"
        for index, code in enumerate(actions)
    }
    return actions, criteria


class MenuLabelTests(unittest.TestCase):
    def test_exact_inventory_prompt_relabels_only_visible_inventory_options(self) -> None:
        actions, base = criteria_for([ord("c"), ord("d"), ord("l"), 27, ord("x")])
        state = {
            "message": "What do you want to read? [cdl or ?*]",
            "terminal": "",
            "inventory": [
                {"letter": "c", "item": "a scroll of identify"},
                {"letter": "d", "item": "a scroll of light"},
                {"letter": "l", "item": "a scroll of teleportation"},
            ],
        }

        labeled = contextual_choices(state, actions, base)

        self.assertEqual(set(labeled), set(base))
        self.assertEqual(len(labeled), len(base))
        self.assertIn("a scroll of identify", labeled["a0"])
        self.assertIn("current read prompt", labeled["a2"])
        self.assertIn("cancel", labeled["a3"].lower())
        self.assertEqual(labeled["a4"], base["a4"])

    def test_inventory_letter_range_is_supported(self) -> None:
        actions, base = criteria_for([ord("a"), ord("b"), ord("c"), ord("d")])
        state = {
            "message": "Choose an item [a-c or ?*]",
            "inventory": [
                {"letter": "a", "item": "apple"},
                {"letter": "b", "item": "bell"},
                {"letter": "c", "item": "cloak"},
                {"letter": "d", "item": "dagger"},
            ],
        }

        labeled = contextual_choices(state, actions, base)

        for action_id in ("a0", "a1", "a2"):
            self.assertIn("current inventory prompt", labeled[action_id])
        self.assertEqual(labeled["a3"], base["a3"])

    def test_inventory_letters_preserve_case(self) -> None:
        actions, base = criteria_for([ord("A"), ord("a")])
        state = {
            "message": "Choose an item [Aa or ?*]",
            "inventory": [
                {"letter": "A", "item": "a wand"},
                {"letter": "a", "item": "an apple"},
            ],
        }

        labeled = contextual_choices(state, actions, base)

        self.assertIn("[a wand]", labeled["a0"])
        self.assertIn("[an apple]", labeled["a1"])

    def test_yes_no_prompt_labels_y_and_n(self) -> None:
        actions, base = criteria_for([ord("y"), ord("n"), 27, ord("x")])
        state = {"message": "Really attack the peaceful monster? [yn]"}

        labeled = contextual_choices(state, actions, base)

        self.assertIn("yes", labeled["a0"])
        self.assertIn("no", labeled["a1"])
        self.assertIn("cancel", labeled["a2"].lower())
        self.assertEqual(labeled["a3"], base["a3"])

    def test_more_labels_enter_and_space_only_when_visible(self) -> None:
        actions, base = criteria_for([10, 32, 27, ord("x")])
        state = {"terminal": "A long message\n--More--"}

        labeled = contextual_choices(state, actions, base)

        self.assertIn("Enter", labeled["a0"])
        self.assertIn("Space", labeled["a1"])
        self.assertIn("cancel", labeled["a2"].lower())
        self.assertEqual(labeled["a3"], base["a3"])

    def test_no_prompt_leaves_generic_inventory_unchanged(self) -> None:
        actions, base = criteria_for([ord("c"), ord("d"), ord("l"), 27])
        state = {
            "message": "You have 3 items in your inventory.",
            "terminal": "",
            "inventory": [{"letter": "c", "item": "a scroll"}],
        }

        self.assertEqual(contextual_choices(state, actions, base), base)

    def test_unknown_prompt_and_all_121_actions_are_preserved(self) -> None:
        actions = list(range(32, 32 + 121))
        base = {f"a{index}": f"base {index}" for index in range(121)}
        state = {
            "message": "The oracle says something mysterious.",
            "terminal": "",
            "inventory": [{"letter": "c", "item": "a scroll"}],
        }

        labeled = contextual_choices(state, actions, base)

        self.assertEqual(list(labeled), list(base))
        self.assertEqual(len(labeled), 121)
        self.assertEqual(labeled, base)


if __name__ == "__main__":
    unittest.main()
