from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import metrics
from metrics import episode_metric_fields, terminal_metrics


def episode(root: Path, episode_id: int = 0, seed: int = 103) -> Path:
    path = root / f"episode-{episode_id:08d}-seed-{seed}-replay"
    (path / "nle-ttyrec").mkdir(parents=True)
    return path


def valid(points: str = "138", *, pid: int = 22464, ttyrec_pid: int | None = None) -> str:
    retained_pid = pid if ttyrec_pid is None else ttyrec_pid
    return (
        f"version=3.6.7\tpoints={points}\tdeath=died of starvation\twhile=fainted"
        f"\tturns=4732\tttyrecname=nle.{retained_pid}.0.ttyrec3.bz2\n"
    )


def write_bound_xlog(root: Path, text: str | None = None, *, pid: int = 22464) -> Path:
    ttyrec_dir = root / "nle-ttyrec"
    (ttyrec_dir / f"nle.{pid}.0.ttyrec3.bz2").write_bytes(b"retained terminal frames")
    path = ttyrec_dir / f"nle.{pid}.xlogfile"
    path.write_text(valid(pid=pid) if text is None else text)
    return path


class NativeMetricTests(unittest.TestCase):
    def test_valid_xlog_is_bound_to_episode_seed_process_and_ttyrec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = episode(Path(temporary))
            write_bound_xlog(root)
            result = terminal_metrics(root, expected_episode_id=0, expected_seed=103)
            self.assertEqual(138, result["officialFinalScore"])
            self.assertEqual("died of starvation", result["deathCause"])
            self.assertEqual("fainted", result["deathWhile"])
            evidence = result["officialScoreEvidence"]
            self.assertEqual("native_xlogfile", evidence["source"])
            self.assertEqual("nle-ttyrec/nle.22464.xlogfile", evidence["relativePath"])
            self.assertEqual("nle-ttyrec/nle.22464.0.ttyrec3.bz2", evidence["ttyrecRelativePath"])
            self.assertEqual(0, evidence["episodeId"])
            self.assertEqual(103, evidence["seed"])
            self.assertGreater(evidence["xlogBytes"], 0)
            self.assertEqual(64, len(evidence["sha256"]))

    def test_large_sparse_ttyrec_is_stat_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = episode(Path(temporary))
            write_bound_xlog(root)
            ttyrec_path = root / "nle-ttyrec" / "nle.22464.0.ttyrec3.bz2"
            sparse_size = 512 * 1024 * 1024
            with ttyrec_path.open("r+b") as ttyrec:
                ttyrec.truncate(sparse_size)

            original_reader = metrics._read_regular_nofollow

            def xlog_only_reader(path: Path, *, max_bytes: int | None = None):
                self.assertTrue(path.name.endswith(".xlogfile"))
                return original_reader(path, max_bytes=max_bytes)

            with patch.object(metrics, "_read_regular_nofollow", side_effect=xlog_only_reader):
                result = metrics.terminal_metrics(root, expected_episode_id=0, expected_seed=103)

            self.assertEqual(138, result["officialFinalScore"])
            self.assertEqual(sparse_size, result["officialScoreEvidence"]["ttyrecBytes"])

    def test_missing_or_ambiguous_xlog_is_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = episode(Path(temporary))
            missing = terminal_metrics(root, expected_episode_id=0, expected_seed=103)
            self.assertIsNone(missing["officialFinalScore"])
            self.assertEqual("missing_xlogfile", missing["officialScoreError"])
            write_bound_xlog(root)
            (root / "nle-ttyrec" / "nle.999.xlogfile").write_text(valid(pid=999))
            ambiguous = terminal_metrics(root, expected_episode_id=0, expected_seed=103)
            self.assertIsNone(ambiguous["officialFinalScore"])
            self.assertEqual("ambiguous_xlogfile", ambiguous["officialScoreError"])

    def test_malformed_unbound_or_missing_ttyrec_never_becomes_official(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            malformed_root = episode(base / "malformed")
            write_bound_xlog(malformed_root, "version=3.6.7\tpoints=not-a-number\n")
            self.assertEqual(
                "malformed_xlogfile",
                terminal_metrics(malformed_root, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

            mismatch_root = episode(base / "mismatch")
            write_bound_xlog(mismatch_root, valid(ttyrec_pid=999))
            self.assertEqual(
                "xlog_ttyrec_binding_mismatch",
                terminal_metrics(mismatch_root, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

            missing_ttyrec_root = episode(base / "missing-ttyrec")
            (missing_ttyrec_root / "nle-ttyrec" / "nle.22464.xlogfile").write_text(valid())
            self.assertEqual(
                "missing_bound_ttyrec",
                terminal_metrics(missing_ttyrec_root, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

    def test_episode_identity_seed_and_symlink_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = episode(base)
            write_bound_xlog(root)
            self.assertEqual(
                "episode_id_mismatch",
                terminal_metrics(root, expected_episode_id=1, expected_seed=103)["officialScoreError"],
            )
            self.assertEqual(
                "episode_seed_mismatch",
                terminal_metrics(root, expected_episode_id=0, expected_seed=104)["officialScoreError"],
            )
            self.assertEqual(
                "unbound_episode_identity",
                terminal_metrics(root, expected_episode_id=None, expected_seed=None)["officialScoreError"],
            )

            symlink_root = episode(base / "symlink")
            external = base / "external.xlogfile"
            external.write_text(valid())
            (symlink_root / "nle-ttyrec" / "nle.22464.xlogfile").symlink_to(external)
            self.assertEqual(
                "unsafe_xlogfile",
                terminal_metrics(symlink_root, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

    def test_episode_and_ttyrec_directories_may_not_be_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            ttyrec_link_root = episode(base / "ttyrec-link")
            (ttyrec_link_root / "nle-ttyrec").rmdir()
            external_ttyrec = base / "external-ttyrec"
            external_ttyrec.mkdir()
            (ttyrec_link_root / "nle-ttyrec").symlink_to(external_ttyrec, target_is_directory=True)
            write_bound_xlog(ttyrec_link_root)
            self.assertEqual(
                "unsafe_ttyrec_directory",
                terminal_metrics(ttyrec_link_root, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

            real_episode = episode(base / "real")
            write_bound_xlog(real_episode)
            link_parent = base / "linked"
            link_parent.mkdir()
            linked_episode = link_parent / real_episode.name
            linked_episode.symlink_to(real_episode, target_is_directory=True)
            self.assertEqual(
                "unsafe_episode_directory",
                terminal_metrics(linked_episode, expected_episode_id=0, expected_seed=103)["officialScoreError"],
            )

            real_parent = base / "ancestor-real"
            nested_episode = episode(real_parent)
            write_bound_xlog(nested_episode)
            ancestor_link = base / "ancestor-link"
            ancestor_link.symlink_to(real_parent, target_is_directory=True)
            aliased_episode = ancestor_link / nested_episode.name
            self.assertEqual(
                "unsafe_episode_directory",
                terminal_metrics(
                    aliased_episode,
                    expected_episode_id=0,
                    expected_seed=103,
                    trusted_root=base,
                )["officialScoreError"],
            )

    def test_official_observed_max_and_reward_remain_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = episode(Path(temporary))
            write_bound_xlog(root, valid("138"))
            result = episode_metric_fields(
                episode_dir=root,
                expected_episode_id=0,
                expected_seed=103,
                last_observed_stats={"score": 102, "hp": 18},
                max_observed_score=120,
                total_reward=77.5,
            )
            self.assertEqual(138, result["score"])
            self.assertEqual("official_final", result["scoreSemantics"])
            self.assertEqual(138, result["officialFinalScore"])
            self.assertEqual(102, result["lastObservedScore"])
            self.assertEqual(120, result["maxObservedScore"])
            self.assertEqual(77.5, result["totalReward"])

    def test_malformed_xlog_falls_back_only_to_last_observed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = episode(Path(temporary))
            write_bound_xlog(root, "version=3.6.7\tpoints=not-a-number\n")
            result = episode_metric_fields(
                episode_dir=root,
                expected_episode_id=0,
                expected_seed=103,
                last_observed_stats={"score": 102},
                max_observed_score=120,
                total_reward=138.0,
            )
            self.assertIsNone(result["officialFinalScore"])
            self.assertEqual(102, result["score"])
            self.assertEqual("last_observed", result["scoreSemantics"])
            self.assertEqual(120, result["maxObservedScore"])
            self.assertEqual(138.0, result["totalReward"])


if __name__ == "__main__":
    unittest.main()
