import tempfile
import unittest
from pathlib import Path

from hm_recsys.m2 import TABULAR_FEATURES
from hm_recsys.m22 import (
    FEATURE_GROUPS,
    _require_bound_file,
    analyze_groups,
    build_ablation_features,
    validate_feature_groups,
)


def _ordering(value: float) -> dict:
    return {"segments": {"overall": {"map@12": value}}}


class M22Tests(unittest.TestCase):
    def test_feature_groups_are_an_exact_partition(self):
        validate_feature_groups()
        flattened = [
            feature for features in FEATURE_GROUPS.values() for feature in features
        ]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(TABULAR_FEATURES))
        variants = build_ablation_features()
        self.assertEqual(len(variants), 2 * len(FEATURE_GROUPS))
        for group, features in FEATURE_GROUPS.items():
            self.assertTrue(set(features).issubset(variants[f"add_{group}"]))
            self.assertTrue(set(features).isdisjoint(variants[f"drop_{group}"]))

    def test_group_analysis_distinguishes_helpful_harmful_and_interaction(self):
        development = {}
        for window_index, window in enumerate(("dev_a", "dev_b")):
            offset = window_index * 0.001
            orderings = {
                "retrieval": _ordering(0.100 + offset),
                "full": _ordering(0.120 + offset),
                "full__inactive_rrf": _ordering(0.121 + offset),
            }
            for group in FEATURE_GROUPS:
                if group == "customer_static":
                    add = 0.110 + offset
                    drop = 0.115 + offset
                    drop_fallback = 0.120 + offset
                elif group == "user_history":
                    add = 0.090 + offset
                    drop = 0.130 + offset
                    drop_fallback = 0.122 + offset
                else:
                    add = (
                        0.105 + offset
                        if window == "dev_a"
                        else 0.099 + offset
                    )
                    drop = (
                        0.119 + offset
                        if window == "dev_a"
                        else 0.122 + offset
                    )
                    drop_fallback = 0.121 + offset
                orderings[f"add_{group}"] = _ordering(add)
                orderings[f"drop_{group}"] = _ordering(drop)
                orderings[f"drop_{group}__inactive_rrf"] = _ordering(
                    drop_fallback
                )
            development[window] = {"evaluation": {"orderings": orderings}}
        result = analyze_groups(development, metric_k=12, min_mean_gain=0.0002)
        self.assertEqual(
            result["groups"]["customer_static"]["classification"],
            "stable_helpful",
        )
        self.assertEqual(
            result["groups"]["user_history"]["classification"],
            "stable_harmful",
        )
        self.assertEqual(
            result["groups"]["item_trend"]["classification"],
            "interaction_or_unstable",
        )
        self.assertTrue(
            result["selection_candidates"][
                "drop_user_history__inactive_rrf"
            ]["passes"]
        )
        self.assertEqual(
            result["selected_development_candidate"],
            "drop_user_history__inactive_rrf",
        )
        self.assertEqual(result["final_confirmation_run"], "not_run")

    def test_bound_file_reuse_checks_root_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            root.mkdir()
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            import hashlib

            evidence = {
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha": hashlib.sha256(b"evidence").hexdigest(),
            }
            resolved = _require_bound_file(
                evidence, "path", "bytes", "sha", root
            )
            self.assertEqual(resolved, artifact.resolve())
            evidence["bytes"] += 1
            with self.assertRaisesRegex(ValueError, "bytes mismatch"):
                _require_bound_file(evidence, "path", "bytes", "sha", root)
            outside = Path(temp) / "outside.bin"
            outside.write_bytes(b"evidence")
            evidence.update(
                {
                    "path": str(outside),
                    "bytes": outside.stat().st_size,
                }
            )
            with self.assertRaisesRegex(ValueError, "escapes M2.1 root"):
                _require_bound_file(evidence, "path", "bytes", "sha", root)


if __name__ == "__main__":
    unittest.main()
