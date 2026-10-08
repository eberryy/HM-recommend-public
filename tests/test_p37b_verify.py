from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from hm_recsys.p37b_verify import (
    _candidate_identity_valid,
    _feature_lineage_manifests_valid,
    _initialization_pair_valid,
    _inner_trace_valid,
    _p33_auxiliary_audit_valid,
    _training_audit_valid,
)


WINDOWS = (
    "winter_20200122",
    "spring_20200318",
    "early_summer_20200624",
    "late_summer_20200819",
)


def _variant() -> dict:
    return {
        "at_k": {
            "20": {"positive_density": 0.2},
            "200": {
                "candidate_rows": 200,
                "positive_rows": 2,
                "segments": {
                    "strict_cold": {"recall": 0.1},
                    "sparse_1_5": {"recall": 0.2},
                },
            },
        },
        "ranking": {
            "mrr": 0.3,
            "coarse_positive_pairs_top200": 2,
            "conversion": {"top200_to_top20": 0.4},
        },
    }


def _identity_row() -> dict:
    variants = {
        variant: {
            "candidate_identity_sha256": "a" * 64,
            "candidate_rows": 200,
            "per_user_rank_permutation_1_200": True,
            "top200_truth_pairs": 2,
            "strict_recall200": 0.1,
            "sparse_recall200": 0.2,
        }
        for variant in ("C0", "C1", "B0", "B1")
    }
    return {
        "variants": variants,
        "all_variants_exact": True,
        "candidate_identity_exact": True,
        "candidate_rows_exact": True,
        "recall200_exact": True,
        "top200_truth_pairs_exact": True,
        "m4_cutoff_safe": True,
        "p33_cutoff_safe": True,
    }


def _training_variant() -> dict:
    pair_audit = {
        "pair_array_sha256": "b" * 64,
        "same_user_only": True,
        "all_buckets_covered": True,
        "maximum_negatives_per_positive": 50,
        "bucket_counts": {"hard": 1, "medium": 1, "easy": 1},
        "pair_rows": 3,
    }
    residual = {
        "rows": 3,
        "delta_max_abs": 0.0,
        "score_vs_raw_m4_max_abs": 0.0,
        "last_layer_weight_zero": True,
        "last_layer_bias_zero": True,
    }
    trace_row = {
        "epoch": 1,
        "bpr_loss": 1.0,
        "pair_accuracy": 0.5,
        "pair_rows": 3,
        "inner_density20": 0.2,
        "inner_mrr": 0.3,
        "inner_top200_to_top20": 0.4,
        "scoring_seconds": 0.1,
    }
    return {
        "inner": {
            "training_cutoff": "2019-11-27",
            "selected_epoch": 1,
            "selected_key": {"density20": 0.2, "mrr": 0.3},
            "max_epochs": 1,
            "patience_consecutive_nonimprovements": 3,
            "trace": [trace_row],
            "pair_audit": pair_audit,
            "initial_residual": residual,
        },
        "outer": {
            "epochs_from_inner_early_stopping": 1,
            "trace": [{"epoch": 1, "pair_rows": 3}],
            "pair_audits": {"2019-11-27": pair_audit},
            "initial_residual": residual,
        },
    }
class P37BVerifierTests(unittest.TestCase):
    def test_feature_lineage_requires_twelve_cutoff_safe_bound_manifests(self) -> None:
        roles_and_cutoffs = [
            *(('training', cutoff) for cutoff in (
                '2019-11-27', '2019-12-25', '2020-01-22', '2020-02-19',
                '2020-04-29', '2020-05-27', '2020-06-24', '2020-07-22',
            )),
            *(('outer_validation', cutoff) for cutoff in (
                '2020-01-22', '2020-03-18', '2020-06-24', '2020-08-19',
            )),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'features-v1'
            rows = []
            first_manifest = None
            for index, (role, cutoff) in enumerate(roles_and_cutoffs):
                directory = root / role / cutoff
                directory.mkdir(parents=True)

                def identity(name: str) -> dict:
                    return {
                        'path': str(directory / name),
                        'bytes': index + 1,
                        'sha256': f'{index + 1:064x}',
                    }

                sources = {
                    'candidates': identity('candidates.npz'),
                    'histories': identity('histories.npz'),
                    'users': identity('users.csv'),
                    'm4_embeddings': identity('m4.npy'),
                    'p33_embeddings': identity('p33.npy'),
                    'p33_manifest': identity('p33_manifest.json'),
                }
                artifacts = {
                    name: identity(f'{name}.npy')
                    for name in (
                        'm4_relation', 'p33_relation', 'p33_global',
                        'user_state', 'candidate_state',
                    )
                }
                payload = {
                    'schema_version': 'phase3-p3.7b-raw-features-v1',
                    'status': 'completed',
                    'cutoff': cutoff,
                    'final_week': 'not_run',
                    'source_artifacts': sources,
                    'artifacts': artifacts,
                    'lineage_audit': {
                        'candidates': {'observed': sources['candidates'], 'passed': True},
                        'histories': {'observed': sources['histories'], 'passed': True},
                        'users': {'observed': sources['users'], 'passed': True},
                        'm4_embedding': {'observed': sources['m4_embeddings'], 'passed': True},
                        'p33_embedding': {
                            'observed': sources['p33_embeddings'],
                            'manifest': sources['p33_manifest'],
                            'declared_training_cutoffs': [cutoff],
                            'content_only_inference': True,
                            'passed': True,
                        },
                        'history_reconstruction': {
                            'catalog_row_exact': True,
                            'days_exact': True,
                            'mask_exact': True,
                            'latest_behavior_before_cutoff': True,
                        },
                        'cutoff': {
                            'cutoff': cutoff,
                            'cutoff_safe': True,
                            'latest_behavior_before_cutoff': '2019-01-01',
                        },
                    },
                }
                manifest_path = directory / 'manifest.json'
                manifest_path.write_text(json.dumps(payload), encoding='utf-8')
                rows.append({
                    'path': str(manifest_path),
                    'bytes': manifest_path.stat().st_size,
                    'sha256': 'f' * 64,
                    'cutoff_identity': cutoff,
                })
                rows.extend({**value, 'cutoff_identity': cutoff} for value in artifacts.values())
                first_manifest = first_manifest or manifest_path
            self.assertTrue(_feature_lineage_manifests_valid(rows))
            broken = json.loads(first_manifest.read_text(encoding='utf-8'))
            broken['lineage_audit']['cutoff']['cutoff_safe'] = False
            first_manifest.write_text(json.dumps(broken), encoding='utf-8')
            self.assertFalse(_feature_lineage_manifests_valid(rows))

    def test_candidate_identity_requires_persisted_safety_and_metric_parity(self) -> None:
        rows = {window: _identity_row() for window in WINDOWS}
        metrics = {
            "windows": {
                window: {
                    "variants": {
                        variant: _variant() for variant in ("C0", "C1", "B0", "B1")
                    },
                    "candidate_identity": rows[window],
                }
                for window in WINDOWS
            }
        }
        audit = {
            "status": "measured",
            "windows": rows,
            "all_windows_exact": True,
            "final_week": "not_run",
        }
        self.assertTrue(_candidate_identity_valid(metrics, audit))
        broken = copy.deepcopy(audit)
        broken["windows"][WINDOWS[0]]["p33_cutoff_safe"] = False
        metrics["windows"][WINDOWS[0]]["candidate_identity"] = broken["windows"][WINDOWS[0]]
        self.assertFalse(_candidate_identity_valid(metrics, broken))

    def test_inner_trace_recomputes_lexicographic_selection_and_patience(self) -> None:
        pair_audit = {
            "pair_array_sha256": "b" * 64,
            "same_user_only": True,
            "all_buckets_covered": True,
            "maximum_negatives_per_positive": 50,
            "bucket_counts": {"hard": 1, "medium": 1, "easy": 1},
            "pair_rows": 3,
        }
        trace = [
            {
                "epoch": epoch,
                "bpr_loss": 1.0,
                "pair_accuracy": 0.5,
                "pair_rows": 3,
                "inner_density20": 0.2 if epoch == 1 else 0.1,
                "inner_mrr": 0.3,
                "inner_top200_to_top20": 0.4,
                "scoring_seconds": 0.1,
            }
            for epoch in range(1, 5)
        ]
        inner = {
            "selected_epoch": 1,
            "selected_key": {"density20": 0.2, "mrr": 0.3},
            "max_epochs": 10,
            "patience_consecutive_nonimprovements": 3,
            "trace": trace,
            "pair_audit": pair_audit,
        }
        self.assertTrue(_inner_trace_valid(inner))
        broken = copy.deepcopy(inner)
        broken["selected_epoch"] = 2
        self.assertFalse(_inner_trace_valid(broken))

    def test_paired_initialization_checks_common_and_downstream_hashes(self) -> None:
        b0 = {
            "scheme": "paired-layerwise-v1",
            "seed": 37,
            "common_relation_input_sha256": "c" * 64,
            "shared_downstream_sha256": "d" * 64,
            "auxiliary_relation_input_sha256": None,
        }
        b1 = {**b0, "auxiliary_relation_input_sha256": "e" * 64}
        self.assertTrue(_initialization_pair_valid(b0, b1))
        b1["shared_downstream_sha256"] = "f" * 64
        self.assertFalse(_initialization_pair_valid(b0, b1))
        self.assertTrue(_initialization_pair_valid(None, None))

    def test_training_audit_requires_same_b0_b1_pair_sampling_identity(self) -> None:
        training = {
            "status": "measured",
            "windows": {
                window: {"B0": _training_variant(), "B1": _training_variant()}
                for window in WINDOWS
            },
            "final_week": "not_run",
        }
        self.assertEqual(_training_audit_valid(training), (True, True, True))
        training["windows"][WINDOWS[0]]["B1"]["inner"]["pair_audit"][
            "pair_array_sha256"
        ] = "f" * 64
        self.assertEqual(_training_audit_valid(training), (True, False, True))

    def test_p33_masked_audit_checks_arithmetic_and_full_metric_parity(self) -> None:
        metrics = {
            "windows": {window: {"variants": {"B1": _variant()}} for window in WINDOWS}
        }
        row = {
            "definition": "mask auxiliary fields",
            "B1_full": {"density20": 0.2, "mrr": 0.3, "top200_to_top20": 0.4},
            "B1_p33_auxiliary_masked": {
                "density20": 0.1,
                "mrr": 0.2,
                "top200_to_top20": 0.3,
            },
            "full_minus_masked": {
                "density20": 0.1,
                "mrr": 0.1,
                "top200_to_top20": 0.1,
            },
            "scoring": {
                "rows": 200,
                "elapsed_seconds": 1.0,
                "mask_p33_auxiliary": True,
            },
            "promotion_metric": False,
        }
        audit = {
            "status": "measured",
            "windows": {window: copy.deepcopy(row) for window in WINDOWS},
            "promotion_metric": False,
            "final_week": "not_run",
        }
        self.assertTrue(_p33_auxiliary_audit_valid(metrics, audit))
        audit["windows"][WINDOWS[0]]["full_minus_masked"]["mrr"] = 0.2
        self.assertFalse(_p33_auxiliary_audit_valid(metrics, audit))


if __name__ == "__main__":
    unittest.main()
