from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from hm_recsys.p34_regime_audit import (
    AuditConfig,
    _build_transition_tables,
    _drop_transition_tables,
    _global_concentration,
    _preference_drift,
    _top_overlap_and_rank_correlation,
    _transition_audit,
    distribution_concentration,
    jensen_shannon_from_counts,
)


class P34RegimeAuditTests(unittest.TestCase):
    def test_concentration_has_auditable_denominator(self) -> None:
        result = distribution_concentration([6, 3, 1], (1, 2))
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["event_count"], 10)
        self.assertAlmostEqual(result["top_shares"]["1"], 0.6)
        self.assertAlmostEqual(result["hhi"], 0.46)

    def test_jsd_bounds_and_identity(self) -> None:
        self.assertAlmostEqual(jensen_shannon_from_counts([1, 1], [1, 1]), 0.0)
        self.assertAlmostEqual(jensen_shannon_from_counts([1, 0], [0, 1]), 1.0)

    def test_user_day_basket_does_not_create_same_day_order(self) -> None:
        connection = duckdb.connect(":memory:")
        connection.execute(
            """
            CREATE TABLE transactions(t_dat DATE, customer_id VARCHAR, article_id BIGINT);
            INSERT INTO transactions VALUES
                ('2020-01-01','u1',1),
                ('2020-01-01','u1',1),
                ('2020-01-01','u1',2),
                ('2020-01-03','u1',3),
                ('2020-01-03','u1',4),
                ('2020-01-04','u2',5);
            CREATE TABLE articles(
                article_id BIGINT,
                product_type_no INTEGER,
                garment_group_no INTEGER,
                department_no INTEGER,
                index_group_no INTEGER,
                colour_group_code INTEGER
            );
            INSERT INTO articles VALUES
                (1,10,1,1,1,1),(2,10,1,1,1,1),(3,20,2,2,2,2),
                (4,30,3,3,3,3),(5,40,4,4,4,4);
            """
        )
        boundary = _build_transition_tables(connection, "2020-01-05")
        self.assertEqual(boundary, "2019-11-24 00:00:00")
        self.assertEqual(connection.execute("SELECT count(*) FROM p34_baskets").fetchone()[0], 3)
        self.assertEqual(
            connection.execute("SELECT count(*) FROM p34_basket_transitions").fetchone()[0], 1
        )
        self.assertEqual(connection.execute("SELECT count(*) FROM p34_item_rows").fetchone()[0], 4)
        self.assertAlmostEqual(
            connection.execute("SELECT sum(weight) FROM p34_item_rows").fetchone()[0], 1.0
        )
        self.assertEqual(
            connection.execute(
                "SELECT count(*) FROM p34_item_rows WHERE source_article_id=1 AND target_article_id=2"
            ).fetchone()[0],
            0,
        )
        _drop_transition_tables(connection)
        connection.close()

    def test_top_overlap_reports_real_denominator(self) -> None:
        connection = duckdb.connect(":memory:")
        connection.execute("CREATE TABLE left_edges(a INTEGER,b INTEGER,weight DOUBLE)")
        connection.execute("CREATE TABLE right_edges(a INTEGER,b INTEGER,weight DOUBLE)")
        connection.execute("INSERT INTO left_edges VALUES (1,2,3),(2,3,2),(3,4,1)")
        connection.execute("INSERT INTO right_edges VALUES (1,2,4),(4,5,2),(3,4,1)")
        result = _top_overlap_and_rank_correlation(
            connection, "left_edges", "right_edges", ("a", "b"), 3
        )
        self.assertEqual(result["shared_edges"], 2)
        self.assertEqual(result["overlap_denominator"], 3)
        self.assertAlmostEqual(result["overlap_share"], 2 / 3)
        connection.close()

    def test_component_queries_run_on_synthetic_history(self) -> None:
        connection = duckdb.connect(":memory:")
        connection.execute(
            """
            CREATE TABLE transactions(t_dat DATE, customer_id VARCHAR, article_id BIGINT);
            INSERT INTO transactions VALUES
                ('2019-11-01','u1',1),('2019-11-08','u1',2),
                ('2019-12-28','u1',3),('2020-01-05','u1',4),('2020-01-12','u1',1),
                ('2019-11-02','u2',2),('2019-11-09','u2',3),
                ('2019-12-29','u2',4),('2020-01-06','u2',1),('2020-01-13','u2',2);
            CREATE TABLE articles(
                article_id BIGINT,
                product_type_no INTEGER,
                garment_group_no INTEGER,
                department_no INTEGER,
                index_group_no INTEGER,
                colour_group_code INTEGER
            );
            INSERT INTO articles VALUES
                (1,10,1,1,1,1),(2,10,1,1,1,2),
                (3,20,2,2,2,1),(4,30,3,3,3,3);
            """
        )
        concentration = _global_concentration(connection, "2020-01-22")
        self.assertEqual(concentration["84d"]["article"]["event_count"], 10)
        preference = _preference_drift(
            connection, "2020-01-22", "product_type_no", min_events=2, high_drift_threshold=0.5
        )
        self.assertEqual(preference["eligibility"]["eligible_users"], 2)
        with TemporaryDirectory() as directory:
            config = AuditConfig(
                raw_dir=Path(directory),
                artifact_dir=Path(directory),
                output_json=Path(directory) / "metrics.json",
                output_md=Path(directory) / "report.md",
            )
            transition = _transition_audit(connection, "2020-01-22", config)
        self.assertEqual(transition["basket_transitions"]["count"], 8)
        self.assertEqual(transition["cross_user_agreement"]["eligible_users"], 2)
        self.assertIsNotNone(transition["turnover"]["category_edge_jsd"])
        _drop_transition_tables(connection)
        connection.close()


if __name__ == "__main__":
    unittest.main()
