from __future__ import annotations

import unittest

from hm_recsys.m39 import FINAL_WEEK_CUTOFF, VARIANTS, _rank_bucket_sql, validate_protocol


class M39ProtocolTests(unittest.TestCase):
    def test_protocol_is_read_only_and_excludes_final_week(self) -> None:
        validate_protocol()
        self.assertEqual(FINAL_WEEK_CUTOFF, "2020-09-16")
        self.assertEqual(len(VARIANTS), 3)

    def test_rank_buckets_are_frozen_and_disjoint(self) -> None:
        expression = _rank_bucket_sql("source_rank")
        self.assertIn("BETWEEN 1 AND 10", expression)
        self.assertIn("BETWEEN 11 AND 50", expression)
        self.assertIn("BETWEEN 51 AND 100", expression)


if __name__ == "__main__":
    unittest.main()
