import unittest

from hm_recsys.cli import build_parser


class ReportPathTests(unittest.TestCase):
    def test_data_audit_defaults_to_project_reports(self):
        args = build_parser().parse_args(["audit", "data"])
        self.assertEqual(args.output, "reports/m0_5/HM_DATA_AUDIT.md")

    def test_explicit_output_is_still_supported(self):
        args = build_parser().parse_args(["audit", "data", "--output", "reports/custom/audit.md"])
        self.assertEqual(args.output, "reports/custom/audit.md")


if __name__ == "__main__":
    unittest.main()
