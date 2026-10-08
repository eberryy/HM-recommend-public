from __future__ import annotations

import argparse
from pathlib import Path

from .m53 import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M5.3 Warm shortlist retention audit")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5_3"))
    args = parser.parse_args(argv)
    run(source_root=args.source_root.resolve(), report_dir=args.report_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
