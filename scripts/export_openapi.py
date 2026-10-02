from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gliner_runner.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the canonical OpenAPI contract")
    parser.add_argument("output", type=Path, nargs="?", default=Path("openapi/openapi.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero instead of writing when the output is stale.",
    )
    args = parser.parse_args()

    document = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != document:
            print(
                f"{args.output} is stale; run 'python scripts/export_openapi.py'",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
