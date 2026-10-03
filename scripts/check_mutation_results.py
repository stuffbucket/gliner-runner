from __future__ import annotations

import subprocess
import sys

from gliner_runner.mutation_gate import parse_verdicts, validate_verdicts


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "mutmut", "results", "--all", "true"],
        check=True,
        capture_output=True,
        text=True,
    )
    verdicts = parse_verdicts(completed.stdout)
    errors = validate_verdicts(verdicts)
    summary = ", ".join(
        f"{name}={verdicts[name]}"
        for name in ("killed", "survived", "timeout", "suspicious")
    )
    if errors:
        print(f"mutation quality gate failed ({summary})", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(completed.stdout, file=sys.stderr, end="")
        return 1
    print(f"mutation quality gate: ok ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
