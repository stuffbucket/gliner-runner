"""Check the GitHub Actions workflows against the repository's pinning and timeout policy.

- Every external action is pinned to a full commit SHA, with the version it is in a trailing
  comment (`uses: owner/repo@<40 hex> # v1.2.3`), so the code that runs cannot change under a tag
  and Dependabot can still name and update the version. Calendar versions (`# 2026.10.26.1`)
  count; branch names (`# main`) do not.
- Every job that runs steps sets `timeout-minutes`, so a hung step cannot hold a runner for the
  six-hour default. Jobs that call a reusable workflow (`uses:` at job level) cannot set one;
  the jobs in the called workflow are checked instead.

The workflows are read line by line, as GitHub writes them here: two-space indents, one job key
per line.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$")
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
VERSION_COMMENT = re.compile(r"^\s+#\s+v?\d+(?:\.\d+)*(?:[-+][\w.]+)?\s*$")
JOB = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")


def check_text(name: str, text: str) -> list[str]:
    errors: list[str] = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        match = USES.match(line)
        if not match:
            continue
        target, rest = match.groups()
        if target.startswith("./"):
            continue
        if not PINNED.match(target):
            errors.append(f"{name}:{number}: {target} is not pinned to a full commit SHA")
        elif not VERSION_COMMENT.match(rest):
            errors.append(f"{name}:{number}: {target} needs a version comment, such as '# v1.2.3'")

    in_jobs = False
    job: str | None = None
    job_line = 0
    has_timeout = calls_workflow = False

    def finish() -> None:
        if job is not None and not has_timeout and not calls_workflow:
            errors.append(f"{name}:{job_line}: job {job} has no timeout-minutes")

    for number, line in enumerate(lines, start=1):
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line and not line.startswith(" ") and not line.startswith("#"):
            finish()
            in_jobs, job = False, None
            continue
        match = JOB.match(line)
        if match:
            finish()
            job, job_line = match.group(1), number
            has_timeout = calls_workflow = False
        elif job is not None and line.startswith("    timeout-minutes:"):
            has_timeout = True
        elif job is not None and line.startswith("    uses:"):
            calls_workflow = True
    finish()
    return errors


def check(directory: Path = WORKFLOWS) -> list[str]:
    errors: list[str] = []
    for path in sorted(directory.glob("*.y*ml")):
        errors.extend(check_text(path.name, path.read_text(encoding="utf-8")))
    return errors


def main() -> None:
    errors = check()
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        sys.exit(1)
    print("workflows: every action is pinned with a version comment, every job has a timeout")


if __name__ == "__main__":
    main()
