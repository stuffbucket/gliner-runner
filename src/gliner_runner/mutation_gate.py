from __future__ import annotations

import re
from collections import Counter

MINIMUM_DETECTED = 677
MAXIMUM_SURVIVED = 103
VERDICT = re.compile(r": (killed|survived|suspicious|timeout)$")


def parse_verdicts(output: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in output.splitlines()
        if (match := VERDICT.search(line.strip())) is not None
    )


def validate_verdicts(verdicts: Counter[str]) -> list[str]:
    errors: list[str] = []
    detected = verdicts["killed"] + verdicts["timeout"]
    if detected < MINIMUM_DETECTED:
        errors.append(
            f"detected mutants regressed: {detected} < {MINIMUM_DETECTED}"
        )
    if verdicts["survived"] > MAXIMUM_SURVIVED:
        errors.append(
            f"surviving mutants regressed: {verdicts['survived']} > {MAXIMUM_SURVIVED}"
        )
    if verdicts["suspicious"]:
        errors.append(f"suspicious mutants remain: {verdicts['suspicious']}")
    return errors
