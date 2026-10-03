from __future__ import annotations

from collections import Counter

from gliner_runner.mutation_gate import (
    MAXIMUM_SURVIVED,
    MINIMUM_DETECTED,
    parse_verdicts,
    validate_verdicts,
)


def test_mutation_verdict_parser_and_baseline() -> None:
    output = "\n".join(
        [
            "module.x__mutmut_1: killed",
            "module.x__mutmut_2: survived",
            "module.x__mutmut_3: timeout",
            "module.x__mutmut_4: suspicious",
        ]
    )

    assert parse_verdicts(output) == Counter(
        {"killed": 1, "survived": 1, "timeout": 1, "suspicious": 1}
    )
    assert (
        validate_verdicts(
            Counter(
                {
                    "killed": MINIMUM_DETECTED,
                    "survived": MAXIMUM_SURVIVED,
                }
            )
        )
        == []
    )


def test_mutation_gate_rejects_each_regression() -> None:
    errors = validate_verdicts(
        Counter(
            {
                "killed": MINIMUM_DETECTED - 1,
                "survived": MAXIMUM_SURVIVED + 1,
                "suspicious": 1,
            }
        )
    )

    assert len(errors) == 3
