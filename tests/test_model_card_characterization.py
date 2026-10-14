from __future__ import annotations

import pytest
from benchmarks.characterize_model_card import (
    discovered_example_count,
    generate_report,
    parse_examples,
)


def example_section(
    *,
    name: str = "Example",
    call: str | None = None,
    potential_output: str = '{"intent": "billing"}',
) -> str:
    source = (
        call
        or """model.classify_text(
    "I was charged twice.",
    {"intent": ["billing", "support"]},
)"""
    )
    return f"""### {name}

A literal classification example.

```python
{source}
```

Potential output:

```text
{potential_output}
```
"""


def test_parse_examples_accounts_for_every_discovered_call() -> None:
    card = example_section(name="Billing") + example_section(name="Support")

    examples = parse_examples(card)

    assert discovered_example_count(card) == 2
    assert [example["name"] for example in examples] == ["Billing", "Support"]
    assert examples[0]["text"] == "I was charged twice."
    assert examples[0]["schema"] == {"intent": ["billing", "support"]}


def test_parse_examples_rejects_unmatched_classify_text_call() -> None:
    card = """### Incomplete

```python
model.classify_text(
    "hello",
    {"intent": ["billing"]},
)
```
"""

    with pytest.raises(ValueError, match=r"discovered 1, parsed 0"):
        parse_examples(card)


def test_parse_examples_rejects_nonliteral_or_wrong_argument_shapes() -> None:
    nonliteral = example_section(
        call="""model.classify_text(
    text_from_network,
    {"intent": ["billing"]},
)"""
    )
    with pytest.raises((ValueError, TypeError)):
        parse_examples(nonliteral)

    wrong_count = example_section(
        call="""model.classify_text(
    "hello",
)"""
    )
    with pytest.raises(ValueError, match="expected text and schema arguments"):
        parse_examples(wrong_count)


def test_parse_examples_rejects_duplicate_headings_and_ignores_prose() -> None:
    card = (
        "The model has a classify_text method, but prose is not executable.\n\n"
        + example_section(name="Duplicate")
        + example_section(name="Duplicate")
    )

    assert discovered_example_count(card) == 2
    with pytest.raises(ValueError, match="duplicate example heading"):
        parse_examples(card)


def test_report_names_the_actual_first_profile_as_baseline() -> None:
    report = generate_report(
        {
            "model_id": "owner/model",
            "model_revision": "a" * 40,
            "hardware": "test",
            "os": "test",
            "versions": {"python": "3.12", "torch": "2", "gliner2": "2.1"},
            "model_card_sha256": "b" * 64,
            "discovered_example_count": 1,
            "example_count": 1,
            "all_discovered_examples_parsed": True,
            "profiles": [
                {
                    "device": "mps",
                    "precision": "fp16",
                    "load_ms": 1.0,
                    "successful_examples": 1,
                    "potential_output_matches": 1,
                    "examples": [
                        {
                            "name": "Example",
                            "actual_output": {"intent": "billing"},
                            "matches_potential_output": True,
                        }
                    ],
                }
            ],
            "comparisons": [],
        }
    )

    assert "MPS/FP16 is the baseline." in report
    assert "## MPS/FP16 outputs" in report
    assert "CPU/FP32 is the baseline." not in report
