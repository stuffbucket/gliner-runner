from __future__ import annotations

import json
from pathlib import Path

from gliner_runner.contracts import InferenceRequest


def test_parity_fixture_contracts_are_valid() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "parity"
    fixtures = [json.loads(path.read_text()) for path in fixture_dir.glob("*.json")]

    assert fixtures
    for fixture in fixtures:
        InferenceRequest.model_validate(fixture["request"])
        assert fixture["oracle_output"] is not None
