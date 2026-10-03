from __future__ import annotations

from email.message import Message
from pathlib import PurePosixPath
from typing import cast

import pytest
from scripts.generate_third_party_inventory import (
    generate_inventory,
    parse_license_overrides,
)


class FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        *,
        license_expression: str = "",
        files: tuple[str, ...] = (),
    ) -> None:
        metadata = Message()
        metadata["Name"] = name
        if license_expression:
            metadata["License-Expression"] = license_expression
        self.metadata = metadata
        self.version = version
        self.files = tuple(PurePosixPath(file) for file in files)


def distributions(*items: FakeDistribution) -> list[object]:
    return list(items)


def test_inventory_records_exact_versions_licenses_and_files() -> None:
    installed = distributions(
        FakeDistribution(
            "gliner2",
            "2.0.0",
            license_expression="Apache-2.0",
            files=("gliner2-2.0.0.dist-info/licenses/LICENSE",),
        ),
        FakeDistribution(
            "torch",
            "2.14.0",
            license_expression="BSD-3-Clause AND MIT",
            files=(
                "torch-2.14.0.dist-info/licenses/LICENSE",
                "torch-2.14.0.dist-info/licenses/NOTICE",
            ),
        ),
    )

    result = generate_inventory(
        cast("list", installed),
        excluded=frozenset(),
        license_overrides={},
        required=("gliner2==2.0.0", "torch"),
    )

    assert "| `gliner2` | `2.0.0` | Apache-2.0 |" in result
    assert "| `torch` | `2.14.0` | BSD-3-Clause AND MIT |" in result
    assert "`torch-2.14.0.dist-info/licenses/NOTICE`" in result


def test_inventory_rejects_missing_license_evidence() -> None:
    installed = distributions(FakeDistribution("mystery", "1.0.0"))

    with pytest.raises(ValueError, match="no license metadata"):
        generate_inventory(
            cast("list", installed),
            excluded=frozenset(),
            license_overrides={},
            required=(),
        )


def test_inventory_rejects_missing_or_changed_required_distribution() -> None:
    installed = distributions(
        FakeDistribution("gliner2", "2.0.1", license_expression="Apache-2.0")
    )

    with pytest.raises(ValueError, match=r"required gliner2==2\.0\.0, found 2\.0\.1"):
        generate_inventory(
            cast("list", installed),
            excluded=frozenset(),
            license_overrides={},
            required=("gliner2==2.0.0", "torch"),
        )


def test_inventory_records_explicit_license_override() -> None:
    installed = distributions(FakeDistribution("cuda-toolkit", "13.0.3.0"))

    result = generate_inventory(
        cast("list", installed),
        excluded=frozenset(),
        license_overrides=parse_license_overrides(
            ("cuda-toolkit=NVIDIA CUDA Toolkit EULA (metadata override)",)
        ),
        required=(),
    )

    assert "NVIDIA CUDA Toolkit EULA (metadata override)" in result


@pytest.mark.parametrize(
    "value",
    ("cuda-toolkit", "=license", "cuda-toolkit="),
)
def test_license_override_rejects_invalid_value(value: str) -> None:
    with pytest.raises(ValueError, match="invalid license override"):
        parse_license_overrides((value,))
