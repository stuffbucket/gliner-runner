from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gliner_runner import model_inventory
from gliner_runner.contracts import Precision, PrecisionProfile
from gliner_runner.errors import ModelDownloadRequiredError
from gliner_runner.model_inventory import (
    KNOWN_MODELS,
    DownloadJobManager,
    DownloadRequest,
    DownloadState,
    ModelInventory,
)


def test_curated_model_catalog_has_exact_release_identity() -> None:
    assert {spec.model_id: (spec.repository, spec.pinned_revision) for spec in KNOWN_MODELS} == {
        "decide-340m": (
            "fastino/GLiNER2.5-Decide",
            "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
        ),
        "decide-multilingual": (
            "fastino/GLiNER2.5-multi-Decide",
            "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
        ),
        "decide-1b": (
            "fastino/GLiNER2.5-Decide-1B",
            "67af949e17bebd20f05fe020a971079889e7a462",
        ),
    }


def test_profile_guidance_uses_reviewed_current_revision_evidence() -> None:
    spec = next(item for item in KNOWN_MODELS if item.model_id == "decide-340m")
    expected = {
        ("cpu", Precision.FP32): 3.5762786865234375e-7,
        ("mps", Precision.FP32): 3.814697265625e-6,
        ("mps", Precision.FP16): 0.0010069608688354492,
    }

    for (device, precision), drift in expected.items():
        guidance = model_inventory.profile_guidance(
            spec,
            PrecisionProfile(device=device, precision=precision),
        )
        assert guidance.observed_max_abs_score_drift == drift
        assert "current-revision" in guidance.characterization_scope

    multilingual = next(item for item in KNOWN_MODELS if item.model_id == "decide-multilingual")
    multilingual_expected = {
        ("cpu", Precision.FP32): 0.0,
        ("mps", Precision.FP32): 2.205371856689453e-6,
        ("mps", Precision.FP16): 0.0053369998931884766,
    }
    for (device, precision), drift in multilingual_expected.items():
        guidance = model_inventory.profile_guidance(
            multilingual,
            PrecisionProfile(device=device, precision=precision),
        )
        assert guidance.observed_max_abs_score_drift == drift
        assert "multilingual-v1" in guidance.characterization_scope

    one_billion = next(item for item in KNOWN_MODELS if item.model_id == "decide-1b")
    one_billion_expected = {
        ("cpu", Precision.FP32): 4.76837158203125e-7,
        ("mps", Precision.FP16): 0.0015290528535842896,
        ("mps", Precision.FP32): 7.748603820800781e-7,
    }
    for (device, precision), drift in one_billion_expected.items():
        guidance = model_inventory.profile_guidance(
            one_billion,
            PrecisionProfile(device=device, precision=precision),
        )
        assert guidance.observed_max_abs_score_drift == drift
        assert "english-v1" in guidance.characterization_scope


def test_inventory_requires_explicit_download_for_missing_model(tmp_path: Path) -> None:
    inventory = ModelInventory(tmp_path)
    spec = KNOWN_MODELS[0]

    with pytest.raises(ModelDownloadRequiredError) as captured:
        inventory.resolve(spec.model_id)

    assert captured.value.revision == spec.pinned_revision
    assert captured.value.destination == str(tmp_path.resolve())


def test_inventory_resolves_only_completed_exact_revision(tmp_path: Path) -> None:
    inventory = ModelInventory(tmp_path)
    spec = KNOWN_MODELS[0]
    target = inventory.expected_path(spec, spec.pinned_revision)
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"model")
    (target / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "model_id": spec.model_id,
                "repository": spec.repository,
                "revision": spec.pinned_revision,
                "files": [
                    {
                        "path": "model.safetensors",
                        "size": 5,
                        "sha256": hashlib.sha256(b"model").hexdigest(),
                    }
                ],
            }
        )
    )

    resolved = inventory.resolve(spec.model_id)

    assert resolved.path == target
    assert inventory.local_revisions(spec) == (spec.pinned_revision,)


def test_inventory_rejects_artifact_paths_outside_snapshot(tmp_path: Path) -> None:
    inventory = ModelInventory(tmp_path)
    spec = KNOWN_MODELS[0]
    target = inventory.expected_path(spec, spec.pinned_revision)
    target.mkdir(parents=True)
    (target / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "repository": spec.repository,
                "revision": spec.pinned_revision,
                "files": [{"path": "../outside", "size": 1, "sha256": "0" * 64}],
            }
        )
    )

    with pytest.raises(ModelDownloadRequiredError):
        inventory.resolve(spec.model_id)


async def test_download_job_requires_approval_and_reports_completion(tmp_path: Path) -> None:
    inventory = ModelInventory(tmp_path)
    inventory.download = AsyncMock(return_value=tmp_path)  # type: ignore[method-assign]
    manager = DownloadJobManager(inventory)
    request = DownloadRequest(approved=False, destination=str(tmp_path))

    with pytest.raises(ValueError, match="approved must be true"):
        manager.create("decide-340m", request)

    job = manager.create(
        "decide-340m",
        request.model_copy(update={"approved": True}),
    )
    await asyncio.sleep(0)
    await manager.close()

    assert manager.get(job.job_id).state is DownloadState.SUCCEEDED


async def test_refresh_caches_revisions_and_uncurated_official_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeApi:
        def model_info(self, repository: str) -> SimpleNamespace:
            revisions = {
                spec.repository: f"{index:040x}" for index, spec in enumerate(KNOWN_MODELS, start=1)
            }
            return SimpleNamespace(sha=revisions[repository])

        def list_models(self, **_kwargs: object) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(id=KNOWN_MODELS[0].repository),
                SimpleNamespace(id="fastino/GLiNER2.5-New-Decide"),
                SimpleNamespace(id="fastino/unrelated"),
            ]

    monkeypatch.setattr(
        model_inventory.importlib,
        "import_module",
        lambda _name: SimpleNamespace(HfApi=FakeApi),
    )
    inventory = ModelInventory(tmp_path)

    result = await inventory.refresh()
    reloaded = ModelInventory(tmp_path)

    assert result.models_checked == 3
    assert result.discovered_models == ("fastino/GLiNER2.5-New-Decide",)
    assert reloaded.discovered_models() == result.discovered_models
    assert reloaded.discovered_revision(KNOWN_MODELS[0]) == f"{1:040x}"
