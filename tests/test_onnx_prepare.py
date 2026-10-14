from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gliner_runner import cli, onnx_prepare
from gliner_runner.model_inventory import KNOWN_MODELS, InventoryModelSpec
from gliner_runner.onnx_prepare import (
    DEFAULT_ATOL,
    GRAPH_FILE,
    MANIFEST_FILE,
    PREPROCESSING_FILE,
    SPECIAL_TOKENS,
    artifact_directory,
    build_manifest,
    build_preprocessing_config,
    compare_task_logits,
    file_inventory,
    fixture_batches,
    fixture_sha256,
    prepare_onnx,
    resolve_curated_model,
    special_token_ids_from_tokenizer_json,
    validate_atol,
    validate_external_data_inventory,
    validate_manifest,
    validate_output_location,
    verify_artifact_directory,
)
from gliner_runner.snapshot import file_sha256

SMALL = KNOWN_MODELS[0]
MULTILINGUAL = KNOWN_MODELS[1]
ONE_B = KNOWN_MODELS[2]


def dynamic_graph() -> dict[str, Any]:
    return {
        "inputs": [
            {"name": "input_ids", "type": "tensor(int64)", "shape": ["batch", "sequence"]},
            {"name": "attention_mask", "type": "tensor(int64)", "shape": ["batch", "sequence"]},
            {"name": "label_positions", "type": "tensor(int64)", "shape": ["batch", "labels"]},
            {"name": "label_mask", "type": "tensor(bool)", "shape": ["batch", "labels"]},
        ],
        "outputs": [{"name": "logits", "type": "tensor(float)", "shape": ["batch", "labels"]}],
    }


def parity_block() -> dict[str, Any]:
    return {
        "absolute_tolerance": 1e-4,
        "max_abs_logit_difference": 5e-6,
        "official_max_abs_logit_difference": 4e-7,
        "onnx_official_max_abs_logit_difference": 5e-6,
        "batch_invariance_max_abs_logit_difference": 2e-6,
        "selected_index_matches": 19,
        "rows": 19,
        "synthetic_zero_threshold": 1e-4,
        "nonzero_differences_zeroed": 17,
    }


def exporter_block() -> dict[str, str]:
    return {
        "gliner_runner": "0.2.1",
        "gliner2": "2.1.0",
        "torch": "2.14.0",
        "transformers": "5.19.0",
        "onnx": "1.23.2",
        "onnxscript": "0.7.2",
        "onnxruntime": "1.31.0",
    }


def write_artifact(
    root: Path,
    *,
    spec: InventoryModelSpec = SMALL,
    sidecar: bool = True,
) -> dict[str, Any]:
    """Write a complete fake artifact directory and return its manifest."""
    root.mkdir(parents=True)
    (root / GRAPH_FILE).write_bytes(b"graph")
    if sidecar:
        (root / "model.onnx.data").write_bytes(b"weights" * 10)
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / PREPROCESSING_FILE).write_text("{}", encoding="utf-8")
    manifest = build_manifest(
        spec=spec,
        revision=spec.pinned_revision,
        source_snapshot_sha256="c" * 64,
        source_snapshot_bytes=123,
        parameter_count=spec.parameter_count,
        graph=dynamic_graph(),
        opset_imports=[{"domain": "ai.onnx", "version": 20}],
        external_data_files=["model.onnx.data"] if sidecar else [],
        exporter=exporter_block(),
        parity=parity_block(),
        files=file_inventory(root),
        license_id="apache-2.0",
    )
    (root / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_snapshot(root: Path, spec: InventoryModelSpec = SMALL) -> Path:
    root.mkdir(parents=True)
    content = b"weights"
    (root / "model.safetensors").write_bytes(content)
    (root / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "repository": spec.repository,
                "revision": spec.pinned_revision,
                "files": [
                    {
                        "path": "model.safetensors",
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


# --- identity and arguments -------------------------------------------------------------


@pytest.mark.parametrize("spec", KNOWN_MODELS)
def test_resolve_accepts_logical_id_and_repository_at_the_pinned_revision(
    spec: InventoryModelSpec,
) -> None:
    for name in (spec.model_id, spec.repository):
        assert resolve_curated_model(name) == (spec, spec.pinned_revision)
        assert resolve_curated_model(name, spec.pinned_revision) == (spec, spec.pinned_revision)


@pytest.mark.parametrize(
    ("model", "revision", "message"),
    [
        ("fastino/not-curated", None, "not curated"),
        ("", None, "curated logical id"),
        (" decide-340m", None, "curated logical id"),
        ("decide-340m@" + "a" * 40, None, "--revision"),
        ("decide-340m", "a" * 40, "pinned revision"),
        ("decide-340m", SMALL.pinned_revision[:12], "40-character"),
        ("decide-340m", SMALL.pinned_revision.upper(), "40-character"),
    ],
)
def test_resolve_rejects_uncurated_models_and_unpinned_revisions(
    model: str, revision: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_curated_model(model, revision)


def test_artifact_directory_layout_uses_owner_name_and_revision(tmp_path: Path) -> None:
    assert artifact_directory(tmp_path, ONE_B, ONE_B.pinned_revision) == (
        tmp_path.resolve() / "onnx" / "fastino--GLiNER2.5-Decide-1B" / ONE_B.pinned_revision
    )


@pytest.mark.parametrize("value", [0.0, -1e-4, 1.1e-3, float("nan"), float("inf")])
def test_atol_rejects_unbounded_values(value: float) -> None:
    with pytest.raises(ValueError, match="absolute tolerance"):
        validate_atol(value)
    assert validate_atol(DEFAULT_ATOL) == DEFAULT_ATOL
    assert validate_atol(1e-3) == 1e-3


def test_output_location_refuses_overlap_and_existing_paths(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    target = tmp_path / "models" / "onnx" / "x" / "rev"
    assert validate_output_location(tmp_path / "models", snapshot, target) == target
    with pytest.raises(ValueError, match="contain each other"):
        validate_output_location(snapshot / "models", snapshot, snapshot / "models" / "x")
    with pytest.raises(ValueError, match="contain each other"):
        validate_output_location(tmp_path, snapshot, tmp_path / "x")
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        validate_output_location(tmp_path / "models", snapshot, target)


def test_prepare_refuses_uncurated_revision_before_touching_anything(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        prepare_onnx(
            "decide-340m",
            snapshot=tmp_path / "missing",
            output_dir=tmp_path / "models",
            revision="b" * 40,
        )
    assert not (tmp_path / "models").exists()


def test_prepare_rejects_snapshot_with_mismatched_identity(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot", MULTILINGUAL)
    with pytest.raises(ValueError, match="identity does not match"):
        prepare_onnx("decide-340m", snapshot=snapshot, output_dir=tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_prepare_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    target = artifact_directory(tmp_path / "models", SMALL, SMALL.pinned_revision)
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_onnx("decide-340m", snapshot=snapshot, output_dir=tmp_path / "models")

    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert [item.name for item in target.parent.iterdir()] == [SMALL.pinned_revision]


def test_prepare_publishes_atomically_and_cleans_up_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    models = tmp_path / "models"
    monkeypatch.setattr(onnx_prepare, "_require_gliner2_version", lambda: None)
    seen: list[Path] = []

    def failing(**kwargs: Any) -> None:
        seen.append(kwargs["work"])
        (kwargs["work"] / "partial.bin").write_bytes(b"x")
        raise RuntimeError("parity failed")

    monkeypatch.setattr(onnx_prepare, "_build_artifact", failing)
    with pytest.raises(RuntimeError, match="parity failed"):
        prepare_onnx("decide-340m", snapshot=snapshot, output_dir=models, log=lambda _m: None)
    target = artifact_directory(models, SMALL, SMALL.pinned_revision)
    assert not target.exists()
    assert not seen[0].exists()
    assert list(target.parent.iterdir()) == []

    def succeeding(**kwargs: Any) -> None:
        assert kwargs["work"].parent == target.parent
        assert kwargs["work"].name.startswith(f".{SMALL.pinned_revision[:12]}.tmp-")
        assert kwargs["atol"] == DEFAULT_ATOL
        write_artifact(kwargs["work"] / "inner")

    monkeypatch.setattr(onnx_prepare, "_build_artifact", succeeding)
    result = prepare_onnx("fastino/GLiNER2.5-Decide", snapshot=snapshot, output_dir=models)
    assert result == target
    assert (target / "inner" / MANIFEST_FILE).is_file()
    assert (target.stat().st_mode & 0o777) == 0o755
    assert [item.name for item in target.parent.iterdir()] == [SMALL.pinned_revision]


def test_modules_import_without_heavy_dependencies() -> None:
    code = """
import sys
for name in ("torch", "onnx", "onnxruntime", "gliner2", "transformers", "onnxscript"):
    sys.modules[name] = None
import gliner_runner.onnx_prepare, gliner_runner.onnx_reference, gliner_runner.cli
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "ok"


# --- fixtures ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", KNOWN_MODELS)
def test_fixtures_cover_multi_task_described_instruction_multi_label_and_padding(
    spec: InventoryModelSpec,
) -> None:
    batches = fixture_batches(spec)
    rows = [row for batch in batches for row in batch]
    tasks = [task for row in rows for task in row.schema["tasks"].values()]

    assert any(len(row.schema["tasks"]) >= 2 for row in rows)
    assert any(isinstance(task["labels"], dict) for task in tasks)
    assert any(task.get("instruction") for task in tasks)
    assert any(task.get("max_labels", 1) > 1 for task in tasks)
    assert any(len(batch) > 1 for batch in batches)
    assert len({sum(row.task_sizes) for batch in batches for row in batch}) > 2
    assert len({len(row.text) for row in rows}) > 3
    strings = json.dumps([(row.text, row.schema) for row in rows], ensure_ascii=False)
    for reserved in ("[P]", "[L]", "[SEP", "(", ")"):
        assert reserved not in strings


def test_fixture_digest_is_stable_and_language_specific() -> None:
    assert fixture_sha256(SMALL) == fixture_sha256(ONE_B)
    assert fixture_sha256(SMALL) != fixture_sha256(MULTILINGUAL)
    assert any(
        ord(character) > 0x3000
        for row in fixture_batches(MULTILINGUAL)[3]
        for character in row.text
    )


# --- logit comparison -------------------------------------------------------------------


def test_compare_task_logits_checks_argmax_per_task_and_ignores_padding() -> None:
    expected = [[0.1, 0.9, 0.3, 0.2, -3.4e38], [0.5, 0.4, 0.2, 0.1, 0.0]]
    actual = [[0.10001, 0.89999, 0.30002, 0.2, -3.0e38], [0.5, 0.4, 0.2, 0.1, 0.0]]
    mask = [[True, True, True, True, False], [True, True, True, True, True]]

    result = compare_task_logits(expected, actual, mask, [[2, 2], [2, 2, 1]])

    assert len(result["selections"]) == 5
    assert result["selected_index_matches"] == 5
    assert result["valid_logits_compared"] == 9
    assert result["max_abs_logit_difference"] == pytest.approx(2e-5)
    assert result["nonzero_differences_zeroed"] == 3
    assert result["differences_below_threshold"] == 9


def test_compare_task_logits_detects_a_flipped_task_selection() -> None:
    result = compare_task_logits(
        [[0.5, 0.51, 0.9, 0.1]],
        [[0.51, 0.5, 0.9, 0.1]],
        [[True] * 4],
        [[2, 2]],
    )
    assert result["selected_index_matches"] == 1
    assert [item["selected_index_match"] for item in result["selections"]] == [False, True]


@pytest.mark.parametrize(
    ("mask", "sizes"),
    [
        ([[True, True, False]], [[3]]),
        ([[True, False, True]], [[2]]),
        ([[True, True, True]], [[2]]),
        ([[True, True, True]], [[0, 3]]),
        ([[True, True, True]], [[]]),
    ],
)
def test_compare_task_logits_rejects_inconsistent_masks(
    mask: list[list[bool]], sizes: list[list[int]]
) -> None:
    with pytest.raises(ValueError, match="label mask"):
        compare_task_logits([[0.0] * 3], [[0.0] * 3], mask, sizes)


def test_compare_task_logits_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        compare_task_logits([[0.0, float("nan")]], [[0.0, 0.0]], [[True, True]], [[2]])


# --- external data ----------------------------------------------------------------------


def test_external_data_inventory_requires_complete_referenced_sidecars(tmp_path: Path) -> None:
    (tmp_path / "model.onnx.data").write_bytes(b"0" * 100)
    entries = [("model.onnx.data", 0, 60), ("model.onnx.data", 60, 40)]
    assert validate_external_data_inventory(tmp_path, entries) == ["model.onnx.data"]
    assert validate_external_data_inventory(tmp_path, [("model.onnx.data", 0, 0)]) == [
        "model.onnx.data"
    ]

    with pytest.raises(ValueError, match="truncated"):
        validate_external_data_inventory(tmp_path, [("model.onnx.data", 60, 41)])
    with pytest.raises(ValueError, match="missing"):
        validate_external_data_inventory(tmp_path, [("other.data", 0, 1)])
    with pytest.raises(ValueError, match="escapes"):
        validate_external_data_inventory(tmp_path, [("../model.onnx.data", 0, 1)])
    with pytest.raises(ValueError, match="invalid range"):
        validate_external_data_inventory(tmp_path, [("model.onnx.data", -1, 1)])
    (tmp_path / "model.onnx.extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="unreferenced"):
        validate_external_data_inventory(tmp_path, entries)


def test_external_data_inventory_accepts_graphs_without_sidecars(tmp_path: Path) -> None:
    assert validate_external_data_inventory(tmp_path, []) == []


# --- file inventory and manifest --------------------------------------------------------


def test_file_inventory_hashes_every_file_except_the_manifest(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.bin").write_bytes(b"")
    (tmp_path / MANIFEST_FILE).write_text("{}", encoding="utf-8")

    assert file_inventory(tmp_path) == [
        {"path": "a.bin", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
        {"path": "nested/b.bin", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()},
    ]
    assert file_sha256(tmp_path / "a.bin") == hashlib.sha256(b"abc").hexdigest()


def test_manifest_schema_matches_the_documented_shape(tmp_path: Path) -> None:
    manifest = write_artifact(tmp_path / "artifact")

    assert list(manifest)[:3] == ["format_version", "kind", "model_id"]
    assert manifest["format_version"] == 1
    assert manifest["kind"] == "gliner-runner-onnx-classifier"
    assert manifest["operations"] == ["classify"]
    assert (manifest["precision"], manifest["quantization"]) == ("fp32", "none")
    assert (manifest["provider"], manifest["opset"]) == ("CPUExecutionProvider", 20)
    assert manifest["graph"]["path"] == "model.onnx"
    assert manifest["graph"]["external_data"] == ["model.onnx.data"]
    assert [item["name"] for item in manifest["graph"]["inputs"]] == [
        "input_ids",
        "attention_mask",
        "label_positions",
        "label_mask",
    ]
    assert manifest["graph"]["outputs"][0]["name"] == "logits"
    assert set(manifest["exporter"]) == {
        "gliner_runner",
        "gliner2",
        "torch",
        "transformers",
        "onnx",
        "onnxscript",
        "onnxruntime",
    }
    assert {item["path"] for item in manifest["files"]} == {
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
        "gliner2.json",
    }
    assert MANIFEST_FILE not in {item["path"] for item in manifest["files"]}
    json.dumps(manifest)


def mutate(path: str, value: object) -> Any:
    def apply(manifest: dict[str, Any]) -> None:
        target: Any = manifest
        *parents, leaf = path.split(".")
        for key in parents:
            target = target[key]
        if value is _DELETE:
            del target[leaf]
        else:
            target[leaf] = value

    return apply


_DELETE = object()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (mutate("precision", "fp16"), "precision"),
        (mutate("quantization", "int8"), "quantization"),
        (mutate("provider", "CoreMLExecutionProvider"), "provider"),
        (mutate("opset", 19), "opset"),
        (mutate("operations", ["classify", "extract_entities"]), "operations"),
        (mutate("model_id", "decide-9b"), "model_id"),
        (mutate("repository", "fastino/GLiNER2.5-multi-Decide"), "repository"),
        (mutate("revision", "d" * 40), "pinned revision"),
        (mutate("source_snapshot_sha256", "xyz"), "source_snapshot_sha256"),
        (mutate("source_snapshot_bytes", 0), "source_snapshot_bytes"),
        (mutate("parameter_count", 1), "parameter_count"),
        (mutate("opset_imports", [{"domain": "ai.onnx", "version": 19}]), "opset"),
        (mutate("graph.path", "classifier.onnx"), "graph.path"),
        (mutate("graph.external_data", ["a", "a"]), "external_data"),
        (mutate("graph.inputs", []), "inputs"),
        (mutate("exporter.torch", _DELETE), "exporter"),
        (mutate("exporter.gliner2", "2.0.0"), "GLiNER2"),
        (mutate("parity.max_abs_logit_difference", 2e-4), "exceeds the absolute tolerance"),
        (mutate("parity.official_max_abs_logit_difference", float("nan")), "tolerance"),
        (mutate("parity.selected_index_matches", 18), "selection"),
        (mutate("parity.absolute_tolerance", 1e-2), "tolerance"),
        (mutate("parity.synthetic_zero_threshold", 1e-3), "synthetic_zero_threshold"),
        (mutate("parity.rows", 0), "rows"),
        (mutate("parity.batch_invariance_max_abs_logit_difference", _DELETE), "required"),
        (mutate("files", []), "files"),
    ],
)
def test_manifest_validation_fails_closed(tmp_path: Path, change: Any, message: str) -> None:
    manifest = write_artifact(tmp_path / "artifact")
    broken = copy.deepcopy(manifest)
    change(broken)

    with pytest.raises(ValueError, match=message):
        validate_manifest(broken)


@pytest.mark.parametrize(
    "entry",
    [
        {"path": MANIFEST_FILE, "size": 1, "sha256": "a" * 64},
        {"path": "../x", "size": 1, "sha256": "a" * 64},
        {"path": "/x", "size": 1, "sha256": "a" * 64},
        {"path": "extra.bin", "size": 1, "sha256": "a" * 64},
        {"path": "model.onnx", "size": 1, "sha256": "a" * 64},
        {"path": "tokenizer.json", "size": -1, "sha256": "a" * 64},
        {"path": "tokenizer.json", "size": 1, "sha256": "short"},
    ],
)
def test_manifest_validation_rejects_bad_file_entries(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    manifest = write_artifact(tmp_path / "artifact")
    manifest["files"].append(entry)

    with pytest.raises(ValueError, match="invalid ONNX artifact manifest"):
        validate_manifest(manifest)


def test_manifest_requires_the_graph_sidecars_to_be_inventoried(tmp_path: Path) -> None:
    manifest = write_artifact(tmp_path / "artifact")
    manifest["files"] = [item for item in manifest["files"] if item["path"] != "model.onnx.data"]

    with pytest.raises(ValueError, match="sidecars"):
        validate_manifest(manifest)


def test_verify_artifact_directory_accepts_a_complete_artifact(tmp_path: Path) -> None:
    for index, spec in enumerate(KNOWN_MODELS):
        root = tmp_path / str(index)
        manifest = write_artifact(root, spec=spec, sidecar=index != 1)
        assert verify_artifact_directory(root) == manifest


def test_verify_artifact_directory_rejects_missing_altered_and_extra_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    write_artifact(root)

    (root / "model.onnx.data").write_bytes(b"weighTs" * 10)
    with pytest.raises(ValueError, match=r"checksum mismatch: model\.onnx\.data"):
        verify_artifact_directory(root)

    (root / "model.onnx.data").write_bytes(b"short")
    with pytest.raises(ValueError, match=r"size mismatch: model\.onnx\.data"):
        verify_artifact_directory(root)

    (root / "model.onnx.data").write_bytes(b"weights" * 10)
    verify_artifact_directory(root)

    (root / "stray.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"inventory mismatch: stray\.bin"):
        verify_artifact_directory(root)
    (root / "stray.bin").unlink()

    (root / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match=r"inventory mismatch: tokenizer\.json"):
        verify_artifact_directory(root)


def test_verify_artifact_directory_rejects_symlinked_files(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_artifact(root)
    original = root / "tokenizer.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    original.unlink()
    original.symlink_to(outside)

    with pytest.raises(ValueError, match=r"size mismatch: tokenizer\.json"):
        verify_artifact_directory(root)


def test_verify_artifact_directory_rejects_unreadable_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    write_artifact(root)
    (root / MANIFEST_FILE).write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        verify_artifact_directory(root)


# --- preprocessing config ---------------------------------------------------------------


def tokenizer_json() -> dict[str, Any]:
    added = [{"id": 5, "content": "[MASK]", "special": True}]
    added += [
        {"id": 100 + index, "content": token, "special": True}
        for index, token in enumerate(SPECIAL_TOKENS)
    ]
    return {"added_tokens": added}


def test_special_token_ids_come_from_the_tokenizer_added_token_table() -> None:
    ids = special_token_ids_from_tokenizer_json(tokenizer_json())

    assert list(ids) == list(SPECIAL_TOKENS)
    assert ids["[SEP_STRUCT]"] == 100
    assert ids["[DESCRIPTION]"] == 109


def test_special_token_ids_reject_missing_or_duplicate_markers() -> None:
    missing = tokenizer_json()
    missing["added_tokens"] = [t for t in missing["added_tokens"] if t["content"] != "[L]"]
    with pytest.raises(ValueError, match=r"missing GLiNER2 marker tokens: \[L\]"):
        special_token_ids_from_tokenizer_json(missing)
    duplicate = tokenizer_json()
    duplicate["added_tokens"][2]["id"] = duplicate["added_tokens"][1]["id"]
    with pytest.raises(ValueError, match="distinct"):
        special_token_ids_from_tokenizer_json(duplicate)
    with pytest.raises(ValueError, match="added_tokens"):
        special_token_ids_from_tokenizer_json({})


def test_preprocessing_config_records_everything_the_port_needs() -> None:
    ids = special_token_ids_from_tokenizer_json(tokenizer_json())
    config = build_preprocessing_config(
        spec=SMALL,
        revision=SMALL.pinned_revision,
        special_token_ids=ids,
        unknown_token=("[UNK]", 3),
        model_config={"architecture": "span", "token_pooling": "first", "max_len": None},
        tokenizer_config={"tokenizer_class": "DebertaV2Tokenizer"},
        has_special_tokens_map=False,
    )

    assert config["kind"] == "gliner-runner-gliner2-preprocessing"
    assert config["special_tokens"] == ids
    assert config["tokenizer"]["unknown_token"] == {"token": "[UNK]", "id": 3}
    assert config["tokenizer"]["encode_each_word_independently"] is True
    assert config["tokenizer"]["add_special_tokens"] is False
    assert config["tokenizer"]["special_tokens_map_json"] is None
    assert config["text"]["lowercase_words"] is True
    assert config["text"]["max_words"] is None
    assert config["text"]["normalization"]["terminal_characters"] == [".", "!", "?"]
    assert config["batching"]["input_ids_pad_value"] == 0
    assert config["batching"]["padding_side"] == "right"
    assert "(" in config["reserved_strings"]
    json.dumps(config)


def test_preprocessing_config_rejects_unsupported_pooling() -> None:
    with pytest.raises(ValueError, match="token pooling"):
        build_preprocessing_config(
            spec=SMALL,
            revision=SMALL.pinned_revision,
            special_token_ids=special_token_ids_from_tokenizer_json(tokenizer_json()),
            unknown_token=None,
            model_config={"token_pooling": "mean"},
            tokenizer_config={},
            has_special_tokens_map=True,
        )


def test_snapshot_license_reads_front_matter(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n# Model\n", encoding="utf-8"
    )
    assert onnx_prepare.snapshot_license(tmp_path) == "apache-2.0"
    (tmp_path / "README.md").write_text("# no front matter\nlicense: mit\n", encoding="utf-8")
    assert onnx_prepare.snapshot_license(tmp_path) is None
    assert onnx_prepare.snapshot_license(tmp_path / "absent") is None


# --- CLI --------------------------------------------------------------------------------


def test_prepare_onnx_cli_passes_arguments_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_prepare(model: str, **kwargs: Any) -> Path:
        calls.append({"model": model, **kwargs})
        if model == "bad":
            raise ValueError("model 'bad' is not curated")
        return tmp_path / "artifact"

    monkeypatch.setattr(cli, "prepare_onnx", fake_prepare)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    base = ["prepare-onnx", "--snapshot", str(snapshot), "--output-dir", str(tmp_path / "models")]

    ok = CliRunner().invoke(
        cli.app, [*base, "--model", "decide-340m", "--revision", "a" * 40, "--atol", "5e-5"]
    )
    failed = CliRunner().invoke(cli.app, [*base, "--model", "bad"])

    assert ok.exit_code == 0
    assert json.loads(ok.stdout) == {"artifact": str(tmp_path / "artifact")}
    assert calls[0] == {
        "model": "decide-340m",
        "snapshot": snapshot,
        "output_dir": tmp_path / "models",
        "revision": "a" * 40,
        "atol": 5e-5,
        "template_dir": None,
    }
    assert failed.exit_code == 1
    assert "not curated" in failed.output


# --- guarded real-model integration -----------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    "GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT" not in os.environ,
    reason="set GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT to a pinned local snapshot",
)
def test_real_snapshot_prepares_a_verified_artifact(tmp_path: Path) -> None:
    model = os.environ.get("GLINER_RUNNER_ONNX_PREPARE_MODEL", "decide-340m")
    spec, _revision = resolve_curated_model(model)

    artifact = prepare_onnx(
        model,
        snapshot=Path(os.environ["GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT"]),
        output_dir=tmp_path,
    )

    assert artifact == artifact_directory(tmp_path, spec, spec.pinned_revision)
    manifest = verify_artifact_directory(artifact)
    parity = manifest["parity"]
    assert parity["selected_index_matches"] == parity["rows"]
    assert parity["max_abs_logit_difference"] <= parity["absolute_tolerance"]
    assert manifest["parameter_count"] == spec.parameter_count
    with pytest.raises(FileExistsError):
        prepare_onnx(
            model,
            snapshot=Path(os.environ["GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT"]),
            output_dir=tmp_path,
        )
