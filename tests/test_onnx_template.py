"""Template format, layout math, and the pure-Python reference assembler (no torch needed)."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gliner_runner import cli
from gliner_runner.onnx_template import (
    DATA_ALIGNMENT,
    align_up,
    assemble_from_template,
    load_template,
    plan_layout,
    read_safetensors_index,
    template_artifact_path,
    transpose_blocks,
    validate_template,
    validate_template_manifest,
)

REVISION = "a" * 40


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (shape, raw) in tensors.items():
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(raw for _, raw in tensors.values())
    )


def naive_transpose(raw: bytes, rows: int, cols: int) -> bytes:
    words = [raw[index * 4 : index * 4 + 4] for index in range(rows * cols)]
    return b"".join(words[r * cols + c] for c in range(cols) for r in range(rows))


MATRIX = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2 x 3
WEIGHTS = {
    "a.weight": ([2, 3], f32([1, 2, 3, 4, 5, 6])),
    "b.bias": ([5], f32([10, 11, 12, 13, 14])),
    "c.weight": ([3, 4], f32(list(range(12)))),
}


class Fixture:
    def __init__(self, root: Path) -> None:
        self.snapshot = root / "snapshot"
        self.template = root / "template"
        self.snapshot.mkdir()
        self.template.mkdir()
        write_safetensors(self.snapshot / "model.safetensors", WEIGHTS)
        self.tokenizer = b'{"model": {}}'
        self.tokenizer_config = b'{"x": 1}'
        (self.snapshot / "tokenizer.json").write_bytes(self.tokenizer)
        (self.snapshot / "tokenizer_config.json").write_bytes(self.tokenizer_config)
        self.graph = b"graph-bytes\x00\x01"
        self.preprocessing = b'{"kind": "pre"}\n'
        a = WEIGHTS["a.weight"][1]
        b = WEIGHTS["b.bias"][1]
        c = WEIGHTS["c.weight"][1]
        c_t = naive_transpose(c, 3, 4)
        # layout: a at 0 (24 bytes), b at 64 (20), c^T at 128 (48)
        self.data = a + bytes(40) + b + bytes(44) + c_t
        self.tensors = [
            self.tensor("init.a", "a.weight", "identity", [2, 3], [2, 3], 0, 24),
            self.tensor("init.b", "b.bias", "identity", [5], [5], 64, 20),
            self.tensor("init.c", "c.weight", "transpose", [3, 4], [4, 3], 128, 48),
        ]
        outputs = {
            "model.onnx": self.graph,
            "model.onnx.data": self.data,
            "gliner2.json": self.preprocessing,
            "tokenizer.json": self.tokenizer,
            "tokenizer_config.json": self.tokenizer_config,
        }
        self.manifest = {
            "format_version": 1,
            "model_id": "decide-340m",
            "repository": "owner/name",
            "revision": REVISION,
            "graph": {"path": "model.onnx", "external_data": ["model.onnx.data"]},
            "files": [
                {"path": path, "size": len(raw), "sha256": sha(raw)}
                for path, raw in sorted(outputs.items())
            ],
        }
        manifest_bytes = (json.dumps(self.manifest, indent=2) + "\n").encode()
        (self.template / "graph.onnx").write_bytes(self.graph)
        (self.template / "gliner2.json").write_bytes(self.preprocessing)
        (self.template / "manifest.json").write_bytes(manifest_bytes)

        def entry(path: str, raw: bytes, **extra: str) -> dict[str, Any]:
            return {"path": path, "size": len(raw), "sha256": sha(raw), **extra}

        weights_raw = (self.snapshot / "model.safetensors").read_bytes()
        self.doc: dict[str, Any] = {
            "format_version": 1,
            "kind": "gliner-runner-onnx-template",
            "model_id": "decide-340m",
            "repository": "owner/name",
            "revision": REVISION,
            "source_files": [
                entry("model.safetensors", weights_raw, role="weights"),
                entry("tokenizer.json", self.tokenizer, role="tokenizer"),
                entry("tokenizer_config.json", self.tokenizer_config, role="tokenizer"),
            ],
            "template_files": [
                entry("graph.onnx", self.graph, output="model.onnx"),
                entry("gliner2.json", self.preprocessing, output="gliner2.json"),
                entry("manifest.json", manifest_bytes, output="manifest.json"),
            ],
            "data": {
                "path": "model.onnx.data",
                "alignment": 64,
                "size": len(self.data),
                "sha256": sha(self.data),
                "tensors": self.tensors,
            },
            "outputs": [entry(path, raw) for path, raw in outputs.items()],
        }
        self.write()

    @staticmethod
    def tensor(
        name: str,
        source: str,
        transform: str,
        source_shape: list[int],
        shape: list[int],
        offset: int,
        length: int,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "source": source,
            "transform": transform,
            "dtype": "float32",
            "source_shape": source_shape,
            "shape": shape,
            "offset": offset,
            "length": length,
        }

    def write(self) -> None:
        (self.template / "template.json").write_text(json.dumps(self.doc, indent=2) + "\n")

    def assemble(self, output: Path) -> Path:
        return assemble_from_template(self.template, self.snapshot, output)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


# --- safetensors --------------------------------------------------------------------------


def test_safetensors_index_parses_offsets_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "w.safetensors"
    write_safetensors(path, WEIGHTS)
    index = read_safetensors_index(path)
    assert set(index.tensors) == set(WEIGHTS)
    tensor = index.tensors["c.weight"]
    assert tensor.shape == (3, 4) and tensor.dtype == "F32" and tensor.element_count == 12
    raw = path.read_bytes()
    assert raw[tensor.start : tensor.end] == WEIGHTS["c.weight"][1]
    assert index.file_size == len(raw)


def header_file(path: Path, header: bytes, data: bytes = b"") -> Path:
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return path


@pytest.mark.parametrize(
    ("header", "data", "message"),
    [
        (b"not json", b"", "valid JSON"),
        (b"[]", b"", "JSON object"),
        (b'{"t": 1}', b"", "must be an object"),
        (b'{"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "t": {}}', b"", "repeats"),
        (b'{"t": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}', bytes(4), "shape"),
        (b'{"t": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}', bytes(4), "data_offsets"),
        (b'{"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}}', bytes(4), "outside"),
        (b'{"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}', bytes(4), "F32 shape"),
        (b'{"t": {"shape": [1], "data_offsets": [0, 4]}}', bytes(4), "dtype"),
    ],
)
def test_safetensors_rejects_malformed_headers(
    tmp_path: Path, header: bytes, data: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_safetensors_index(header_file(tmp_path / "w", header, data))


def test_safetensors_rejects_truncated_and_oversized_headers(tmp_path: Path) -> None:
    short = tmp_path / "short"
    short.write_bytes(b"abc")
    with pytest.raises(ValueError, match="truncated"):
        read_safetensors_index(short)
    lying = tmp_path / "lying"
    lying.write_bytes(struct.pack("<Q", 1000) + b"{}")
    with pytest.raises(ValueError, match="truncated"):
        read_safetensors_index(lying)
    huge = tmp_path / "huge"
    huge.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(ValueError, match="size is invalid"):
        read_safetensors_index(huge)
    with pytest.raises(ValueError, match="unreadable"):
        read_safetensors_index(tmp_path / "missing")


# --- layout and transpose math ------------------------------------------------------------


def test_align_up_and_plan_layout() -> None:
    assert [align_up(value) for value in (0, 1, 63, 64, 65, 128)] == [0, 64, 64, 64, 128, 128]
    assert align_up(10, 4) == 12
    offsets, total = plan_layout([24, 20, 48])
    assert offsets == [0, 64, 128] and total == 176
    assert plan_layout([]) == ([], 0)
    assert plan_layout([64, 64]) == ([0, 64], 128)
    assert plan_layout([0, 5]) == ([0, 0], 5)
    assert DATA_ALIGNMENT == 64
    with pytest.raises(ValueError):
        plan_layout([-1])
    with pytest.raises(ValueError):
        align_up(1, 0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 3), (5, 4), (13, 17)])
@pytest.mark.parametrize("block_bytes", [4, 100, 1 << 20])
def test_transpose_blocks_matches_naive_transpose(shape: tuple[int, int], block_bytes: int) -> None:
    rows, cols = shape
    raw = f32([float(index) * 1.5 - 3 for index in range(rows * cols)])
    blocks = list(transpose_blocks(memoryview(raw), rows, cols, block_bytes))
    assert b"".join(blocks) == naive_transpose(raw, rows, cols)
    if block_bytes == 4:
        assert len(blocks) == cols


def test_transpose_blocks_preserves_non_finite_bit_patterns() -> None:
    raw = bytes([0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80, 0x7F, 0x01, 0x00, 0x00, 0x00])
    assert b"".join(transpose_blocks(memoryview(raw), 1, 3)) == raw
    assert b"".join(transpose_blocks(memoryview(raw), 3, 1)) == raw


def test_transpose_blocks_rejects_size_mismatch_and_handles_empty() -> None:
    with pytest.raises(ValueError, match="does not match"):
        list(transpose_blocks(memoryview(bytes(12)), 2, 3))
    assert list(transpose_blocks(memoryview(b""), 0, 5)) == []


# --- template schema ----------------------------------------------------------------------


def test_valid_template_passes(fixture: Fixture) -> None:
    assert validate_template(copy.deepcopy(fixture.doc))["revision"] == REVISION
    assert load_template(fixture.template)["data"]["size"] == 176


def mutations() -> list[tuple[str, Any, str]]:
    def set_path(path: str, value: Any) -> Any:
        def apply(doc: dict[str, Any]) -> None:
            target: Any = doc
            *parents, last = path.split(".")
            for part in parents:
                target = target[int(part)] if isinstance(target, list) else target[part]
            if isinstance(target, list):
                target[int(last)] = value
            else:
                target[last] = value

        return apply

    def delete(key: str) -> Any:
        return lambda doc: doc.pop(key)

    def drop_output(path: str) -> Any:
        def apply(doc: dict[str, Any]) -> None:
            doc["outputs"] = [item for item in doc["outputs"] if item["path"] != path]

        return apply

    return [
        ("format", set_path("format_version", 2), "format_version"),
        ("format-bool", set_path("format_version", True), "format_version"),
        ("kind", set_path("kind", "other"), "kind"),
        ("repository", set_path("repository", "no-slash"), "repository"),
        ("repository-escape", set_path("repository", "../x"), "repository"),
        ("revision", set_path("revision", "abc"), "revision"),
        ("missing-key", delete("outputs"), "exactly the keys"),
        ("extra-key", set_path("surprise", 1), "exactly the keys"),
        ("source-path", set_path("source_files.1.path", "../tokenizer.json"), "plain file name"),
        ("source-role", set_path("source_files.1.role", "other"), "role"),
        ("source-sha", set_path("source_files.1.sha256", "XYZ"), "SHA-256"),
        ("source-size", set_path("source_files.1.size", -1), "non-negative"),
        ("template-output", set_path("template_files.0.output", "evil.onnx"), "template_files"),
        ("data-path", set_path("data.path", "x.data"), "data.path"),
        ("data-align", set_path("data.alignment", 16), "alignment"),
        ("data-size", set_path("data.size", 177), "data.size"),
        ("tensor-offset", set_path("data.tensors.1.offset", 60), "offsets"),
        ("tensor-length", set_path("data.tensors.1.length", 24), "length"),
        ("tensor-transform", set_path("data.tensors.0.transform", "scale"), "transform"),
        ("tensor-dtype", set_path("data.tensors.0.dtype", "float16"), "dtype"),
        ("tensor-shape", set_path("data.tensors.2.shape", [3, 4]), "swap"),
        ("identity-shape", set_path("data.tensors.0.shape", [3, 2]), "keep its shape"),
        ("tensor-name", set_path("data.tensors.1.name", "init.a"), "repeats"),
        ("no-tensors", set_path("data.tensors", []), "empty"),
        ("output-missing", drop_output("model.onnx.data"), "outputs must list"),
        ("output-extra", set_path("outputs.0.path", "model.bin"), "outputs must list"),
        ("output-tokenizer", set_path("outputs.3.sha256", "0" * 64), "source file"),
        ("output-graph", set_path("outputs.0.sha256", "0" * 64), "template file"),
        ("output-data", set_path("outputs.1.size", 1), "data section"),
    ]


@pytest.mark.parametrize(("label", "mutate", "message"), mutations(), ids=lambda v: str(v)[:24])
def test_template_schema_rejects_mutations(
    fixture: Fixture, label: str, mutate: Any, message: str
) -> None:
    doc = copy.deepcopy(fixture.doc)
    mutate(doc)
    with pytest.raises(ValueError, match=message):
        validate_template(doc)


def test_template_requires_one_weights_file_and_tokenizers(fixture: Fixture) -> None:
    doc = copy.deepcopy(fixture.doc)
    doc["source_files"][0]["role"] = "tokenizer"
    with pytest.raises(ValueError, match="weights"):
        validate_template(doc)
    doc = copy.deepcopy(fixture.doc)
    doc["source_files"] = doc["source_files"][:1]
    with pytest.raises(ValueError, match="tokenizer"):
        validate_template(doc)
    with pytest.raises(ValueError, match="template must be an object"):
        validate_template([])


def test_manifest_must_inventory_exactly_the_template_outputs(fixture: Fixture) -> None:
    template = validate_template(copy.deepcopy(fixture.doc))
    validate_template_manifest(template, fixture.manifest)
    broken = copy.deepcopy(fixture.manifest)
    broken["files"].pop()
    with pytest.raises(ValueError, match="files do not equal"):
        validate_template_manifest(template, broken)
    broken = copy.deepcopy(fixture.manifest)
    broken["graph"]["external_data"] = []
    with pytest.raises(ValueError, match="external_data"):
        validate_template_manifest(template, broken)
    broken = copy.deepcopy(fixture.manifest)
    broken["revision"] = "b" * 40
    with pytest.raises(ValueError, match="identity"):
        validate_template_manifest(template, broken)
    with pytest.raises(ValueError, match=r"manifest\.json must be an object"):
        validate_template_manifest(template, [])


# --- assembler ----------------------------------------------------------------------------


def test_assembler_rebuilds_the_exact_artifact(fixture: Fixture, tmp_path: Path) -> None:
    template = load_template(fixture.template)
    target = template_artifact_path(tmp_path / "models", template)
    assert target.parts[-3:] == ("owner--name", REVISION) or target.parts[-3:][1:] == (
        "owner--name",
        REVISION,
    )
    result = fixture.assemble(target)
    assert result == target
    assert sorted(item.name for item in target.iterdir()) == [
        "gliner2.json",
        "manifest.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert (target / "model.onnx").read_bytes() == fixture.graph
    assert (target / "model.onnx.data").read_bytes() == fixture.data
    assert (target / "tokenizer.json").read_bytes() == fixture.tokenizer
    assert (target / "manifest.json").read_bytes() == (
        fixture.template / "manifest.json"
    ).read_bytes()
    assert [item.name for item in target.parent.iterdir()] == [REVISION]


def test_assembler_data_padding_is_zero_without_trailing_bytes(
    fixture: Fixture, tmp_path: Path
) -> None:
    target = fixture.assemble(tmp_path / "out")
    data = (target / "model.onnx.data").read_bytes()
    assert len(data) == 128 + 48
    assert data[24:64] == bytes(40) and data[84:128] == bytes(44)
    assert data[128:] == naive_transpose(WEIGHTS["c.weight"][1], 3, 4)


def test_assembler_refuses_existing_target_and_overlap(fixture: Fixture, tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("x")
    with pytest.raises(FileExistsError):
        fixture.assemble(existing)
    assert (existing / "keep").read_text() == "x"
    with pytest.raises(ValueError, match="contain each other"):
        fixture.assemble(fixture.snapshot / "out")


def assert_clean_failure(fixture: Fixture, tmp_path: Path, match: str) -> None:
    out = tmp_path / "models" / "artifact"
    with pytest.raises(ValueError, match=match):
        fixture.assemble(out)
    assert not out.exists()
    assert not out.parent.exists() or list(out.parent.iterdir()) == []


def test_assembler_rejects_altered_weights(fixture: Fixture, tmp_path: Path) -> None:
    path = fixture.snapshot / "model.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert_clean_failure(fixture, tmp_path, "checksum mismatch: model.safetensors")


def test_assembler_rejects_truncated_weights_and_missing_tokenizer(
    fixture: Fixture, tmp_path: Path
) -> None:
    path = fixture.snapshot / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-4])
    assert_clean_failure(fixture, tmp_path, "size mismatch: model.safetensors")


def test_assembler_rejects_missing_and_altered_tokenizer(fixture: Fixture, tmp_path: Path) -> None:
    (fixture.snapshot / "tokenizer.json").write_bytes(b'{"model": {"x": 1}}')
    assert_clean_failure(fixture, tmp_path, "tokenizer.json")
    (fixture.snapshot / "tokenizer.json").unlink()
    assert_clean_failure(fixture, tmp_path, "tokenizer.json")


def test_assembler_rejects_symlinked_snapshot_file(fixture: Fixture, tmp_path: Path) -> None:
    real = fixture.snapshot / "tokenizer.json"
    moved = tmp_path / "elsewhere.json"
    moved.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(moved)
    assert_clean_failure(fixture, tmp_path, "not a regular file")


def test_assembler_rejects_altered_template_files(fixture: Fixture, tmp_path: Path) -> None:
    (fixture.template / "graph.onnx").write_bytes(b"tampered")
    assert_clean_failure(fixture, tmp_path, "template file")


def test_assembler_rejects_manifest_that_does_not_match_outputs(
    fixture: Fixture, tmp_path: Path
) -> None:
    manifest = json.loads((fixture.template / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (fixture.template / "manifest.json").write_bytes(raw)
    fixture.doc["template_files"][2]["size"] = len(raw)
    fixture.doc["template_files"][2]["sha256"] = sha(raw)
    fixture.write()
    assert_clean_failure(fixture, tmp_path, "files do not equal")


def test_assembler_rejects_template_pointing_at_missing_or_mismatched_tensors(
    fixture: Fixture, tmp_path: Path
) -> None:
    fixture.tensors[0]["source"] = "nope"
    fixture.write()
    assert_clean_failure(fixture, tmp_path, "no tensor 'nope'")
    fixture.tensors[0]["source"] = "a.weight"
    fixture.tensors[0]["source_shape"] = fixture.tensors[0]["shape"] = [3, 2]
    fixture.write()
    assert_clean_failure(fixture, tmp_path, "does not match the template")


def test_assembler_rejects_data_whose_checksum_disagrees(fixture: Fixture, tmp_path: Path) -> None:
    # A template whose data hash is wrong must never publish an artifact, even though every
    # other field is internally consistent.
    fixture.doc["data"]["sha256"] = "0" * 64
    fixture.doc["outputs"][1]["sha256"] = "0" * 64
    manifest = json.loads((fixture.template / "manifest.json").read_text())
    next(item for item in manifest["files"] if item["path"] == "model.onnx.data")["sha256"] = (
        "0" * 64
    )
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (fixture.template / "manifest.json").write_bytes(raw)
    fixture.doc["template_files"][2]["size"] = len(raw)
    fixture.doc["template_files"][2]["sha256"] = sha(raw)
    fixture.write()
    assert_clean_failure(fixture, tmp_path, "data checksum")


def test_assembler_rejects_corrupt_template_json(fixture: Fixture, tmp_path: Path) -> None:
    (fixture.template / "template.json").write_text("{")
    assert_clean_failure(fixture, tmp_path, "unreadable")
    (fixture.template / "template.json").unlink()
    assert_clean_failure(fixture, tmp_path, "not a regular file")


# --- CLI and import hygiene ---------------------------------------------------------------


def test_assemble_onnx_cli(fixture: Fixture, tmp_path: Path) -> None:
    models = tmp_path / "models"
    result = CliRunner().invoke(
        cli.app,
        [
            "assemble-onnx",
            "--template",
            str(fixture.template),
            "--snapshot",
            str(fixture.snapshot),
            "--output-dir",
            str(models),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = Path(json.loads(result.stdout)["artifact"])
    assert artifact == (models / "onnx" / "owner--name" / REVISION).resolve()
    assert (artifact / "model.onnx.data").read_bytes() == fixture.data
    repeat = CliRunner().invoke(
        cli.app,
        [
            "assemble-onnx",
            "--template",
            str(fixture.template),
            "--snapshot",
            str(fixture.snapshot),
            "--output-dir",
            str(models),
        ],
    )
    assert repeat.exit_code == 1 and "already exists" in repeat.output


def test_assembler_module_needs_no_third_party_imports() -> None:
    code = """
import sys
for name in ("numpy", "torch", "onnx", "onnxruntime", "safetensors"):
    sys.modules[name] = None
import gliner_runner.onnx_template
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "ok"
