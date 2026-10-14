"""Canonicalization of an exported graph against safetensors (needs numpy and onnx)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from gliner_runner.model_inventory import KNOWN_MODELS  # noqa: E402
from gliner_runner.onnx_prepare import (  # noqa: E402
    build_template,
    canonicalize_graph,
    check_assembler_equivalence,
    file_inventory,
)
from gliner_runner.onnx_template import (  # noqa: E402
    LARGE_TENSOR_BYTES,
    assemble_from_template,
    read_safetensors_index,
)

SIDE = 132  # 132 * 132 * 4 bytes > 64 KiB


def write_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    header: dict[str, Any] = {}
    offset = 0
    blobs = []
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def make_weights(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    weights = {
        "layer.a.weight": rng.standard_normal((SIDE, SIDE + 4), dtype="float32"),
        "layer.b.weight": rng.standard_normal((SIDE + 8, SIDE), dtype="float32"),
        "layer.c.bias": rng.standard_normal((300,), dtype="float32"),
        "unused": rng.standard_normal((3, 3), dtype="float32"),
    }
    write_safetensors(tmp_path / "model.safetensors", weights)
    return weights


def write_graph(
    directory: Path,
    initializers: list[Any],
    *,
    external: bool = False,
    nodes: list[Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"]), *(nodes or [])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    onnx.save_model(
        model,
        str(directory / "model.onnx"),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="exported.data",
        size_threshold=0,
    )


def initializers(weights: dict[str, Any]) -> list[Any]:
    return [
        numpy_helper.from_array(weights["layer.b.weight"].T.copy(), name="init.b_t"),
        numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), name="shape_const"),
        numpy_helper.from_array(weights["layer.a.weight"].copy(), name="init.a"),
        numpy_helper.from_array(weights["layer.c.bias"].copy(), name="init.c"),
        numpy_helper.from_array(np.array([0.5, 0.25], dtype=np.float32), name="small_unmatched"),
    ]


@pytest.mark.parametrize("external", [False, True])
def test_canonicalization_matches_and_lays_out_tensors(tmp_path: Path, external: bool) -> None:
    weights = make_weights(tmp_path)
    export = tmp_path / "export"
    write_graph(export, initializers(weights), external=external)
    out = tmp_path / "out"

    result = canonicalize_graph(
        exported_dir=export, weights=tmp_path / "model.safetensors", output_dir=out
    )

    assert [(item["name"], item["source"], item["transform"]) for item in result.tensors] == [
        ("init.b_t", "layer.b.weight", "transpose"),
        ("init.a", "layer.a.weight", "identity"),
        ("init.c", "layer.c.bias", "identity"),
    ]
    first = result.tensors[0]
    assert first["offset"] == 0
    assert first["source_shape"] == [SIDE + 8, SIDE] and first["shape"] == [SIDE, SIDE + 8]
    for previous, current in zip(result.tensors, result.tensors[1:], strict=False):
        end = previous["offset"] + previous["length"]
        assert current["offset"] == (end + 63) // 64 * 64
    data = (out / "model.onnx.data").read_bytes()
    assert len(data) == result.size
    last = result.tensors[-1]
    assert result.size == last["offset"] + last["length"]
    for item in result.tensors:
        expected = weights[item["source"]]
        if item["transform"] == "transpose":
            expected = expected.T
        raw = data[item["offset"] : item["offset"] + item["length"]]
        assert raw == np.ascontiguousarray(expected, dtype="<f4").tobytes()

    model = onnx.load(str(out / "model.onnx"), load_external_data=False)
    by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    assert [t.name for t in model.graph.initializer] == [
        "init.b_t",
        "shape_const",
        "init.a",
        "init.c",
        "small_unmatched",
    ]
    for item in result.tensors:
        values = {entry.key: entry.value for entry in by_name[item["name"]].external_data}
        assert values == {
            "location": "model.onnx.data",
            "offset": str(item["offset"]),
            "length": str(item["length"]),
        }
        assert not by_name[item["name"]].raw_data
    for name in ("shape_const", "small_unmatched"):
        assert not by_name[name].external_data and by_name[name].raw_data
    loaded = onnx.load(str(out / "model.onnx"))
    assert np.array_equal(
        numpy_helper.to_array(next(t for t in loaded.graph.initializer if t.name == "init.a")),
        weights["layer.a.weight"],
    )
    onnx.checker.check_model(str(out / "model.onnx"))


def test_unmatched_large_initializer_is_a_hard_failure(tmp_path: Path) -> None:
    weights = make_weights(tmp_path)
    assert weights["layer.a.weight"].nbytes >= LARGE_TENSOR_BYTES
    altered = weights["layer.a.weight"].copy()
    altered[0, 0] += 1.0
    export = tmp_path / "export"
    write_graph(export, [numpy_helper.from_array(altered, name="init.altered")])
    with pytest.raises(ValueError, match=r"do not match.*init\.altered"):
        canonicalize_graph(
            exported_dir=export, weights=tmp_path / "model.safetensors", output_dir=tmp_path / "o"
        )


def test_large_int_initializer_and_nested_constants_are_rejected(tmp_path: Path) -> None:
    weights = make_weights(tmp_path)
    export = tmp_path / "export"
    write_graph(
        export,
        [numpy_helper.from_array(np.zeros(LARGE_TENSOR_BYTES // 8, dtype=np.int64), name="ids")],
    )
    with pytest.raises(ValueError, match="do not match"):
        canonicalize_graph(
            exported_dir=export, weights=tmp_path / "model.safetensors", output_dir=tmp_path / "o"
        )
    big = numpy_helper.from_array(weights["layer.a.weight"].copy(), name="const")
    node = helper.make_node("Constant", [], ["c"], value=big)
    write_graph(tmp_path / "export2", initializers(weights), nodes=[node])
    with pytest.raises(ValueError, match="too large to template"):
        canonicalize_graph(
            exported_dir=tmp_path / "export2",
            weights=tmp_path / "model.safetensors",
            output_dir=tmp_path / "o2",
        )


def test_template_assembles_identically_to_the_canonical_artifact(tmp_path: Path) -> None:
    """Build template + manifest for a synthetic artifact and rebuild it with the assembler."""
    weights = make_weights(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (tmp_path / "model.safetensors").rename(snapshot / "model.safetensors")
    (snapshot / "tokenizer.json").write_text('{"model": {}}')
    (snapshot / "tokenizer_config.json").write_text("{}")
    export = tmp_path / "export"
    write_graph(export, initializers(weights))
    artifact = tmp_path / "artifact"
    spec = KNOWN_MODELS[0]
    canonical = canonicalize_graph(
        exported_dir=export, weights=snapshot / "model.safetensors", output_dir=artifact
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (artifact / name).write_bytes((snapshot / name).read_bytes())
    (artifact / "gliner2.json").write_text('{"kind": "pre"}\n')
    manifest = {
        "format_version": 1,
        "model_id": spec.model_id,
        "repository": spec.repository,
        "revision": spec.pinned_revision,
        "graph": {"path": "model.onnx", "external_data": ["model.onnx.data"]},
        "files": file_inventory(artifact),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    stage = tmp_path / "stage"
    template = build_template(
        spec=spec,
        revision=spec.pinned_revision,
        snapshot=snapshot,
        artifact=artifact,
        canonical=canonical,
        stage=stage,
    )
    assert [item["path"] for item in template["outputs"]] == [
        "model.onnx",
        "model.onnx.data",
        "gliner2.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert (stage / "graph.onnx").read_bytes() == (artifact / "model.onnx").read_bytes()
    assert read_safetensors_index(snapshot / "model.safetensors").tensors
    summary = check_assembler_equivalence(
        template_dir=stage, snapshot=snapshot, artifact=artifact, scratch_parent=tmp_path
    )
    assert summary["files"] == 6
    rebuilt = assemble_from_template(stage, snapshot, tmp_path / "rebuilt")
    assert file_inventory(rebuilt, exclude=()) == file_inventory(artifact, exclude=())
    # A corrupted artifact must be detected by the equivalence check.
    (artifact / "model.onnx.data").write_bytes((artifact / "model.onnx.data").read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="differs"):
        check_assembler_equivalence(
            template_dir=stage, snapshot=snapshot, artifact=artifact, scratch_parent=tmp_path
        )
