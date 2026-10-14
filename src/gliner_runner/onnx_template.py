"""ONNX artifact template: safetensors parsing, layout math, and a reference assembler.

The Python-free Node runner downloads a pinned Hugging Face snapshot and assembles the
production ONNX artifact itself from a tracked *template package*. This module defines the
template schema (see ``dist/port-spec/onnx-template-contract.md``) and a pure-stdlib reference
assembler that rebuilds the canonical artifact byte for byte. It deliberately imports neither
NumPy, PyTorch, nor ONNX so that it stays independent of the build-time converter that
produced the template and can run anywhere a plain Python interpreter exists.

Canonical artifact (``<output>/onnx/<owner>--<name>/<revision>/``)::

    model.onnx             template graph, copied verbatim
    model.onnx.data        weights in ``template.data.tensors`` order, 64-byte aligned
    tokenizer*.json etc.   copied verbatim from the snapshot
    gliner2.json           copied verbatim from the template
    manifest.json          copied verbatim from the template, written last
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATE_FORMAT_VERSION = 1
TEMPLATE_KIND = "gliner-runner-onnx-template"
TEMPLATE_FILE = "template.json"
TEMPLATE_GRAPH_FILE = "graph.onnx"
GRAPH_OUTPUT = "model.onnx"
DATA_OUTPUT = "model.onnx.data"
PREPROCESSING_OUTPUT = "gliner2.json"
MANIFEST_OUTPUT = "manifest.json"
WEIGHTS_FILE = "model.safetensors"
DATA_ALIGNMENT = 64
LARGE_TENSOR_BYTES = 64 * 1024
FLOAT32_BYTES = 4
TRANSFORMS = ("identity", "transpose")
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_CHUNK_BYTES = 8 * 1024 * 1024
_TRANSPOSE_BLOCK_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SOURCE_ROLES = ("weights", "tokenizer")
_REQUIRED_OUTPUTS = (GRAPH_OUTPUT, DATA_OUTPUT, PREPROCESSING_OUTPUT)


# --------------------------------------------------------------------------------------
# Safetensors header parsing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetensorsTensor:
    """One tensor entry with absolute byte offsets inside the safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def element_count(self) -> int:
        return shape_elements(self.shape)


@dataclass(frozen=True)
class SafetensorsIndex:
    file_size: int
    data_start: int
    tensors: Mapping[str, SafetensorsTensor]


def shape_elements(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"safetensors header repeats key {key!r}")
        result[key] = value
    return result


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_safetensors_index(path: Path) -> SafetensorsIndex:
    """Parse and structurally validate a safetensors header without reading tensor data."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("safetensors file is truncated")
            (header_size,) = struct.unpack("<Q", prefix)
            if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("safetensors header size is invalid")
            if 8 + header_size > file_size:
                raise ValueError("safetensors header is truncated")
            raw = stream.read(header_size)
    except OSError as error:
        raise ValueError(f"safetensors file is unreadable: {path}") from error
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("safetensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    data_start = 8 + header_size
    data_size = file_size - data_start
    tensors: dict[str, SafetensorsTensor] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"safetensors entry {name!r} must be an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"safetensors entry {name!r} has no dtype")
        if not isinstance(shape, list) or not all(_is_int(item) and item >= 0 for item in shape):
            raise ValueError(f"safetensors entry {name!r} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(item) for item in offsets)
        ):
            raise ValueError(f"safetensors entry {name!r} has invalid data_offsets")
        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            raise ValueError(f"safetensors entry {name!r} lies outside the data section")
        if dtype == "F32" and end - begin != FLOAT32_BYTES * shape_elements(shape):
            raise ValueError(f"safetensors entry {name!r} size does not match its F32 shape")
        tensors[name] = SafetensorsTensor(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            start=data_start + begin,
            end=data_start + end,
        )
    return SafetensorsIndex(file_size=file_size, data_start=data_start, tensors=tensors)


# --------------------------------------------------------------------------------------
# Layout and transform math
# --------------------------------------------------------------------------------------


def align_up(value: int, alignment: int = DATA_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment arguments must be non-negative and positive")
    return (value + alignment - 1) // alignment * alignment


def plan_layout(lengths: Sequence[int], alignment: int = DATA_ALIGNMENT) -> tuple[list[int], int]:
    """Return per-tensor offsets and the total size for tensors written back to back.

    Each offset is the running end rounded up to ``alignment``; there is no trailing padding.
    """
    offsets: list[int] = []
    end = 0
    for length in lengths:
        if length < 0:
            raise ValueError("tensor lengths must be non-negative")
        offset = align_up(end, alignment)
        offsets.append(offset)
        end = offset + length
    return offsets, end


def transposed_shape(shape: Sequence[int]) -> list[int]:
    if len(shape) != 2:
        raise ValueError("transpose requires a 2-D tensor")
    return [shape[1], shape[0]]


def transpose_blocks(
    source: memoryview, rows: int, cols: int, block_bytes: int = _TRANSPOSE_BLOCK_BYTES
) -> Iterator[bytes]:
    """Yield the C-order bytes of the transpose of a row-major ``rows x cols`` 4-byte matrix.

    Only 4-byte words are moved, so the element values and host byte order do not matter.
    Output is produced a bounded block of output rows (source columns) at a time.
    """
    if rows < 0 or cols < 0 or len(source) != rows * cols * FLOAT32_BYTES:
        raise ValueError("transpose source size does not match its shape")
    if rows == 0 or cols == 0:
        return
    words = source.cast("B").cast("I")
    block_columns = max(1, block_bytes // (rows * FLOAT32_BYTES))
    for first in range(0, cols, block_columns):
        last = min(cols, first + block_columns)
        block = bytearray((last - first) * rows * FLOAT32_BYTES)
        view = memoryview(block).cast("I")
        for index, column in enumerate(range(first, last)):
            view[index * rows : (index + 1) * rows] = words[column::cols]
        yield bytes(block)


# --------------------------------------------------------------------------------------
# Template schema
# --------------------------------------------------------------------------------------


def _fail(message: str) -> ValueError:
    return ValueError(f"invalid ONNX template: {message}")


def _expect_dict(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(f"{what} must be an object")
    return value


def _expect_list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(f"{what} must be an array")
    return value


def _expect_keys(value: Mapping[str, Any], keys: set[str], what: str) -> None:
    if set(value) != keys:
        raise _fail(f"{what} must have exactly the keys {sorted(keys)}")


def _expect_name(value: object, what: str) -> str:
    if not isinstance(value, str) or _FILE_NAME.fullmatch(value) is None:
        raise _fail(f"{what} must be a plain file name")
    return value


def _expect_size(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(f"{what} must be a non-negative integer")
    return value


def _expect_sha(value: object, what: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{what} must be a lowercase SHA-256 digest")
    return value


def validate_template(template: object) -> dict[str, Any]:
    """Validate the structure and internal consistency of ``template.json``.

    Checks every field, tensor offsets against the layout rules, and that ``outputs`` covers
    exactly the artifact files the assembler will produce. Returns the template as a dict.
    """
    root = _expect_dict(template, "template")
    _expect_keys(
        root,
        {
            "format_version",
            "kind",
            "model_id",
            "repository",
            "revision",
            "source_files",
            "template_files",
            "data",
            "outputs",
        },
        "template",
    )
    if root["format_version"] != TEMPLATE_FORMAT_VERSION or isinstance(
        root["format_version"], bool
    ):
        raise _fail("unsupported format_version")
    if root["kind"] != TEMPLATE_KIND:
        raise _fail("unexpected kind")
    if not isinstance(root["model_id"], str) or not root["model_id"]:
        raise _fail("model_id must be a non-empty string")
    repository = root["repository"]
    if not isinstance(repository, str) or repository.count("/") != 1:
        raise _fail("repository must be <owner>/<name>")
    if not all(_FILE_NAME.fullmatch(part) for part in repository.split("/")):
        raise _fail("repository must be <owner>/<name>")
    if not isinstance(root["revision"], str) or _REVISION.fullmatch(root["revision"]) is None:
        raise _fail("revision must be a 40-character lowercase commit SHA")

    sources: dict[str, dict[str, Any]] = {}
    for item in _expect_list(root["source_files"], "source_files"):
        entry = _expect_dict(item, "source_files entry")
        _expect_keys(entry, {"path", "size", "sha256", "role"}, "source_files entry")
        path = _expect_name(entry["path"], "source_files path")
        _expect_size(entry["size"], "source_files size")
        _expect_sha(entry["sha256"], "source_files sha256")
        if entry["role"] not in _SOURCE_ROLES:
            raise _fail(f"source_files role must be one of {list(_SOURCE_ROLES)}")
        if path in sources:
            raise _fail(f"source_files repeats {path}")
        sources[path] = entry
    weights = [path for path, entry in sources.items() if entry["role"] == "weights"]
    if weights != [WEIGHTS_FILE]:
        raise _fail(f"source_files must contain exactly one weights file, {WEIGHTS_FILE}")
    tokenizers = [path for path, entry in sources.items() if entry["role"] == "tokenizer"]
    if not tokenizers:
        raise _fail("source_files must contain tokenizer files")

    template_outputs: dict[str, str] = {}
    template_names: set[str] = set()
    for item in _expect_list(root["template_files"], "template_files"):
        entry = _expect_dict(item, "template_files entry")
        _expect_keys(entry, {"path", "size", "sha256", "output"}, "template_files entry")
        path = _expect_name(entry["path"], "template_files path")
        _expect_size(entry["size"], "template_files size")
        _expect_sha(entry["sha256"], "template_files sha256")
        output = _expect_name(entry["output"], "template_files output")
        if path in template_names or output in template_outputs.values():
            raise _fail("template_files repeats a path or output")
        template_names.add(path)
        template_outputs[path] = output
    expected_template = {
        TEMPLATE_GRAPH_FILE: GRAPH_OUTPUT,
        PREPROCESSING_OUTPUT: PREPROCESSING_OUTPUT,
        MANIFEST_OUTPUT: MANIFEST_OUTPUT,
    }
    if template_outputs != expected_template:
        raise _fail(f"template_files must map exactly {expected_template}")

    data = _expect_dict(root["data"], "data")
    _expect_keys(data, {"path", "alignment", "size", "sha256", "tensors"}, "data")
    if data["path"] != DATA_OUTPUT:
        raise _fail(f"data.path must be {DATA_OUTPUT}")
    if data["alignment"] != DATA_ALIGNMENT or isinstance(data["alignment"], bool):
        raise _fail(f"data.alignment must be {DATA_ALIGNMENT}")
    data_size = _expect_size(data["size"], "data.size")
    _expect_sha(data["sha256"], "data.sha256")
    tensors = _expect_list(data["tensors"], "data.tensors")
    if not tensors:
        raise _fail("data.tensors must not be empty")
    lengths: list[int] = []
    names: set[str] = set()
    for item in tensors:
        entry = _expect_dict(item, "data.tensors entry")
        _expect_keys(
            entry,
            {
                "name",
                "source",
                "transform",
                "dtype",
                "source_shape",
                "shape",
                "offset",
                "length",
            },
            "data.tensors entry",
        )
        for field in ("name", "source"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise _fail(f"data.tensors {field} must be a non-empty string")
        if entry["name"] in names:
            raise _fail(f"data.tensors repeats {entry['name']}")
        names.add(entry["name"])
        if entry["transform"] not in TRANSFORMS:
            raise _fail(f"data.tensors transform must be one of {list(TRANSFORMS)}")
        if entry["dtype"] != "float32":
            raise _fail("data.tensors dtype must be float32")
        shapes: dict[str, list[int]] = {}
        for field in ("source_shape", "shape"):
            value = _expect_list(entry[field], f"data.tensors {field}")
            if not value or not all(_is_int(dim) and dim >= 0 for dim in value):
                raise _fail(f"data.tensors {field} must list non-negative integers")
            shapes[field] = value
        if entry["transform"] == "identity":
            if shapes["shape"] != shapes["source_shape"]:
                raise _fail(f"identity tensor {entry['name']} must keep its shape")
        else:
            if len(shapes["source_shape"]) != 2 or shapes["shape"] != transposed_shape(
                shapes["source_shape"]
            ):
                raise _fail(f"transpose tensor {entry['name']} must swap a 2-D shape")
        length = _expect_size(entry["length"], "data.tensors length")
        if length != FLOAT32_BYTES * shape_elements(shapes["shape"]):
            raise _fail(f"tensor {entry['name']} length does not match its shape")
        _expect_size(entry["offset"], "data.tensors offset")
        lengths.append(length)
    offsets, total = plan_layout(lengths)
    if [entry["offset"] for entry in tensors] != offsets:
        raise _fail("data.tensors offsets do not follow the 64-byte alignment layout")
    if data_size != total:
        raise _fail("data.size does not equal the end of the last tensor")

    outputs: dict[str, dict[str, Any]] = {}
    for item in _expect_list(root["outputs"], "outputs"):
        entry = _expect_dict(item, "outputs entry")
        _expect_keys(entry, {"path", "size", "sha256"}, "outputs entry")
        path = _expect_name(entry["path"], "outputs path")
        _expect_size(entry["size"], "outputs size")
        _expect_sha(entry["sha256"], "outputs sha256")
        if path in outputs:
            raise _fail(f"outputs repeats {path}")
        outputs[path] = entry
    if set(outputs) != {*_REQUIRED_OUTPUTS, *tokenizers}:
        raise _fail("outputs must list the graph, sidecar, preprocessing, and tokenizer files")
    for path in tokenizers:
        if outputs[path]["size"] != sources[path]["size"]:
            raise _fail(f"outputs {path} must equal its source file")
        if outputs[path]["sha256"] != sources[path]["sha256"]:
            raise _fail(f"outputs {path} must equal its source file")
    by_template = {entry["output"]: entry for entry in root["template_files"]}
    for path in (GRAPH_OUTPUT, PREPROCESSING_OUTPUT):
        if (outputs[path]["size"], outputs[path]["sha256"]) != (
            by_template[path]["size"],
            by_template[path]["sha256"],
        ):
            raise _fail(f"outputs {path} must equal its template file")
    if (outputs[DATA_OUTPUT]["size"], outputs[DATA_OUTPUT]["sha256"]) != (
        data["size"],
        data["sha256"],
    ):
        raise _fail("outputs model.onnx.data must equal the data section")
    return root


def validate_template_manifest(template: Mapping[str, Any], manifest: object) -> None:
    """Require the template manifest to inventory exactly ``template.outputs``."""
    if not isinstance(manifest, dict):
        raise _fail("manifest.json must be an object")
    graph = manifest.get("graph")
    if not isinstance(graph, dict) or graph.get("external_data") != [DATA_OUTPUT]:
        raise _fail("manifest graph.external_data must be [model.onnx.data]")
    if (
        manifest.get("repository") != template["repository"]
        or manifest.get("revision") != template["revision"]
        or manifest.get("model_id") != template["model_id"]
    ):
        raise _fail("manifest identity does not match the template")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise _fail("manifest files must be an array")
    listed = {
        item.get("path"): (item.get("size"), item.get("sha256"))
        for item in files
        if isinstance(item, dict)
    }
    expected = {item["path"]: (item["size"], item["sha256"]) for item in template["outputs"]}
    if len(listed) != len(files) or listed != expected:
        raise _fail("manifest files do not equal template outputs")


# --------------------------------------------------------------------------------------
# Reference assembler
# --------------------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required file is missing or not a regular file: {path}")


def _verify_file(path: Path, size: int, sha256: str, what: str) -> None:
    _require_regular_file(path)
    actual_size, actual_sha = _sha256_of_file(path)
    if actual_size != size:
        raise ValueError(f"{what} size mismatch: {path.name} is {actual_size}, expected {size}")
    if actual_sha != sha256:
        raise ValueError(f"{what} checksum mismatch: {path.name}")


def load_template(template_dir: Path) -> dict[str, Any]:
    """Load, validate, and verify a template package directory."""
    root = template_dir.expanduser().resolve()
    manifest_path = root / TEMPLATE_FILE
    _require_regular_file(manifest_path)
    try:
        template = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ONNX template is unreadable: {manifest_path}") from error
    validated = validate_template(template)
    for entry in validated["template_files"]:
        _verify_file(root / entry["path"], entry["size"], entry["sha256"], "template file")
    try:
        manifest = json.loads((root / MANIFEST_OUTPUT).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ONNX template manifest.json is unreadable") from error
    validate_template_manifest(validated, manifest)
    return validated


def template_artifact_path(output_dir: Path, template: Mapping[str, Any]) -> Path:
    """Return ``<output-dir>/onnx/<owner>--<name>/<revision>`` for a validated template."""
    return (
        output_dir.expanduser().resolve()
        / "onnx"
        / str(template["repository"]).replace("/", "--")
        / str(template["revision"])
    )


def _write_data_file(
    template: Mapping[str, Any], weights_path: Path, index: SafetensorsIndex, target: Path
) -> None:
    data = template["data"]
    digest = hashlib.sha256()
    position = 0
    with weights_path.open("rb") as weights_stream, target.open("wb") as out:
        mapped = mmap.mmap(weights_stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            source = memoryview(mapped)
            try:
                for entry in data["tensors"]:
                    tensor = index.tensors.get(entry["source"])
                    if tensor is None:
                        raise ValueError(f"weights file has no tensor {entry['source']!r}")
                    if tensor.dtype != "F32" or list(tensor.shape) != entry["source_shape"]:
                        raise ValueError(
                            f"weights tensor {tensor.name!r} does not match the template"
                        )
                    padding = entry["offset"] - position
                    if padding:
                        out.write(bytes(padding))
                        digest.update(bytes(padding))
                        position += padding
                    raw = source[tensor.start : tensor.end]
                    if entry["transform"] == "identity":
                        chunks: Iterator[bytes] = (
                            bytes(raw[start : start + _CHUNK_BYTES])
                            for start in range(0, len(raw), _CHUNK_BYTES)
                        )
                    else:
                        chunks = transpose_blocks(raw, *entry["source_shape"])
                    written = 0
                    for chunk in chunks:
                        out.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    if written != entry["length"]:
                        raise ValueError(f"tensor {entry['name']!r} has an unexpected length")
                    position += written
                    del raw, chunks
            finally:
                source.release()
        finally:
            mapped.close()
    if position != data["size"]:
        raise ValueError("assembled data size does not match the template")
    if digest.hexdigest() != data["sha256"]:
        raise ValueError("assembled data checksum does not match the template")


def assemble_from_template(template_dir: Path, snapshot_dir: Path, output_dir: Path) -> Path:
    """Rebuild the canonical ONNX artifact from a template package and a pinned snapshot.

    ``output_dir`` is the artifact directory itself (it must not exist). The artifact is built
    in a sibling staging directory, every output is checked against ``template.outputs``, the
    manifest is written last, and only then is the directory renamed into place. Any
    missing, short, or altered input fails closed and leaves nothing behind.
    """
    template_root = template_dir.expanduser().resolve()
    snapshot = snapshot_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"ONNX artifact path already exists: {target}")
    if target == snapshot or target.is_relative_to(snapshot) or snapshot.is_relative_to(target):
        raise ValueError("assembly output and model snapshot must not contain each other")
    template = load_template(template_root)

    sources = {entry["path"]: entry for entry in template["source_files"]}
    for entry in sources.values():
        _verify_file(snapshot / entry["path"], entry["size"], entry["sha256"], "snapshot file")
    if sys.byteorder != "little":
        raise RuntimeError("the reference assembler requires a little-endian host")
    index = read_safetensors_index(snapshot / WEIGHTS_FILE)

    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{template['revision'][:12]}.assemble-", dir=target.parent)
    )
    try:
        for entry in template["template_files"]:
            if entry["output"] != MANIFEST_OUTPUT:
                shutil.copyfile(template_root / entry["path"], work / entry["output"])
        for path, entry in sources.items():
            if entry["role"] == "tokenizer":
                shutil.copyfile(snapshot / path, work / path)
        _write_data_file(template, snapshot / WEIGHTS_FILE, index, work / DATA_OUTPUT)
        for output in template["outputs"]:
            _verify_file(work / output["path"], output["size"], output["sha256"], "output")
        extra = sorted(
            item.name
            for item in work.iterdir()
            if item.name not in {entry["path"] for entry in template["outputs"]}
        )
        if extra:
            raise ValueError(f"unexpected assembled file: {extra[0]}")
        manifest_entry = next(
            entry for entry in template["template_files"] if entry["output"] == MANIFEST_OUTPUT
        )
        shutil.copyfile(template_root / manifest_entry["path"], work / MANIFEST_OUTPUT)
        _verify_file(
            work / MANIFEST_OUTPUT, manifest_entry["size"], manifest_entry["sha256"], "output"
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"ONNX artifact path already exists: {target}")
        os.chmod(work, 0o755)
        work.rename(target)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return target
