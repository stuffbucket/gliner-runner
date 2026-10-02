from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"
EXAMPLE_PATTERN = re.compile(
    r"^### (.+?)\n.*?```python\n(model\.classify_text\(.*?\n\))\n```"
    r".*?Potential output:\n\n```text\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)
SUPPORTED_PROFILES = {
    ("cpu", "fp32"),
    ("mps", "fp16"),
    ("mps", "fp32"),
}


def parse_examples(card: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for name, source, expected_source in EXAMPLE_PATTERN.findall(card):
        expression = ast.parse(source).body[0]
        if not isinstance(expression, ast.Expr) or not isinstance(expression.value, ast.Call):
            raise ValueError(f"unexpected example shape: {name}")
        arguments = [ast.literal_eval(argument) for argument in expression.value.args]
        if len(arguments) != 2:
            raise ValueError(f"expected text and schema arguments: {name}")
        examples.append(
            {
                "name": name,
                "text": arguments[0],
                "schema": arguments[1],
                "potential_output": ast.literal_eval(expected_source),
            }
        )
    if not examples:
        raise ValueError("model card contains no classify_text examples")
    return examples


def synchronize(device: str, torch: Any) -> None:
    if device == "mps":
        torch.mps.synchronize()


def run_profile(
    model_path: Path,
    examples: list[dict[str, Any]],
    device: str,
    precision: str,
) -> dict[str, Any]:
    import torch
    from gliner2 import AutoExtractor

    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    load_started = time.perf_counter()
    model = AutoExtractor.from_pretrained(str(model_path)).to(device=device, dtype=dtype).eval()
    synchronize(device, torch)
    load_ms = (time.perf_counter() - load_started) * 1000
    results: list[dict[str, Any]] = []
    for example in examples:
        started = time.perf_counter()
        try:
            actual = model.classify_text(example["text"], example["schema"])
            synchronize(device, torch)
        except Exception as error:
            results.append(
                {
                    "name": example["name"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        results.append(
            {
                "name": example["name"],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "potential_output": example["potential_output"],
                "actual_output": actual,
                "matches_potential_output": actual == example["potential_output"],
            }
        )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "device": device,
        "precision": precision,
        "load_ms": load_ms,
        "examples": results,
        "successful_examples": sum("actual_output" in result for result in results),
        "potential_output_matches": sum(
            result.get("matches_potential_output", False) for result in results
        ),
    }


def build_comparison(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = profiles[0]
    baseline_outputs = {
        example["name"]: example.get("actual_output") for example in baseline["examples"]
    }
    comparisons: list[dict[str, Any]] = []
    for profile in profiles[1:]:
        matches = 0
        for example in profile["examples"]:
            if (
                "actual_output" in example
                and example["actual_output"] == baseline_outputs[example["name"]]
            ):
                matches += 1
        comparisons.append(
            {
                "baseline": f"{baseline['device']}/{baseline['precision']}",
                "candidate": f"{profile['device']}/{profile['precision']}",
                "exact_output_matches": matches,
                "examples": len(baseline["examples"]),
            }
        )
    return comparisons


def generate_report(result: dict[str, Any]) -> str:
    profile_rows = "\n".join(
        "| "
        f"{profile['device'].upper()} / {profile['precision'].upper()} | "
        f"{profile['load_ms']:.1f} | "
        f"{profile['successful_examples']}/{result['example_count']} | "
        f"{profile['potential_output_matches']}/{result['example_count']} |"
        for profile in result["profiles"]
    )
    comparison_rows = "\n".join(
        "| "
        f"{comparison['candidate'].upper()} | "
        f"{comparison['exact_output_matches']}/{comparison['examples']} |"
        for comparison in result["comparisons"]
    )
    baseline = result["profiles"][0]
    example_rows = "\n".join(
        "| "
        f"{example['name']} | "
        f"{'yes' if example.get('matches_potential_output') else 'no'} | "
        f"`{json.dumps(example.get('actual_output'), sort_keys=True)}` |"
        for example in baseline["examples"]
    )
    return f"""# Pinned model-card example characterization

This run parsed and executed every `model.classify_text` example in the model
card shipped with `{result["model_id"]}` at `{result["model_revision"]}`.
The card calls its displayed results “potential outputs,” so disagreement is
reported as characterization rather than a test failure.

## Environment

- Hardware: {result["hardware"]}
- OS: {result["os"]}
- Python {result["versions"]["python"]}; PyTorch {result["versions"]["torch"]};
  GLiNER2 {result["versions"]["gliner2"]}
- Model-card SHA-256: `{result["model_card_sha256"]}`
- Parsed examples: {result["example_count"]}

## Execution

| Profile | Load (ms) | Executed | Potential-output matches |
| --- | ---: | ---: | ---: |
{profile_rows}

## Device output parity

CPU/FP32 is the baseline.

| Candidate | Exact output matches |
| --- | ---: |
{comparison_rows}

## CPU/FP32 outputs

Input text and schemas remain in the pinned upstream card and are not duplicated
in this artifact.

| Example | Matches potential output | Actual output |
| --- | --- | --- |
{example_rows}
"""


def hardware_name() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def orchestrate(arguments: argparse.Namespace) -> None:
    model_path = arguments.model_path.expanduser().resolve()
    card_path = model_path / "README.md"
    if not card_path.is_file():
        raise FileNotFoundError(card_path)
    card = card_path.read_text(encoding="utf-8")
    examples = parse_examples(card)
    profiles: list[dict[str, Any]] = []
    for value in arguments.profile:
        device, precision = value.split("/", maxsplit=1)
        if (device, precision) not in SUPPORTED_PROFILES:
            raise ValueError(f"unsupported profile {value}; no fallback was attempted")
        profiles.append(run_profile(model_path, examples, device, precision))

    import psutil
    import torch

    result = {
        "format_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_card_sha256": hashlib.sha256(card.encode()).hexdigest(),
        "example_count": len(examples),
        "hardware": hardware_name(),
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gliner2": importlib.metadata.version("gliner2"),
        },
        "profiles": profiles,
        "comparisons": build_comparison(profiles),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "characterization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "REPORT.md").write_text(
        generate_report(result),
        encoding="utf-8",
    )
    failures = [
        example for profile in profiles for example in profile["examples"] if "error" in example
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} model-card example executions failed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--model-path", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Device/precision pair; repeat to compare profiles.",
    )
    return root


def main() -> None:
    arguments = parser().parse_args()
    if not arguments.profile:
        raise SystemExit("at least one --profile is required")
    orchestrate(arguments)


if __name__ == "__main__":
    main()
