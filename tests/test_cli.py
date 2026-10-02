from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from typer.testing import CliRunner

from gliner_runner import cli
from gliner_runner.contracts import InferenceResponse, InferenceUsage, Timing


class FakeRuntime:
    def __init__(self, _config: object) -> None:
        self.closed = False

    async def infer(self, request: Any) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            backend=request.backend,
            precision=request.precision,
            output={"label": "useful"},
            timing=Timing(queue_ms=0, inference_ms=1),
            usage=InferenceUsage(input_tokens=17, output_tokens=0),
        )

    async def close(self) -> None:
        self.closed = True


def test_infer_cli_reads_stdin_and_emits_one_json_document(
    monkeypatch: Any, request_factory: Any
) -> None:
    monkeypatch.setattr(cli, "Runtime", FakeRuntime)
    request = request_factory()

    result = CliRunner().invoke(
        cli.app,
        ["infer"],
        input=request.model_dump_json(by_alias=True),
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "request_id": str(request.request_id),
        "model": request.model,
        "backend": "pytorch",
        "precision": "fp32",
        "output": {"label": "useful"},
        "error": None,
        "timing": {"queue_ms": 0.0, "inference_ms": 1.0},
        "usage": {"inputTokens": 17, "outputTokens": 0},
    }
    assert result.stdout.count("\n") == 1


def test_infer_cli_rejects_malformed_stdin_without_stdout() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gliner_runner.cli", "infer"],
        input="{",
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "validation error" in result.stderr.lower()


def test_serve_cli_starts_health_endpoint_and_stops_cleanly(tmp_path: Path) -> None:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gliner_runner.cli",
            "serve",
            "--port",
            str(port),
            "--manifest-directory",
            str(tmp_path / "manifests"),
            "--model-store",
            str(tmp_path / "models"),
            "--model-provider-directory",
            str(tmp_path / "providers"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        health = _wait_for_health(port, process)
    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10)

    assert health == {"status": "ok", "accepting_requests": True}
    assert process.returncode in {0, -signal.SIGTERM}, (
        f"stdout={stdout}\nstderr={stderr}"
    )
    assert "Application shutdown complete" in stderr


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[str]) -> dict[str, object]:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"server exited before readiness: {process.returncode}\n{stdout}\n{stderr}"
            )
        try:
            with urlopen(url, timeout=0.5) as response:
                return json.load(response)
        except (URLError, TimeoutError):
            time.sleep(0.05)
    raise AssertionError("server did not become healthy within 10 seconds")
