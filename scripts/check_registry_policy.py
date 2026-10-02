from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "dist",
    "node_modules",
    "__pycache__",
}
FORBIDDEN_NAMES = {
    ".npmrc",
    ".netrc",
    "pip.conf",
    "pip.ini",
    "credentials",
    "credentials.toml",
}
SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?:_authToken|_password|password|secret|access[_-]?token)"
    r"\s*[:=]\s*(?!<|example|redacted|\$\{)[^\s]+"
)
URL = re.compile(r"https?://[^\s,'\"}\]]+")
SENSITIVE_QUERY_KEYS = {"access_token", "api_key", "auth", "password", "secret", "token"}


def repository_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(ROOT).parts)
    ]


def check_forbidden_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES and relative not in {
            Path(".gitignore"),
            Path(".dockerignore"),
        }:
            errors.append(f"forbidden credential/config file: {relative}")
    return errors


def check_generated_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix not in {".json", ".lock", ".toml", ".yaml", ".yml"}:
            continue
        if path.name in {"package.json", "pyproject.toml", "mise.toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SENSITIVE_ASSIGNMENT.search(text):
            errors.append(f"possible credential assignment: {path.relative_to(ROOT)}")
        for match in URL.finditer(text):
            parsed = urlsplit(match.group())
            if parsed.username is not None or parsed.password is not None:
                errors.append(f"credential-bearing URL: {path.relative_to(ROOT)}")
            query_keys = {key.lower() for key, _ in parse_qsl(parsed.query)}
            if query_keys & SENSITIVE_QUERY_KEYS:
                errors.append(f"sensitive URL query: {path.relative_to(ROOT)}")
    return errors


def check_dockerfile() -> list[str]:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    errors: list[str] = []
    forbidden = {
        "COPY npmrc": r"(?im)^\s*COPY\b[^\n]*\.npmrc",
        "ADD npmrc": r"(?im)^\s*ADD\b[^\n]*\.npmrc",
        "registry credential ARG": (
            r"(?im)^\s*ARG\s+[A-Za-z0-9_]*(?:TOKEN|PASSWORD|SECRET|NPMRC|AUTH)"
        ),
        "registry credential ENV": (
            r"(?im)^\s*ENV\s+[A-Za-z0-9_]*(?:TOKEN|PASSWORD|SECRET|NPMRC|AUTH)"
        ),
    }
    for name, pattern in forbidden.items():
        if re.search(pattern, dockerfile):
            errors.append(f"unsafe Dockerfile pattern: {name}")
    required = {
        "npmrc BuildKit secret": (
            "--mount=type=secret,id=npmrc,target=/root/.npmrc,required=true"
        ),
        "pip BuildKit secret": (
            "--mount=type=secret,id=pip_config,target=/etc/pip.conf,required=false"
        ),
    }
    for name, value in required.items():
        if value not in dockerfile:
            errors.append(f"missing Dockerfile adapter: {name}")
    return errors


def check_repository_guards() -> list[str]:
    errors: list[str] = []
    required_ignores = {
        ".npmrc",
        ".netrc",
        ".cargo/credentials",
        ".cargo/credentials.toml",
        "pip.conf",
        "pip.ini",
    }
    for name in (".gitignore", ".dockerignore"):
        entries = {
            line.strip()
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for missing in sorted(required_ignores - entries):
            errors.append(f"{name} must exclude {missing}")
    lockfile = (ROOT / "pnpm-lock.yaml").read_text(encoding="utf-8")
    if "tarball:" in lockfile:
        errors.append("pnpm-lock.yaml must not persist registry tarball URLs")
    return errors


def main() -> None:
    files = repository_files()
    errors = [
        *check_forbidden_files(files),
        *check_generated_files(files),
        *check_dockerfile(),
        *check_repository_guards(),
    ]
    if errors:
        print("\n".join(f"registry policy: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("registry policy: ok")


if __name__ == "__main__":
    main()
