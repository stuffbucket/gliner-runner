#!/bin/sh
# Install the GLiNER Runner ONNX runner (no Python required).
#
#   curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh
#
# Environment:
#   GLINER_RUNNER_VERSION        X.Y.Z or X.Y.Z-rc.N to pin (default: latest stable release)
#   GLINER_RUNNER_HOME           install root (default: $HOME/.local/share/gliner-runner)
#   GLINER_RUNNER_BIN_DIR        where the gliner-runner and gliner-runner-onnx symlinks go (default: $HOME/.local/bin)
#   GLINER_RUNNER_DOWNLOAD_BASE  release URL prefix (default: https://github.com/stuffbucket/gliner-runner/releases)
#
# Supported: macOS arm64, Linux x64, Linux arm64. Safe to re-run.
set -eu

BASE=${GLINER_RUNNER_DOWNLOAD_BASE:-https://github.com/stuffbucket/gliner-runner/releases}
BASE=${BASE%/}
NAME=gliner-runner-onnx # archive name and the original command name
CMD=gliner-runner       # the command users run; the archive provides both launchers

say() { printf '%s\n' "$*"; }
die() {
  printf 'install.sh: error: %s\n' "$*" >&2
  exit 1
}

# --- platform -------------------------------------------------------------
case "$(uname -s)" in
  Darwin) os=darwin ;;
  Linux) os=linux ;;
  *) die "unsupported OS: $(uname -s). Supported: macOS arm64, Linux x64, Linux arm64." ;;
esac
case "$(uname -m)" in
  arm64 | aarch64) arch=arm64 ;;
  x86_64 | amd64) arch=x64 ;;
  *) die "unsupported CPU architecture: $(uname -m)." ;;
esac
if [ "$os" = darwin ] && [ "$arch" != arm64 ]; then
  die "Intel Macs are not supported; macOS requires Apple Silicon (arm64)."
fi
platform="$os-$arch"

# --- tools ----------------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL --retry 2 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
else
  die "curl or wget is required."
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
else
  die "sha256sum or shasum is required."
fi
command -v tar >/dev/null 2>&1 || die "tar is required."

home_dir=${GLINER_RUNNER_HOME:-}
bin_dir=${GLINER_RUNNER_BIN_DIR:-}
if [ -z "$home_dir" ] || [ -z "$bin_dir" ]; then
  [ -n "${HOME:-}" ] || die "HOME is not set; set GLINER_RUNNER_HOME and GLINER_RUNNER_BIN_DIR."
fi
[ -n "$home_dir" ] || home_dir=$HOME/.local/share/gliner-runner
[ -n "$bin_dir" ] || bin_dir=$HOME/.local/bin

tmp=$(mktemp -d "${TMPDIR:-/tmp}/gliner-runner-install.XXXXXX") || die "cannot create a temporary directory."
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# --- version --------------------------------------------------------------
valid_version() {
  printf '%s' "$1" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$'
}

expected_sha=
if [ -n "${GLINER_RUNNER_VERSION:-}" ]; then
  version=${GLINER_RUNNER_VERSION#v}
  valid_version "$version" || die "GLINER_RUNNER_VERSION must look like 1.2.3 or 1.2.3-rc.1 (got: $GLINER_RUNNER_VERSION)."
  release_url=$BASE/download/v$version
else
  say "Resolving the latest stable release..."
  fetch "$BASE/latest/download/latest.json" "$tmp/latest.json" ||
    die "could not download $BASE/latest/download/latest.json."
  flat=$(tr -d '\n\r' <"$tmp/latest.json")
  version=$(printf '%s' "$flat" | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  valid_version "$version" || die "latest.json has no usable version."
  release_url=$BASE/download/v$version
  entry=$(printf '%s' "$flat" | sed -n 's/.*"'"$platform"'"[[:space:]]*:[[:space:]]*{\([^}]*\)}.*/\1/p')
  [ -n "$entry" ] || die "release $version has no build for $platform."
  expected_sha=$(printf '%s' "$entry" | sed -n 's/.*"sha256"[[:space:]]*:[[:space:]]*"\([0-9a-fA-F]*\)".*/\1/p')
fi

archive_name=$NAME-$version-$platform
target=$home_dir/$version

installed() { [ -x "$target/bin/$CMD" ] && [ -x "$target/bin/$NAME" ]; }

# --- download, verify, install --------------------------------------------
if installed; then
  say "$CMD $version is already installed in $target."
else
  say "Downloading $archive_name.tar.gz..."
  fetch "$release_url/$archive_name.tar.gz" "$tmp/archive.tar.gz" ||
    die "could not download $release_url/$archive_name.tar.gz (is $version published for $platform?)."
  fetch "$release_url/$archive_name.tar.gz.sha256" "$tmp/archive.sha256" ||
    die "could not download $release_url/$archive_name.tar.gz.sha256."

  want=$(cut -d ' ' -f 1 <"$tmp/archive.sha256" | head -n 1 | tr 'A-F' 'a-f')
  printf '%s' "$want" | grep -Eq '^[0-9a-f]{64}$' || die "malformed checksum file."
  if [ -n "$expected_sha" ]; then
    [ "$want" = "$(printf '%s' "$expected_sha" | tr 'A-F' 'a-f')" ] ||
      die "checksum file does not match latest.json; refusing to install."
  fi
  got=$(sha256 "$tmp/archive.tar.gz")
  [ "$got" = "$want" ] || die "checksum mismatch for $archive_name.tar.gz (expected $want, got $got)."
  say "Checksum verified."

  mkdir "$tmp/extract"
  tar -xzf "$tmp/archive.tar.gz" -C "$tmp/extract" || die "could not extract the archive."
  [ -x "$tmp/extract/$archive_name/bin/$CMD" ] || die "archive does not contain $archive_name/bin/$CMD."
  [ -x "$tmp/extract/$archive_name/bin/$NAME" ] || die "archive does not contain $archive_name/bin/$NAME."

  mkdir -p "$home_dir"
  staging=$home_dir/.$version.$$
  rm -rf "$staging"
  mv "$tmp/extract/$archive_name" "$staging"
  rm -rf "$target"
  mv "$staging" "$target"
  say "Installed $CMD $version to $target."
fi

mkdir -p "$bin_dir"
for command in "$CMD" "$NAME"; do
  ln -sfn "$target/bin/$command" "$bin_dir/$command"
  say "Linked $bin_dir/$command."
done

case ":${PATH:-}:" in
  *":$bin_dir:"*) path_hint= ;;
  *) path_hint=yes ;;
esac

say ""
say "Next steps:"
if [ -n "$path_hint" ]; then
  say "  1. Add $bin_dir to your PATH, for example:"
  say "       export PATH=\"$bin_dir:\$PATH\""
  say "  2. Install a model (downloads about 1.8 GiB):"
else
  say "  1. Install a model (downloads about 1.8 GiB):"
fi
say "       $CMD models install decide-340m"
say "  Then try some examples:"
say "       $CMD demo"
say "  Or start the local service on http://127.0.0.1:8090:"
say "       $CMD serve"
