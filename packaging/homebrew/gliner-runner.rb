# frozen_string_literal: true

# Homebrew formula for gliner-runner (stable channel; self-contained ONNX runtime, no Python).
#
# This is the template; the rendered copy is attached to every stable GitHub release as
# `gliner-runner.rb` and synced into stuffbucket/homebrew-tap. Placeholder tokens are replaced from
# the release's archive digests by scripts/render-homebrew-formula.mjs. Do not commit real
# versions or digests to this template.

class GlinerRunner < Formula
  desc "Local GLiNER2 inference runner (self-contained ONNX runtime, no Python)"
  homepage "https://github.com/stuffbucket/gliner-runner"
  license "Apache-2.0"

  livecheck do
    url :stable
    strategy :github_latest
  end

  on_macos do
    on_arm do
      url "https://github.com/stuffbucket/gliner-runner/releases/download/vPLACEHOLDER_VERSION/gliner-runner-onnx-PLACEHOLDER_VERSION-darwin-arm64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_DARWIN_ARM64"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/stuffbucket/gliner-runner/releases/download/vPLACEHOLDER_VERSION/gliner-runner-onnx-PLACEHOLDER_VERSION-linux-x64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_LINUX_X64"
    end
    on_arm do
      url "https://github.com/stuffbucket/gliner-runner/releases/download/vPLACEHOLDER_VERSION/gliner-runner-onnx-PLACEHOLDER_VERSION-linux-arm64.tar.gz"
      sha256 "PLACEHOLDER_SHA256_LINUX_ARM64"
    end
  end

  conflicts_with "gliner-runner-rc", because: "both install a `gliner-runner` executable"

  def install
    # The archive bundles its own Node.js and ONNX Runtime. The launcher resolves symlinks to find
    # its root, so link it rather than wrapping it. Models are not bundled; install them with
    # `gliner-runner models install <id> --yes`.
    libexec.install Dir["*"]
    bin.install_symlink libexec/"bin/gliner-runner"
    bin.install_symlink libexec/"bin/gliner-runner-onnx"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/gliner-runner --version")
  end
end
