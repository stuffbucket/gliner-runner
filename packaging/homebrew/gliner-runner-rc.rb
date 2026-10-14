# frozen_string_literal: true

# Homebrew formula for gliner-runner release candidates (self-contained ONNX runtime, no Python).
#
# This is the template; the rendered copy is attached to every release candidate (GitHub
# prerelease) as `gliner-runner-rc.rb` and synced into stuffbucket/homebrew-tap. Placeholder tokens
# are replaced from the release's archive digests by scripts/render-homebrew-formula.mjs. Do not
# commit real versions or digests to this template.

class GlinerRunnerRc < Formula
  desc "GLiNER2 inference runner, release candidate for testing (ONNX, no Python)"
  homepage "https://github.com/stuffbucket/gliner-runner"
  license "Apache-2.0"

  # Prereleases are not served as the latest release and livecheck's GitHub strategies skip them, so
  # read the vX.Y.Z-rc.N tags from git.
  livecheck do
    url :homepage
    regex(/^v?(\d+\.\d+\.\d+-rc\.\d+)$/i)
    strategy :git
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

  conflicts_with "gliner-runner", because: "both install a `gliner-runner` executable"

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
