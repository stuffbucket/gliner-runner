// Installs the pinned browser packages into a cache folder outside the repository and imports them from there.
//
// The tool is not a pnpm workspace member on purpose: adding it would re-resolve the root pnpm-lock.yaml through
// the npm mirror, which writes sha1 hashes and tarball URLs that check:registry rejects (docs/registries.md). The
// versions are pinned in package.json, and npm checks each tarball against the registry's integrity hash.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pins = JSON.parse(readFileSync(join(here, "../package.json"), "utf8")).dependencies;

export const DEFAULT_DEPS_DIR = join(homedir(), ".cache/gliner-runner/site-review");

const installedVersion = (dir, name) => {
  const file = join(dir, "node_modules", name, "package.json");
  return existsSync(file) ? JSON.parse(readFileSync(file, "utf8")).version : null;
};

/** Makes sure every pinned package is in `dir`, then returns { chromium, AxeBuilder }. */
export async function loadBrowserTools(dir = DEFAULT_DEPS_DIR) {
  const stale = Object.entries(pins).filter(([name, version]) => installedVersion(dir, name) !== version);
  if (stale.length > 0) {
    mkdirSync(dir, { recursive: true });
    console.log(`installing ${stale.map(([n, v]) => `${n}@${v}`).join(", ")} into ${dir}`);
    // --ignore-scripts: playwright's install script would download a browser; the run uses a cached one.
    execFileSync(
      "npm",
      ["install", "--prefix", dir, "--no-save", "--no-package-lock", "--no-audit", "--no-fund", "--ignore-scripts",
        ...Object.entries(pins).map(([n, v]) => `${n}@${v}`)],
      { stdio: ["ignore", "ignore", "inherit"] },
    );
  }
  const load = (name, file) => import(pathToFileURL(join(dir, "node_modules", name, file)).href);
  const { chromium } = await load("playwright", "index.mjs");
  const axe = await load("@axe-core/playwright", "dist/index.mjs");
  return { chromium, AxeBuilder: axe.default ?? axe.AxeBuilder };
}

/**
 * Playwright 1.63.0 uses Chromium revision 1243. When that browser is in the Playwright cache, launch it by path,
 * so a run never downloads one; otherwise let Playwright find its own (after `npx playwright@1.63.0 install`).
 */
export function cachedChromium() {
  const app = join(homedir(), "Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64",
    "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing");
  const linux = join(homedir(), ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome");
  return [app, linux].find((path) => existsSync(path));
}
