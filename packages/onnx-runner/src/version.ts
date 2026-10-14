import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { BUILD_INFO } from "./build-info.js";

export const VERSION_FLAGS: readonly string[] = ["--version", "-v", "--v"];

export interface VersionInfo {
  version: string;
  identity_sha256: string;
  eval_receipt_sha256: string;
  eval_run_id: string;
  node: string;
  onnxruntime: string;
}

/** Nearest package.json above this module that satisfies `accept`, walking up from the compiled file. */
function findPackage(start: string, accept: (manifest: Record<string, unknown>) => boolean): Record<string, unknown> | undefined {
  let directory = start;
  for (;;) {
    const candidate = join(directory, "package.json");
    if (existsSync(candidate)) {
      try {
        const manifest = JSON.parse(readFileSync(candidate, "utf8")) as Record<string, unknown>;
        if (accept(manifest)) {
          return manifest;
        }
      } catch {
        // keep walking
      }
    }
    const parent = dirname(directory);
    if (parent === directory) {
      return undefined;
    }
    directory = parent;
  }
}

function onnxruntimeVersion(start: string): string {
  let directory = start;
  for (;;) {
    const candidate = join(directory, "node_modules", "onnxruntime-node", "package.json");
    if (existsSync(candidate)) {
      try {
        return String((JSON.parse(readFileSync(candidate, "utf8")) as { version?: unknown }).version ?? "unknown");
      } catch {
        return "unknown";
      }
    }
    const parent = dirname(directory);
    if (parent === directory) {
      return "unknown";
    }
    directory = parent;
  }
}

/**
 * Version and evaluation-receipt identity of this build. Reads files only (no model, no native library), so it works
 * in the packaged archive without model artifacts. The version is the live package version (release tooling bumps it
 * after the receipt was recorded); the SHAs are embedded at receipt time by scripts/onnx-eval-receipt.mjs.
 */
export function readVersionInfo(): VersionInfo {
  const here = dirname(fileURLToPath(import.meta.url));
  const manifest = findPackage(here, (candidate) => candidate["name"] === "@gliner-runner/onnx-runner");
  const version = typeof manifest?.["version"] === "string" ? manifest["version"] : BUILD_INFO.version;
  return {
    version,
    identity_sha256: BUILD_INFO.identitySha256,
    eval_receipt_sha256: BUILD_INFO.evalReceiptSha256,
    eval_run_id: BUILD_INFO.evalRunId,
    node: process.version,
    onnxruntime: onnxruntimeVersion(here),
  };
}

export function formatVersion(info: VersionInfo): string {
  return [
    `gliner-runner-onnx ${info.version}`,
    `identity sha256: ${info.identity_sha256}`,
    `eval receipt sha256: ${info.eval_receipt_sha256}`,
    `eval run id: ${info.eval_run_id}`,
    "",
  ].join("\n");
}
