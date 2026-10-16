import { platform } from "#platform";

import type { Precision } from "./contracts.js";

export const MIN_MEMORY_LIMIT_BYTES = 512 * 1024 ** 2;

export function physicalMemoryBytes(): number {
  return platform.physicalMemoryBytes();
}

export function processRssBytes(): number {
  return platform.processRssBytes();
}

export function defaultMemoryLimitBytes(totalBytes: number = physicalMemoryBytes()): number {
  return Math.trunc(totalBytes * 0.75);
}

/** Residency floor: weights at the precision width, 1.6x working set, plus fixed overhead. */
export function minimumMemoryBytes(parameterCount: number, precision: Precision): number {
  const bytesPerParameter = precision === "fp32" ? 4 : 2;
  return Math.trunc(parameterCount * bytesPerParameter * 1.6 + 512 * 1024 ** 2);
}
