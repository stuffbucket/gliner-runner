import type { RunnerPlatform } from "./port.js";

/**
 * The browser build's `#platform`: empty until the embedding code (a Web Worker) installs an implementation with
 * `installPlatform`, which must happen before any runner module touches files, hashing or ONNX Runtime.
 */
let installed: RunnerPlatform | undefined;

export const platform: RunnerPlatform = new Proxy({} as RunnerPlatform, {
  get(_target, key) {
    if (installed === undefined) {
      throw new Error("no runner platform is installed; call installPlatform() first");
    }
    return installed[key as keyof RunnerPlatform];
  },
});

export function installPlatform(implementation: RunnerPlatform): void {
  installed = implementation;
}
