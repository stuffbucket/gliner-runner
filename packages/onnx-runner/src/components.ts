import { fileURLToPath } from "node:url";

import type { BackendFactory } from "./backend.js";
import { InMemoryArtifactCatalog, type ArtifactCatalog } from "./catalog.js";
import type { ModelProvisioner } from "./downloads.js";
import type { Settings } from "./settings.js";

/** What the CLI needs from the ONNX implementation to assemble a server. */
export interface RunnerComponents {
  catalog: ArtifactCatalog;
  backends: readonly BackendFactory[];
  /** Fetches and prepares models for download jobs; without one the server declines download requests. */
  provisioner?: ModelProvisioner;
}

/**
 * Seam between the protocol server and the ONNX implementation. When `dist/src/onnx/factory.js`
 * exists the CLI imports it dynamically and calls its `createComponents` export; otherwise the CLI
 * serves with an empty catalog and no backends, so every inference request is rejected with 409
 * `unsupported_configuration` instead of falling back to another engine.
 */
export type CreateComponents = (settings: Settings) => Promise<RunnerComponents> | RunnerComponents;

export const FACTORY_SPECIFIER = "./onnx/factory.js";

export interface LoadComponentsOptions {
  log?: (message: string) => void;
  importModule?: (specifier: string) => Promise<unknown>;
}

export async function loadComponents(
  settings: Settings,
  { log = () => undefined, importModule = (specifier) => import(specifier) }: LoadComponentsOptions = {},
): Promise<RunnerComponents> {
  let module: { createComponents?: CreateComponents };
  try {
    module = (await importModule(FACTORY_SPECIFIER)) as typeof module;
  } catch (error) {
    if (!isMissingFactory(error)) {
      throw error;
    }
    log("no ONNX backend is installed; inference requests will be rejected");
    return {
      catalog: new InMemoryArtifactCatalog({ destination: settings.modelsDir }),
      backends: [],
    };
  }
  if (typeof module.createComponents !== "function") {
    throw new Error(`${FACTORY_SPECIFIER} must export createComponents(settings)`);
  }
  return module.createComponents(settings);
}

/** Only a missing factory file counts; a factory that fails to import its own dependencies must surface. */
function isMissingFactory(error: unknown): boolean {
  if (!(error instanceof Error) || (error as { code?: unknown }).code !== "ERR_MODULE_NOT_FOUND") {
    return false;
  }
  const path = fileURLToPath(new URL(FACTORY_SPECIFIER, import.meta.url));
  return error.message.startsWith(`Cannot find module '${path}'`);
}
