import {
  type BackendCapabilities,
  type BackendName,
  type BackendOutput,
  type InferenceRequest,
  type InferenceUsage,
  supports,
} from "./contracts.js";
import type { ResolvedModel } from "./catalog.js";
import { BackendUnavailableError, UnsupportedCapabilityError } from "./errors.js";

export type { BackendCapabilities };

export interface BackendResult {
  output: BackendOutput;
  usage: InferenceUsage;
}

/**
 * One loaded model on one device and precision. The runtime owns exactly one at a time and the
 * scheduler never calls `inferBatch` concurrently. Throw `InvalidSchemaError` for semantic schema
 * failures; any other error becomes a 500 for every request in the batch.
 */
export interface InferenceBackend {
  readonly capabilities: BackendCapabilities;
  load(): Promise<void>;
  /** Must resolve one result per request, in request order. */
  inferBatch(requests: readonly InferenceRequest[]): Promise<BackendResult[]>;
  close(): Promise<void>;
}

export interface BackendFactory {
  readonly name: "onnx";
  capabilities(): BackendCapabilities;
  create(model: ResolvedModel, device: string, precision: string): InferenceBackend;
}

/** Explicit backend selection. A request never falls back to a different backend. */
export class BackendRegistry {
  readonly #factories = new Map<BackendName, BackendFactory>();

  constructor(factories: readonly BackendFactory[] = []) {
    for (const factory of factories) {
      if (this.#factories.has(factory.name)) {
        throw new Error(`backend already registered: ${factory.name}`);
      }
      this.#factories.set(factory.name, factory);
    }
  }

  capabilities(): BackendCapabilities[] {
    return [...this.#factories.values()].map((factory) => factory.capabilities());
  }

  /** Throws unless the requested backend is registered; call before touching model artifacts. */
  require(name: BackendName): BackendFactory {
    const factory = this.#factories.get(name);
    if (factory === undefined) {
      throw new BackendUnavailableError(
        `backend ${name} is not installed; no fallback was attempted`,
      );
    }
    return factory;
  }

  /** Verifies the resolved model, operation, precision and device against advertised support. */
  assertSupported(request: InferenceRequest, modelId: string, device: string): BackendFactory {
    const factory = this.require(request.backend);
    if (!supports(factory.capabilities(), { ...request, model: modelId }, device)) {
      throw new UnsupportedCapabilityError(
        `backend ${request.backend} does not support model=${request.model}, ` +
          `operation=${request.operation}, precision=${request.precision}, device=${device}; ` +
          "no fallback was attempted",
      );
    }
    return factory;
  }
}
