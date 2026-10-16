import { platform } from "#platform";

import type { BackendCapabilities, BackendFactory, BackendResult, InferenceBackend } from "../backend.js";
import type { ResolvedModel } from "../catalog.js";
import type { InferenceRequest } from "../contracts.js";
import { InvalidSchemaError, ModelIntegrityError, UnsupportedCapabilityError } from "../errors.js";
import {
  Gliner2Preprocessor,
  SPECIAL_TOKENS,
  collateRows,
  compileRunnerSchema,
  decode,
  isGliner2SchemaError,
} from "../gliner2/index.js";
import type { CompiledSchema, Gliner2Row, RunnerClassificationSchema } from "../gliner2/index.js";
import { batchKey } from "../scheduler.js";
import { orderedTaskEntries } from "../task-order.js";
import { GRAPH_INPUTS, GRAPH_OUTPUTS } from "./manifest.js";
import { readPreprocessingConfig } from "./preprocessing.js";

import type * as Ort from "onnxruntime-node";

export const ONNX_BACKEND_CAPABILITIES = {
  backend: "onnx",
  operations: ["classify"],
  precisions: ["fp32"],
  devices: ["cpu"],
  dynamic_batching: true,
  precision_profiles: [{ device: "cpu", precision: "fp32" }],
} as const satisfies Omit<BackendCapabilities, "validated_models">;

export interface OnnxBackendOptions {
  /** ONNX Runtime intra-op threads; `undefined` lets ONNX Runtime pick the physical core count. */
  intraOpThreads?: number | undefined;
  /** Largest number of rows in one session run; larger batches are executed in chunks. */
  maxRowsPerRun?: number | undefined;
  /** Largest padded `rows x sequence` token count of one session run (a single row is always allowed). */
  maxPaddedTokensPerRun?: number | undefined;
}

const DEFAULT_MAX_ROWS = 32;
const DEFAULT_MAX_PADDED_TOKENS = 32_768;
const COMPILE_CACHE_SIZE = 64;

function threadsFromEnvironment(): number | undefined {
  const raw = platform.env()["GLINER_RUNNER_ONNX_INTRA_OP_THREADS"];
  const value = raw === undefined || raw === "" ? Number.NaN : Number(raw);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

/** One prepared GLiNER2 classifier on ONNX Runtime (CPU, fp32). */
export class OnnxBackend implements InferenceBackend {
  readonly capabilities: BackendCapabilities;
  readonly #model: ResolvedModel;
  readonly #options: OnnxBackendOptions;
  readonly #compiled = new Map<string, CompiledSchema>();
  #session: Ort.InferenceSession | undefined;
  #ort: typeof Ort | undefined;
  #preprocessor: Gliner2Preprocessor | undefined;
  #loading: Promise<void> | undefined;

  constructor(model: ResolvedModel, options: OnnxBackendOptions = {}) {
    this.#model = model;
    this.#options = options;
    this.capabilities = { ...ONNX_BACKEND_CAPABILITIES, validated_models: [model.modelId] };
  }

  load(): Promise<void> {
    this.#loading ??= this.#load().catch(async (error: unknown) => {
      this.#loading = undefined;
      await this.close();
      throw error;
    });
    return this.#loading;
  }

  async #load(): Promise<void> {
    if (this.#session !== undefined) {
      return;
    }
    const model = this.#model;
    const identity = { modelId: model.modelId, repository: model.repository, revision: model.revision };
    const config = await readPreprocessingConfig(model.path, identity);

    const Tokenizer = await platform.importTokenizers();
    const readJson = async (name: string): Promise<object> =>
      JSON.parse(await platform.fs.readFile(platform.join(model.path, name))) as object;
    const tokenizer = new Tokenizer(
      await readJson("tokenizer.json"),
      await readJson("tokenizer_config.json"),
    );
    const preprocessor = new Gliner2Preprocessor(tokenizer);
    for (const token of SPECIAL_TOKENS) {
      if (preprocessor.specialTokenIds[token] !== config.specialTokenIds[token]) {
        throw new ModelIntegrityError(
          `tokenizer id of ${token} (${preprocessor.specialTokenIds[token]}) does not match the artifact's ` +
            `gliner2.json (${config.specialTokenIds[token]})`,
        );
      }
    }
    if (tokenizer.token_to_id?.(config.unknownToken) !== config.unknownTokenId) {
      throw new ModelIntegrityError(
        `tokenizer id of ${config.unknownToken} does not match the artifact's gliner2.json`,
      );
    }

    const threads = this.#options.intraOpThreads ?? threadsFromEnvironment();
    const { ort, session } = await platform.createSession(model.path, { intraOpThreads: threads });
    try {
      const expectedInputs = GRAPH_INPUTS.map((input) => input.name);
      const expectedOutputs = GRAPH_OUTPUTS.map((output) => output.name);
      if (
        JSON.stringify([...session.inputNames]) !== JSON.stringify(expectedInputs) ||
        JSON.stringify([...session.outputNames]) !== JSON.stringify(expectedOutputs)
      ) {
        throw new ModelIntegrityError(
          `ONNX graph inputs ${JSON.stringify(session.inputNames)} / outputs ` +
            `${JSON.stringify(session.outputNames)} do not match the artifact manifest`,
        );
      }
    } catch (error) {
      await session.release().catch(() => undefined);
      throw error;
    }
    this.#ort = ort;
    this.#preprocessor = preprocessor;
    this.#session = session;
  }

  async inferBatch(requests: readonly InferenceRequest[]): Promise<BackendResult[]> {
    if (requests.length === 0) {
      return [];
    }
    const session = this.#session;
    const preprocessor = this.#preprocessor;
    const ort = this.#ort;
    if (session === undefined || preprocessor === undefined || ort === undefined) {
      throw new Error("backend must be loaded before inference");
    }
    const first = requests[0] as InferenceRequest;
    // A backend serves exactly one resolved model identity, so requests naming it through different aliases
    // ("decide-340m", "fastino/GLiNER2.5-Decide", "decide-340m@<sha>") are compatible; each response echoes the
    // `model` its own request carried, as the PyTorch runner does.
    for (const request of requests) {
      if (
        request.backend !== first.backend ||
        request.precision !== first.precision ||
        batchKey(request) !== batchKey(first)
      ) {
        throw new Error("infer_batch received incompatible requests");
      }
    }
    if (first.schema.kind !== "classification") {
      throw new UnsupportedCapabilityError(`unsupported schema: ${first.schema.kind}`);
    }
    const compiled = this.#compile(first.schema);
    const decodeConfig = {
      candidateThreshold: first.options.threshold,
      includeConfidence: first.options.include_confidence,
    };

    const rows = requests.map((request) => preprocessor.buildInput(request.text, compiled));
    const results: BackendResult[] = [];
    for (const chunk of this.#chunks(rows)) {
      const batch = collateRows(chunk);
      const feeds = {
        input_ids: new ort.Tensor("int64", batch.inputIds, [batch.batchSize, batch.seqLen]),
        attention_mask: new ort.Tensor("int64", batch.attentionMask, [batch.batchSize, batch.seqLen]),
        label_positions: new ort.Tensor("int64", batch.labelPositions, [batch.batchSize, batch.labelLen]),
        label_mask: new ort.Tensor("bool", batch.labelMask, [batch.batchSize, batch.labelLen]),
      };
      const outputs = await session.run(feeds);
      const logits = outputs["logits"];
      try {
        if (
          logits === undefined ||
          logits.type !== "float32" ||
          logits.dims.length !== 2 ||
          logits.dims[0] !== batch.batchSize ||
          logits.dims[1] !== batch.labelLen
        ) {
          throw new Error("ONNX graph returned logits with an unexpected type or shape");
        }
        const data = logits.data as Float32Array;
        chunk.forEach((row, index) => {
          // Padded label slots hold float32 min; only the row's real label count is decoded.
          const rowLogits = data.subarray(index * batch.labelLen, index * batch.labelLen + row.labelPositions.length);
          results.push({
            output: decode(compiled, rowLogits, decodeConfig).output,
            usage: { inputTokens: row.inputTokens, outputTokens: 0 },
          });
        });
      } finally {
        for (const tensor of Object.values(feeds)) {
          tensor.dispose();
        }
        logits?.dispose();
      }
    }
    return results;
  }

  /** Compiles (and briefly caches) the request schema; schema problems become `InvalidSchemaError`. */
  #compile(schema: Extract<InferenceRequest["schema"], { kind: "classification" }>): CompiledSchema {
    const tasks = orderedTaskEntries(schema.tasks);
    const key = JSON.stringify([schema, tasks.map(([name]) => name)]);
    const hit = this.#compiled.get(key);
    if (hit !== undefined) {
      return hit;
    }
    let compiled: CompiledSchema;
    try {
      compiled = compileRunnerSchema({ ...schema, tasks } as RunnerClassificationSchema);
    } catch (error) {
      if (isGliner2SchemaError(error)) {
        throw new InvalidSchemaError(error.message);
      }
      throw error;
    }
    if (this.#compiled.size >= COMPILE_CACHE_SIZE) {
      this.#compiled.clear();
    }
    this.#compiled.set(key, compiled);
    return compiled;
  }

  *#chunks(rows: readonly Gliner2Row[]): Generator<readonly Gliner2Row[]> {
    const maxRows = this.#options.maxRowsPerRun ?? DEFAULT_MAX_ROWS;
    const maxTokens = this.#options.maxPaddedTokensPerRun ?? DEFAULT_MAX_PADDED_TOKENS;
    let chunk: Gliner2Row[] = [];
    let longest = 0;
    for (const row of rows) {
      const grown = Math.max(longest, row.inputIds.length);
      if (chunk.length > 0 && (chunk.length >= maxRows || (chunk.length + 1) * grown > maxTokens)) {
        yield chunk;
        chunk = [];
        longest = 0;
      }
      chunk.push(row);
      longest = Math.max(longest, row.inputIds.length);
    }
    if (chunk.length > 0) {
      yield chunk;
    }
  }

  /** Releases the session. Safe to call repeatedly, before load, and after a failed load. */
  async close(): Promise<void> {
    const session = this.#session;
    this.#session = undefined;
    this.#preprocessor = undefined;
    this.#ort = undefined;
    this.#compiled.clear();
    this.#loading = undefined;
    await session?.release();
  }
}

export interface InstalledModels {
  installedModelIds(): readonly string[];
}

/** Registers the ONNX backend; advertises exactly the curated artifacts the catalog found installed. */
export class OnnxBackendFactory implements BackendFactory {
  readonly name = "onnx" as const;
  readonly #installed: InstalledModels;
  readonly #options: OnnxBackendOptions;

  constructor(installed: InstalledModels, options: OnnxBackendOptions = {}) {
    this.#installed = installed;
    this.#options = options;
  }

  capabilities(): BackendCapabilities {
    return { ...ONNX_BACKEND_CAPABILITIES, validated_models: this.#installed.installedModelIds() };
  }

  create(model: ResolvedModel, device: string, precision: string): InferenceBackend {
    if (device !== "cpu" || precision !== "fp32") {
      throw new UnsupportedCapabilityError(
        `onnx backend supports only cpu/fp32, not ${device}/${precision}; no fallback was attempted`,
      );
    }
    return new OnnxBackend(model, this.#options);
  }
}
