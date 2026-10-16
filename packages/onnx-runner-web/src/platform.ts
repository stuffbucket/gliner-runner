import { createSHA256 } from "hash-wasm";

import type { RunnerPlatform } from "@gliner-runner/onnx-runner/portable";

import { joinPath, opfs, splitPath } from "./opfs.js";
import { PROBES } from "./probe.js";

import type * as OrtWeb from "onnxruntime-web/webgpu";

export type ExecutionProvider = "webgpu" | "wasm";

export interface BrowserPlatformOptions {
  /** OPFS directory holding the artifact templates (seeded by the runner from the site). */
  templatesDir: string;
  /** The runner's package version, for processing records. */
  runnerVersion: string;
  /** `auto` (the default) prefers WebGPU when the browser has it and the self-check passes. */
  executionProvider?: "auto" | ExecutionProvider | undefined;
  /** Reports the provider a session actually runs on (after any fallback) and why it fell back. */
  onExecutionProvider?: ((provider: ExecutionProvider, note?: string) => void) | undefined;
}

/** Browsers do not expose memory reliably, so the runner's residency guard gets a generous fixed ceiling. */
const ASSUMED_MEMORY_BYTES = 16 * 1024 ** 3;

/** Whether this browser offers a usable WebGPU adapter. */
async function hasWebGpu(): Promise<boolean> {
  const gpu = (navigator as WorkerNavigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (gpu === undefined) {
    return false;
  }
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

/**
 * Runs the model's known probe row and compares it with the logits onnxruntime-node produced; a GPU path that
 * disagrees by more than the parity tolerance is not trusted.
 */
async function probeAgrees(ort: typeof OrtWeb, session: OrtWeb.InferenceSession, modelId: string | undefined): Promise<boolean> {
  const probe = modelId === undefined ? undefined : PROBES[modelId];
  if (probe === undefined) {
    return true;
  }
  const int64 = (values: readonly number[]): BigInt64Array => BigInt64Array.from(values.map(BigInt));
  const feeds = {
    input_ids: new ort.Tensor("int64", int64(probe.inputIds), [1, probe.inputIds.length]),
    attention_mask: new ort.Tensor("int64", int64(probe.attentionMask), [1, probe.attentionMask.length]),
    label_positions: new ort.Tensor("int64", int64(probe.labelPositions), [1, probe.labelPositions.length]),
    label_mask: new ort.Tensor("bool", Uint8Array.from(probe.labelMask), [1, probe.labelMask.length]),
  };
  try {
    const output = (await session.run(feeds))["logits"];
    const logits = output === undefined ? [] : [...((await output.getData()) as Float32Array)];
    output?.dispose();
    return logits.length === probe.logits.length && logits.every((value, index) => Math.abs(value - (probe.logits[index] as number)) <= 1e-4);
  } finally {
    for (const tensor of Object.values(feeds)) {
      tensor.dispose();
    }
  }
}

export function createBrowserPlatform(options: BrowserPlatformOptions): RunnerPlatform {
  return {
    fs: opfs,
    join: joinPath,
    async createSha256() {
      const hash = await createSHA256();
      hash.init();
      return {
        update: (data) => void hash.update(data),
        digest: () => hash.digest("hex"),
      };
    },
    async freeBytes() {
      const { quota, usage } = await navigator.storage.estimate();
      return quota === undefined || usage === undefined ? undefined : quota - usage;
    },
    physicalMemoryBytes: () => ASSUMED_MEMORY_BYTES,
    processRssBytes: () => 0,
    env: () => ({}),
    defaultTemplatesDir: () => options.templatesDir,
    runnerVersion: () => options.runnerVersion,
    resolvePath: (path) => Promise.resolve(`/${splitPath(path).join("/")}`),
    discoverSnapshots: () => Promise.resolve({ candidates: [], selected: undefined }),
    catalogCache: () => ({
      discoveredRevision: () => undefined,
      discoveredModels: () => [],
      refresh: () => Promise.reject(new Error("the in-browser runner does not refresh the model catalog")),
    }),
    async importTokenizers() {
      const module = (await import("@huggingface/tokenizers")) as unknown as { Tokenizer: Awaited<ReturnType<RunnerPlatform["importTokenizers"]>> };
      return module.Tokenizer;
    },
    async createSession(modelDirectory) {
      // The bundled build finds its WebAssembly next to itself (the bundler emits it as an asset). Setting
      // `env.wasm.wasmPaths` on this build breaks session creation, so it is left alone.
      const ort = await import("onnxruntime-web/webgpu");
      ort.env.logLevel = "error";
      // Threads need cross-origin isolation, which a GitHub Pages site cannot turn on.
      ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

      const manifest = JSON.parse(await opfs.readFile(joinPath(modelDirectory, "manifest.json"))) as { model_id?: string };
      const graph = await opfs.open(joinPath(modelDirectory, "model.onnx"), "read");
      const graphBytes = new Uint8Array((await graph.stat()).size);
      await graph.read(graphBytes, 0, graphBytes.length, 0);
      const dataFile = await fileOf(joinPath(modelDirectory, "model.onnx.data"));

      const create = async (provider: ExecutionProvider): Promise<OrtWeb.InferenceSession> => {
        // A blob URL of the OPFS file lets onnxruntime-web stream the weights into its heap without a second copy.
        const url = URL.createObjectURL(dataFile);
        try {
          return await ort.InferenceSession.create(graphBytes, {
            executionProviders: [provider],
            externalData: [{ path: "model.onnx.data", data: url }],
            graphOptimizationLevel: "all",
            enableCpuMemArena: false,
            logSeverityLevel: 3, // errors only; node placement warnings are expected
          });
        } finally {
          URL.revokeObjectURL(url);
        }
      };

      const preference = options.executionProvider ?? "auto";
      let note: string | undefined;
      if (preference !== "wasm" && !(await hasWebGpu())) {
        note = "this browser has no WebGPU";
      } else if (preference !== "wasm") {
        try {
          const session = await create("webgpu");
          if (await probeAgrees(ort, session, manifest.model_id)) {
            options.onExecutionProvider?.("webgpu");
            return { ort: ort as never, session: session as never };
          }
          await session.release();
          note = "WebGPU results did not match the reference";
        } catch (error) {
          if (preference === "webgpu") {
            throw error;
          }
          note = `WebGPU failed (${error instanceof Error ? error.message : String(error)})`;
        }
      }
      // Asking for WebGPU without an adapter can hang some engines, so WASM is chosen up front in that case.
      const session = await create("wasm");
      options.onExecutionProvider?.("wasm", note);
      return { ort: ort as never, session: session as never };
    },
  };
}

async function fileOf(path: string): Promise<File> {
  const parts = splitPath(path);
  const name = parts.pop() as string;
  let current = await navigator.storage.getDirectory();
  for (const part of parts) {
    current = await current.getDirectoryHandle(part);
  }
  return (await current.getFileHandle(name)).getFile();
}
