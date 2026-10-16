import {
  HuggingFaceClient,
  KNOWN_MODELS,
  OnnxArtifactCatalog,
  OnnxBackendFactory,
  OnnxModelProvisioner,
  TEMPLATE_NAME,
  createRunnerHandler,
  installPlatform,
  templateDirectory,
  type InstallProgress,
  type KnownModel,
  type RunnerHandler,
} from "@gliner-runner/onnx-runner/portable";
import { version as runnerVersion } from "@gliner-runner/onnx-runner/package.json";

import { joinPath, opfs } from "./opfs.js";
import { createBrowserPlatform, type ExecutionProvider } from "./platform.js";

/** The model the in-tab runner offers; larger ones exceed what a browser tab can hold. */
export const IN_TAB_MODEL = "decide-340m";
export const MODELS_DIR = "/models";
const TEMPLATES_DIR = "/templates";

export interface BrowserRunnerOptions {
  /** Where the site serves the artifact templates: `<base><owner>--<name>/<revision>/<file>`. */
  templatesBaseUrl: string;
  /** Hugging Face endpoint (default https://huggingface.co). */
  hfEndpoint?: string | undefined;
  executionProvider?: "auto" | ExecutionProvider | undefined;
  onProgress?: ((progress: InstallProgress) => void) | undefined;
  onExecutionProvider?: ((provider: ExecutionProvider, note?: string) => void) | undefined;
  log?: ((message: string) => void) | undefined;
}

export interface BrowserRunner {
  handler: RunnerHandler;
  close(): Promise<void>;
}

const repositoryDirectory = (model: KnownModel): string => model.repository.replace("/", "--");

/** Copies the bundled template files of `models` from the site into OPFS; `loadTemplate` verifies every hash. */
async function seedTemplates(baseUrl: string, models: readonly KnownModel[]): Promise<void> {
  for (const model of models) {
    const source = new URL(`${repositoryDirectory(model)}/${model.pinnedRevision}/`, new URL(baseUrl, self.location.href));
    const target = templateDirectory(TEMPLATES_DIR, model, model.pinnedRevision);
    await opfs.mkdir(target, { recursive: true });
    const fetchText = async (name: string): Promise<string> => {
      const response = await fetch(new URL(name, source));
      if (!response.ok) {
        throw new Error(`could not load the ${model.modelId} template file ${name} (HTTP ${response.status})`);
      }
      return response.text();
    };
    const templateText = await fetchText(TEMPLATE_NAME);
    const files = (JSON.parse(templateText) as { template_files: { path: string }[] }).template_files;
    for (const file of files) {
      const response = await fetch(new URL(file.path, source));
      if (!response.ok) {
        throw new Error(`could not load the ${model.modelId} template file ${file.path} (HTTP ${response.status})`);
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      await opfs.rm(joinPath(target, file.path), { force: true });
      const handle = await opfs.open(joinPath(target, file.path), "write-truncate");
      try {
        for (let done = 0; done < bytes.length; ) {
          done += (await handle.write(bytes, done, bytes.length - done, done)).bytesWritten;
        }
        await handle.sync();
      } finally {
        await handle.close();
      }
    }
    await opfs.writeFile(joinPath(target, TEMPLATE_NAME), templateText);
  }
}

/** Removes what an interrupted download or preparation left behind (the runner cleans up, unless the tab closed). */
async function sweep(): Promise<void> {
  const onnx = joinPath(MODELS_DIR, "onnx");
  let repositories: string[];
  try {
    repositories = await opfs.readdir(onnx);
  } catch {
    return;
  }
  for (const repository of repositories) {
    const directory = joinPath(onnx, repository);
    for (const name of await opfs.readdir(directory).catch(() => [] as string[])) {
      if (name.startsWith(".download-") || name.startsWith(".staging-") || name.includes(".tmp-")) {
        await opfs.rm(joinPath(directory, name), { recursive: true, force: true });
      }
    }
  }
}

/**
 * Starts the runner in this worker: the same catalog, installer, ONNX backend and request handler as
 * `gliner-runner serve`, on OPFS and onnxruntime-web, offering `decide-340m`.
 */
export async function createBrowserRunner(options: BrowserRunnerOptions): Promise<BrowserRunner> {
  installPlatform(
    createBrowserPlatform({
      templatesDir: TEMPLATES_DIR,
      runnerVersion,
      executionProvider: options.executionProvider,
      onExecutionProvider: options.onExecutionProvider,
    }),
  );
  await sweep();
  const models = KNOWN_MODELS.filter((model) => model.modelId === IN_TAB_MODEL);
  await seedTemplates(options.templatesBaseUrl, models);

  const huggingface = new HuggingFaceClient({ endpoint: options.hfEndpoint, token: undefined, redirect: "follow" });
  const catalog = new OnnxArtifactCatalog(MODELS_DIR, { models });
  await catalog.inventory();
  const handler = createRunnerHandler({
    catalog,
    backends: [new OnnxBackendFactory(catalog, { maxRowsPerRun: 8 })],
    provisioner: new OnnxModelProvisioner({
      modelsDir: MODELS_DIR,
      templatesDir: TEMPLATES_DIR,
      models,
      huggingface,
      onProgress: options.onProgress,
    }),
    settings: {
      device: "cpu",
      queueCapacity: 32,
      maxBatchSize: 8,
      batchWindowMs: 4,
      modelIdleTtlSeconds: 0,
      memoryLimitBytes: undefined,
      maxBodyBytes: 1024 * 1024,
      autoDownload: false,
    },
    log: options.log,
  });
  return { handler, close: () => handler.close() };
}

/** Deletes every downloaded model and partial download from this site's browser storage. */
export async function deleteModels(): Promise<void> {
  await opfs.rm(MODELS_DIR, { recursive: true, force: true });
}
