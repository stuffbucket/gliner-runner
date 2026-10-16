/**
 * The runner without Node: the request handler, model catalog, installer and ONNX backend, for embedding in a
 * browser Web Worker. Nothing reachable from here imports `node:*` or onnxruntime-node at run time (checked by
 * `test/portable-graph.test.ts`); the host is supplied with `installPlatform`.
 */
export { installPlatform } from "#platform";
export type {
  CatalogCacheLike,
  FileStatus,
  OpenMode,
  PlatformFile,
  PlatformFileSystem,
  PlatformSession,
  RunnerPlatform,
  Sha256,
  SessionOptions,
  TokenizerConstructor,
} from "./platform/port.js";
export {
  createRunnerHandler,
  type HandlerSettings,
  type RunnerHandler,
  type RunnerHandlerOptions,
  type RunnerReply,
  type RunnerRequest,
} from "./handler.js";
export { KNOWN_MODELS, type KnownModel } from "./catalog.js";
export { OnnxArtifactCatalog } from "./onnx/artifacts.js";
export { ONNX_BACKEND_CAPABILITIES, OnnxBackendFactory } from "./onnx/backend.js";
export { HuggingFaceClient, type DownloadProgress, type HuggingFaceOptions } from "./onnx/huggingface.js";
export { OnnxModelProvisioner, type InstallProgress, type OnnxProvisionerOptions } from "./onnx/install.js";
export { TEMPLATE_NAME, templateDirectory } from "./onnx/templates.js";
