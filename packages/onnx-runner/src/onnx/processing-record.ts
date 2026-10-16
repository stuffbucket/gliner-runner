import { platform } from "#platform";

import type { KnownModel } from "../catalog.js";
import { randomHex } from "../timers.js";
import { repositoryDirectoryName } from "./artifacts.js";
import type { OnnxTemplate } from "./templates.js";

export const RECORD_KIND = "gliner-runner-onnx-processing-record";
export const RECORD_FORMAT_VERSION = 1;
export const RECORD_SUFFIX = ".gliner-runner.json";

export type ProcessingState = "not_processed" | "verified_source" | "prepared" | "failed";

export interface RecordedSourceFile {
  path: string;
  size: number;
  sha256: string;
  verified: boolean;
}

/** What operators see about one model revision: whether gliner-runner processed it and what was applied. */
export interface ProcessingRecord {
  format_version: 1;
  kind: typeof RECORD_KIND;
  model_id: string;
  repository: string;
  revision: string;
  gliner_runner_version: string;
  updated_at: string;
  source: {
    /** A directory path, `huggingface:<endpoint>`, or `null` when no source was found. */
    location: string | null;
    files: RecordedSourceFile[];
  };
  processed: boolean;
  state: ProcessingState;
  steps: string[];
  template: { sha256: string; format_version: number };
  artifact?: { path: string; manifest_sha256: string; files: { path: string; size: number; sha256: string }[] };
  error?: string;
  /** Candidate snapshots that were looked at and refused, with the reason (most recent attempt). */
  rejected_candidates?: { location: string; reason: string }[];
}

const join = (...parts: string[]): string => platform.join(...parts);

/** The runner's version, from the package's `package.json` (also present in packaged releases). */
export function runnerVersion(): string {
  return platform.runnerVersion();
}

/** Sibling of the artifact directory: `<modelsDir>/onnx/<owner>--<name>/<revision>.gliner-runner.json`. */
export function recordPath(modelsDir: string, repository: string, revision: string): string {
  return join(modelsDir, "onnx", repositoryDirectoryName(repository), `${revision}${RECORD_SUFFIX}`);
}

/** Reads a record; `undefined` when there is none, it is unreadable, or it is not a v1 processing record. */
export async function readProcessingRecord(
  modelsDir: string,
  model: KnownModel,
  revision: string,
): Promise<ProcessingRecord | undefined> {
  try {
    const value = JSON.parse(await platform.fs.readFile(recordPath(modelsDir, model.repository, revision))) as Partial<ProcessingRecord> | null;
    return value !== null && typeof value === "object" && value.kind === RECORD_KIND && value.format_version === RECORD_FORMAT_VERSION
      ? (value as ProcessingRecord)
      : undefined;
  } catch {
    return undefined;
  }
}

/** Writes a record atomically (temporary sibling, then rename). Never touches the artifact directory. */
export async function writeProcessingRecord(modelsDir: string, record: ProcessingRecord): Promise<string> {
  const path = recordPath(modelsDir, record.repository, record.revision);
  await platform.fs.mkdir(join(modelsDir, "onnx", repositoryDirectoryName(record.repository)), { recursive: true });
  const temporary = `${path}.tmp-${randomHex(4)}`;
  try {
    await platform.fs.writeFile(temporary, `${JSON.stringify(record, null, 2)}\n`);
    await platform.fs.rename(temporary, path);
  } catch (error) {
    await platform.fs.rm(temporary, { force: true });
    throw error;
  }
  return path;
}

export interface RecordInput {
  template: OnnxTemplate;
  state: ProcessingState;
  steps: readonly string[];
  source: { location: string | null; verified: boolean };
  error?: string | undefined;
  artifactPath?: string | undefined;
  rejected?: readonly { location: string; reason: string }[] | undefined;
  now?: Date | undefined;
}

/** Assembles the record for the current state of a model revision. */
export function buildRecord(input: RecordInput): ProcessingRecord {
  const { template } = input;
  const record: ProcessingRecord = {
    format_version: RECORD_FORMAT_VERSION,
    kind: RECORD_KIND,
    model_id: template.model.modelId,
    repository: template.model.repository,
    revision: template.revision,
    gliner_runner_version: runnerVersion(),
    updated_at: (input.now ?? new Date()).toISOString(),
    source: {
      location: input.source.location,
      files: template.sourceFiles.map((file) => ({
        path: file.path,
        size: file.size,
        sha256: file.sha256,
        verified: input.source.verified,
      })),
    },
    processed: input.state === "prepared",
    state: input.state,
    steps: [...input.steps],
    template: { sha256: template.templateSha256, format_version: template.templateFormatVersion },
  };
  if (input.state === "prepared" && input.artifactPath !== undefined) {
    const manifest = template.templateFiles.find((file) => file.output === "manifest.json");
    record.artifact = {
      path: input.artifactPath,
      manifest_sha256: manifest?.sha256 ?? "",
      files: template.outputs.map((output) => ({ path: output.path, size: output.size, sha256: output.sha256 })),
    };
  }
  if (input.error !== undefined) {
    record.error = input.error;
  }
  if (input.rejected !== undefined && input.rejected.length > 0) {
    record.rejected_candidates = input.rejected.slice(0, 20).map((entry) => ({ ...entry }));
  }
  return record;
}

/** The step name for the template assembly, e.g. `assembled_onnx_from_template:v1 (98 transpose, 296 identity of 394 tensors)`. */
export function assemblyStep(template: OnnxTemplate): string {
  const transposed = template.data.tensors.filter((tensor) => tensor.transform === "transpose").length;
  const total = template.data.tensors.length;
  return `assembled_onnx_from_template:v${template.templateFormatVersion} (${transposed} transpose, ${total - transposed} identity of ${total} tensors)`;
}
