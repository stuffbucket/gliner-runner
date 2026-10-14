import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open, readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { KNOWN_MODELS, isCommitSha, type KnownModel } from "../catalog.js";
import { ModelIntegrityError } from "../errors.js";
import { repositoryDirectoryName } from "./artifacts.js";
import { GRAPH_NAME, MANIFEST_NAME, PREPROCESSING_NAME, parseManifest, type OnnxManifest } from "./manifest.js";

export const TEMPLATE_NAME = "template.json";
export const TEMPLATE_KIND = "gliner-runner-onnx-template";
export const TEMPLATE_FORMAT_VERSION = 1;
export const TEMPLATES_ENV = "GLINER_RUNNER_ONNX_TEMPLATES_DIR";
export const WEIGHTS_FILE = "model.safetensors";

export interface TemplateSourceFile {
  path: string;
  size: number;
  sha256: string;
  role: string;
}

export interface TemplateFile {
  /** Name inside the template directory. */
  path: string;
  size: number;
  sha256: string;
  /** Name inside the artifact. */
  output: string;
}

export interface TemplateTensor {
  /** ONNX initializer name (informational). */
  name: string;
  /** safetensors key. */
  source: string;
  transform: "identity" | "transpose";
  sourceShape: readonly number[];
  shape: readonly number[];
  offset: number;
  length: number;
}

export interface TemplateData {
  path: string;
  alignment: number;
  size: number;
  sha256: string;
  tensors: readonly TemplateTensor[];
}

export interface TemplateOutput {
  path: string;
  size: number;
  sha256: string;
}

/** The structure of `template.json` after strict validation; no file contents have been checked yet. */
export interface ParsedTemplate {
  model: KnownModel;
  revision: string;
  sourceFiles: readonly TemplateSourceFile[];
  templateFiles: readonly TemplateFile[];
  data: TemplateData;
  outputs: readonly TemplateOutput[];
}

/** A strictly validated, pin-matched artifact template whose files have been verified. */
export interface OnnxTemplate extends ParsedTemplate {
  /** Directory the template was loaded from; `templateFiles[].path` is relative to it. */
  directory: string;
  manifest: OnnxManifest;
  /** SHA-256 of the `template.json` bytes, recorded in processing records. */
  templateSha256: string;
  templateFormatVersion: number;
}

export class TemplateError extends ModelIntegrityError {}

/** Bundled templates: `<package>/templates`, resolved relative to this compiled module (`dist/src/onnx`). */
export function defaultTemplatesDir(): string {
  return fileURLToPath(new URL("../../../templates", import.meta.url));
}

/** `--templates-dir` wins over `GLINER_RUNNER_ONNX_TEMPLATES_DIR`, which wins over the bundled directory. */
export function resolveTemplatesDir(explicit?: string, env: NodeJS.ProcessEnv = process.env): string {
  if (explicit !== undefined && explicit !== "") {
    return explicit;
  }
  const fromEnv = env[TEMPLATES_ENV];
  return fromEnv !== undefined && fromEnv !== "" ? fromEnv : defaultTemplatesDir();
}

/** Directory of one template: `<templatesDir>/<owner>--<name>/<revision>`. */
export function templateDirectory(templatesDir: string, model: KnownModel, revision: string): string {
  return join(templatesDir, repositoryDirectoryName(model.repository), revision);
}

type Json = Record<string, unknown>;
const isRecord = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const SHA256 = /^[a-f0-9]{64}$/;
const O_NOFOLLOW = constants.O_NOFOLLOW ?? 0;

function plainName(name: unknown): name is string {
  return (
    typeof name === "string" &&
    name.length > 0 &&
    !/[\\/\0]/.test(name) &&
    !name.startsWith(".")
  );
}

const sameList = (left: readonly number[], right: readonly number[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index]);

const roundUp = (value: number, alignment: number): number => Math.ceil(value / alignment) * alignment;

/**
 * Pure structural validation of a parsed `template.json` against the curated pin. Throws `TemplateError` on the
 * first violation. File contents are not touched here (see `loadTemplate`).
 */
export function parseTemplate(raw: unknown, model: KnownModel, revision: string): ParsedTemplate {
  const fail = (detail: string): never => {
    throw new TemplateError(`ONNX template for '${model.modelId}' at ${revision} is invalid: ${detail}`);
  };
  if (!isRecord(raw)) {
    return fail("template.json must be a JSON object");
  }
  const expect = (key: string, expected: unknown): void => {
    if (JSON.stringify(raw[key]) !== JSON.stringify(expected)) {
      fail(`${key} must be ${JSON.stringify(expected)}, found ${JSON.stringify(raw[key])}`);
    }
  };
  expect("format_version", TEMPLATE_FORMAT_VERSION);
  expect("kind", TEMPLATE_KIND);
  expect("model_id", model.modelId);
  expect("repository", model.repository);
  expect("revision", model.pinnedRevision);
  if (revision !== model.pinnedRevision || !isCommitSha(revision)) {
    fail(`revision ${revision} is not the pinned revision ${model.pinnedRevision}`);
  }

  const integer = (value: unknown, label: string, minimum = 0): number => {
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) {
      return fail(`${label} must be an integer of at least ${minimum}`);
    }
    return value;
  };
  const digest = (value: unknown, label: string): string => {
    if (typeof value !== "string" || !SHA256.test(value)) {
      return fail(`${label} must be 64 lowercase hex characters`);
    }
    return value;
  };
  const list = (value: unknown, label: string): unknown[] => {
    if (!Array.isArray(value) || value.length === 0) {
      return fail(`${label} must be a non-empty list`);
    }
    return value;
  };

  const sourceFiles: TemplateSourceFile[] = [];
  const sourceNames = new Set<string>();
  for (const entry of list(raw["source_files"], "source_files")) {
    if (!isRecord(entry) || !plainName(entry["path"])) {
      return fail("source_files entries must be objects with a plain file name path");
    }
    const path = entry["path"];
    if (sourceNames.has(path)) {
      fail(`source_files lists ${path} more than once`);
    }
    sourceNames.add(path);
    const role = entry["role"];
    if (typeof role !== "string") {
      return fail(`source_files ${path} role must be a string`);
    }
    sourceFiles.push({
      path,
      size: integer(entry["size"], `source_files ${path} size`),
      sha256: digest(entry["sha256"], `source_files ${path} sha256`),
      role,
    });
  }
  if (!sourceNames.has(WEIGHTS_FILE)) {
    fail(`source_files must include ${WEIGHTS_FILE}`);
  }

  const templateFiles: TemplateFile[] = [];
  const byOutput = new Map<string, TemplateFile>();
  for (const entry of list(raw["template_files"], "template_files")) {
    if (!isRecord(entry) || !plainName(entry["path"]) || !plainName(entry["output"])) {
      return fail("template_files entries must be objects with plain file name path and output");
    }
    const path = entry["path"];
    const output = entry["output"];
    if (templateFiles.some((file) => file.path === path) || byOutput.has(output)) {
      fail(`template_files lists ${path} -> ${output} more than once`);
    }
    const file: TemplateFile = {
      path,
      output,
      size: integer(entry["size"], `template_files ${path} size`),
      sha256: digest(entry["sha256"], `template_files ${path} sha256`),
    };
    templateFiles.push(file);
    byOutput.set(output, file);
  }
  for (const required of [GRAPH_NAME, PREPROCESSING_NAME, MANIFEST_NAME]) {
    if (!byOutput.has(required)) {
      fail(`template_files must produce ${required}`);
    }
  }

  const dataRaw = raw["data"];
  if (!isRecord(dataRaw)) {
    return fail("data must be an object");
  }
  const dataPath = dataRaw["path"];
  if (!plainName(dataPath) || dataPath === MANIFEST_NAME) {
    return fail("data.path must be a plain file name other than manifest.json");
  }
  const alignment = integer(dataRaw["alignment"], "data.alignment", 1);
  const tensors: TemplateTensor[] = [];
  let cursor = 0;
  const names = new Set<string>();
  for (const [index, entry] of list(dataRaw["tensors"], "data.tensors").entries()) {
    const where = `data.tensors[${index}]`;
    if (!isRecord(entry)) {
      return fail(`${where} must be an object`);
    }
    const name = entry["name"];
    const source = entry["source"];
    if (typeof name !== "string" || name === "" || typeof source !== "string" || source === "") {
      return fail(`${where} needs a name and a source`);
    }
    if (names.has(name)) {
      fail(`${where} repeats initializer ${JSON.stringify(name)}`);
    }
    names.add(name);
    const transform = entry["transform"];
    if (transform !== "identity" && transform !== "transpose") {
      return fail(`${where} transform must be "identity" or "transpose"`);
    }
    if (entry["dtype"] !== "float32") {
      return fail(`${where} dtype must be "float32"`);
    }
    const shapeOf = (value: unknown, label: string): number[] => {
      if (!Array.isArray(value) || value.some((dim) => typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0)) {
        return fail(`${where} ${label} must be a list of non-negative integers`);
      }
      return [...(value as number[])];
    };
    const sourceShape = shapeOf(entry["source_shape"], "source_shape");
    const shape = shapeOf(entry["shape"], "shape");
    if (transform === "transpose") {
      if (sourceShape.length !== 2 || !sameList(shape, [sourceShape[1] as number, sourceShape[0] as number])) {
        fail(`${where} transpose needs a 2-D source_shape [r, c] and shape [c, r]`);
      }
    } else if (!sameList(shape, sourceShape)) {
      fail(`${where} identity shape must equal source_shape`);
    }
    const elements = sourceShape.reduce((product, dim) => product * dim, 1);
    const offset = integer(entry["offset"], `${where} offset`);
    const length = integer(entry["length"], `${where} length`);
    if (!Number.isSafeInteger(elements * 4) || length !== elements * 4) {
      fail(`${where} length ${length} does not match shape ${JSON.stringify(sourceShape)} of float32`);
    }
    if (offset !== roundUp(cursor, alignment)) {
      fail(`${where} offset ${offset} is not the previous end ${cursor} rounded up to ${alignment}`);
    }
    cursor = offset + length;
    tensors.push({ name, source, transform, sourceShape, shape, offset, length });
  }
  const data: TemplateData = {
    path: dataPath,
    alignment,
    size: integer(dataRaw["size"], "data.size"),
    sha256: digest(dataRaw["sha256"], "data.sha256"),
    tensors,
  };
  if (data.size !== cursor) {
    fail(`data.size ${data.size} must equal the end of the last tensor ${cursor} (no trailing padding)`);
  }

  const outputs: TemplateOutput[] = [];
  const outputPaths = new Set<string>();
  for (const entry of list(raw["outputs"], "outputs")) {
    if (!isRecord(entry) || !plainName(entry["path"])) {
      return fail("outputs entries must be objects with a plain file name path");
    }
    const path = entry["path"];
    if (path === MANIFEST_NAME) {
      fail("outputs must not list manifest.json");
    }
    if (outputPaths.has(path)) {
      fail(`outputs lists ${path} more than once`);
    }
    outputPaths.add(path);
    const output: TemplateOutput = {
      path,
      size: integer(entry["size"], `outputs ${path} size`),
      sha256: digest(entry["sha256"], `outputs ${path} sha256`),
    };
    outputs.push(output);
    // Every output has exactly one origin, whose recorded size and digest it equals.
    if (path === dataPath) {
      if (output.size !== data.size || output.sha256 !== data.sha256) {
        fail(`outputs ${path} does not match data.size/data.sha256`);
      }
      continue;
    }
    const templated = byOutput.get(path);
    const sourced = sourceFiles.find((file) => file.path === path);
    const origin = templated ?? sourced;
    if (origin === undefined || (templated !== undefined && sourced !== undefined)) {
      fail(`output ${path} has no unambiguous origin (template file, data, or source file)`);
    } else if (origin.size !== output.size || origin.sha256 !== output.sha256) {
      fail(`output ${path} does not match the recorded size/sha256 of its origin`);
    }
  }
  for (const required of [GRAPH_NAME, PREPROCESSING_NAME, dataPath, "tokenizer.json", "tokenizer_config.json"]) {
    if (!outputPaths.has(required)) {
      fail(`outputs must include ${required}`);
    }
  }
  for (const file of templateFiles) {
    if (file.output !== MANIFEST_NAME && !outputPaths.has(file.output)) {
      fail(`template file ${file.path} produces ${file.output}, which is not listed in outputs`);
    }
  }

  return { model, revision, sourceFiles, templateFiles, data, outputs };
}

async function hashFile(path: string): Promise<{ bytes: number; sha256: string } | undefined> {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | O_NOFOLLOW);
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    if (code === "ENOENT" || code === "ELOOP" || code === "EMLINK") {
      return undefined;
    }
    throw error;
  }
  try {
    if (!(await handle.stat()).isFile()) {
      return undefined;
    }
    const hash = createHash("sha256");
    let bytes = 0;
    for await (const chunk of handle.createReadStream({ autoClose: false })) {
      hash.update(chunk as Buffer);
      bytes += (chunk as Buffer).length;
    }
    return { bytes, sha256: hash.digest("hex") };
  } finally {
    await handle.close();
  }
}

const loaded = new Map<string, Promise<OnnxTemplate>>();

/**
 * Loads and fully verifies the template for `model` at `revision`: structure, the size and SHA-256 of every
 * template file, and agreement between the manifest, the outputs list and the data layout. Returns `undefined`
 * when the revision is not the curated pin or no template directory exists for it. Verified results are cached
 * per template identity for the life of the process.
 */
export async function loadTemplate(
  templatesDir: string,
  model: KnownModel,
  revision: string,
): Promise<OnnxTemplate | undefined> {
  if (revision !== model.pinnedRevision) {
    return undefined;
  }
  const directory = templateDirectory(templatesDir, model, revision);
  let text: string;
  try {
    text = await readFile(join(directory, TEMPLATE_NAME), "utf8");
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    if (code === "ENOENT" || code === "ENOTDIR") {
      return undefined;
    }
    throw error;
  }
  const key = `${directory}\n${createHash("sha256").update(text).digest("hex")}`;
  let pending = loaded.get(key);
  if (pending === undefined) {
    pending = verifyTemplate(directory, text, model, revision);
    loaded.set(key, pending);
    pending.catch(() => loaded.delete(key));
  }
  return pending;
}

async function verifyTemplate(
  directory: string,
  text: string,
  model: KnownModel,
  revision: string,
): Promise<OnnxTemplate> {
  const fail = (detail: string): never => {
    throw new TemplateError(`ONNX template for '${model.modelId}' at ${revision} is invalid: ${detail}`);
  };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return fail("template.json is not valid JSON");
  }
  const parsed = parseTemplate(raw, model, revision);
  for (const file of parsed.templateFiles) {
    const actual = await hashFile(join(directory, file.path));
    if (actual === undefined) {
      fail(`template file ${file.path} is missing or not a regular file`);
    } else if (actual.bytes !== file.size || actual.sha256 !== file.sha256) {
      fail(`template file ${file.path} does not match its recorded size/sha256`);
    }
  }
  const manifestFile = parsed.templateFiles.find((file) => file.output === MANIFEST_NAME) as TemplateFile;
  let manifestRaw: unknown;
  try {
    manifestRaw = JSON.parse(await readFile(join(directory, manifestFile.path), "utf8"));
  } catch {
    return fail("the manifest template is not valid JSON");
  }
  const manifest = parseManifest(manifestRaw, model, revision);
  const listed = new Map(manifest.files.map((file) => [file.path, file]));
  const mismatched = parsed.outputs.some((output) => {
    const entry = listed.get(output.path);
    return entry === undefined || entry.size !== output.size || entry.sha256 !== output.sha256;
  });
  if (mismatched || listed.size !== parsed.outputs.length) {
    fail("manifest files do not equal the template outputs");
  }
  if (manifest.externalData.length !== 1 || manifest.externalData[0] !== parsed.data.path) {
    fail(`manifest graph.external_data must be exactly [${JSON.stringify(parsed.data.path)}]`);
  }
  return {
    ...parsed,
    directory,
    manifest,
    templateSha256: createHash("sha256").update(text).digest("hex"),
    templateFormatVersion: TEMPLATE_FORMAT_VERSION,
  };
}

/** Every valid template found for the curated models; invalid templates are reported through `onInvalid`. */
export async function listTemplates(
  templatesDir: string,
  models: readonly KnownModel[] = KNOWN_MODELS,
  onInvalid: (model: KnownModel, error: unknown) => void = () => undefined,
): Promise<OnnxTemplate[]> {
  const found: OnnxTemplate[] = [];
  let present: string[];
  try {
    present = await readdir(templatesDir);
  } catch {
    return found;
  }
  for (const model of models) {
    if (!present.includes(repositoryDirectoryName(model.repository))) {
      continue;
    }
    try {
      const template = await loadTemplate(templatesDir, model, model.pinnedRevision);
      if (template !== undefined) {
        found.push(template);
      }
    } catch (error) {
      onInvalid(model, error);
    }
  }
  return found;
}
