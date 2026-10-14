import { constants } from "node:fs";
import { open, rename, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { setTimeout as sleepFor } from "node:timers/promises";

import { RequestCancelledError } from "../errors.js";
import type { TemplateSourceFile } from "./templates.js";

export const DEFAULT_HF_ENDPOINT = "https://huggingface.co";
const MAX_REDIRECTS = 10;
const MAX_CATALOG_PAGES = 50;

export class HuggingFaceError extends Error {
  override readonly name = "HuggingFaceError";
  readonly status: number | undefined;
  readonly retryable: boolean;

  constructor(message: string, options: { status?: number; retryable: boolean }) {
    super(message);
    this.status = options.status;
    this.retryable = options.retryable;
  }
}

export interface HuggingFaceOptions {
  /** Hub base URL; defaults to `HF_ENDPOINT`, then `https://huggingface.co`. */
  endpoint?: string | undefined;
  /** Bearer token; defaults to `HF_TOKEN`. */
  token?: string | undefined;
  /** Test seam; defaults to the global `fetch`. */
  fetch?: typeof globalThis.fetch | undefined;
  /** Attempts per file, including the first (default 5). */
  maxAttempts?: number | undefined;
  /** First retry delay in ms, doubled each time up to `retryMaxMs` (defaults 500 and 8000). */
  retryBaseMs?: number | undefined;
  retryMaxMs?: number | undefined;
  /** A transfer that delivers no bytes for this long is aborted and retried (default 60000). */
  idleTimeoutMs?: number | undefined;
}

/** Endpoint and token from the standard Hugging Face environment variables. */
export function huggingFaceOptionsFromEnv(env: NodeJS.ProcessEnv = process.env): HuggingFaceOptions {
  const endpoint = env["HF_ENDPOINT"];
  const token = env["HF_TOKEN"];
  return {
    endpoint: endpoint === undefined || endpoint === "" ? undefined : endpoint,
    token: token === undefined || token === "" ? undefined : token,
  };
}

export interface DownloadProgress {
  file: string;
  /** Bytes of `file` received so far and its total. */
  fileBytes: number;
  fileTotal: number;
  /** Bytes received across all files and their total. */
  bytes: number;
  total: number;
}

export interface DownloadContext {
  signal?: AbortSignal | undefined;
  onProgress?: ((progress: DownloadProgress) => void) | undefined;
}

const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

function cancelled(): RequestCancelledError {
  return new RequestCancelledError("model download was cancelled");
}

/** A minimal Hugging Face Hub client: pinned file downloads and the catalog lookups behind `/v1/models/refresh`. */
export class HuggingFaceClient {
  readonly endpoint: string;
  readonly #token: string | undefined;
  readonly #fetch: typeof globalThis.fetch;
  readonly #maxAttempts: number;
  readonly #retryBaseMs: number;
  readonly #retryMaxMs: number;
  readonly #idleTimeoutMs: number;

  constructor(options: HuggingFaceOptions = {}) {
    const fromEnv = huggingFaceOptionsFromEnv();
    this.endpoint = (options.endpoint ?? fromEnv.endpoint ?? DEFAULT_HF_ENDPOINT).replace(/\/+$/, "");
    this.#token = "token" in options ? options.token : fromEnv.token;
    this.#fetch = options.fetch ?? globalThis.fetch;
    this.#maxAttempts = options.maxAttempts ?? 5;
    this.#retryBaseMs = options.retryBaseMs ?? 500;
    this.#retryMaxMs = options.retryMaxMs ?? 8000;
    this.#idleTimeoutMs = options.idleTimeoutMs ?? 60_000;
  }

  /** `{endpoint}/{repository}/resolve/{revision}/{path}`. */
  sourceUrl(repository: string, revision: string, path: string): string {
    return `${this.endpoint}/${repository}/resolve/${revision}/${path.split("/").map(encodeURIComponent).join("/")}`;
  }

  /**
   * GET with manual redirect handling: the bearer token goes only to the configured endpoint's origin and is
   * dropped as soon as a redirect leaves it (CDN URLs are pre-signed).
   */
  async #get(url: string, signal: AbortSignal, accept: string): Promise<Response> {
    const origin = new URL(this.endpoint).origin;
    let current = url;
    for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
      const headers: Record<string, string> = { accept, "accept-encoding": "identity", "user-agent": "gliner-runner-onnx" };
      if (this.#token !== undefined && new URL(current).origin === origin) {
        headers["authorization"] = `Bearer ${this.#token}`;
      }
      const response = await this.#fetch(current, { headers, redirect: "manual", signal });
      if (response.status >= 300 && response.status < 400) {
        const location = response.headers.get("location");
        await response.body?.cancel().catch(() => undefined);
        if (location === null) {
          throw new HuggingFaceError(`HTTP ${response.status} without a Location header from ${current}`, {
            status: response.status,
            retryable: false,
          });
        }
        current = new URL(location, current).toString();
        continue;
      }
      return response;
    }
    throw new HuggingFaceError(`too many redirects fetching ${url}`, { retryable: false });
  }

  #statusError(response: Response, url: string): HuggingFaceError {
    const hint =
      response.status === 401 || response.status === 403
        ? " (the repository may be gated or private; set HF_TOKEN)"
        : "";
    return new HuggingFaceError(`HTTP ${response.status} fetching ${url}${hint}`, {
      status: response.status,
      retryable: TRANSIENT_STATUS.has(response.status),
    });
  }

  /** Downloads one pinned file into `destination`, verifying size and SHA-256; retries transient failures. */
  async downloadFile(
    repository: string,
    revision: string,
    file: Pick<TemplateSourceFile, "path" | "size" | "sha256">,
    destination: string,
    context: DownloadContext & { baseBytes?: number; totalBytes?: number } = {},
  ): Promise<void> {
    const url = this.sourceUrl(repository, revision, file.path);
    let lastError: unknown;
    for (let attempt = 1; attempt <= this.#maxAttempts; attempt += 1) {
      if (context.signal?.aborted) {
        throw cancelled();
      }
      try {
        await this.#attempt(url, file, destination, context);
        return;
      } catch (error) {
        lastError = error;
        await rm(`${destination}.partial`, { force: true });
        if (context.signal?.aborted) {
          throw cancelled();
        }
        const retryable = !(error instanceof HuggingFaceError) || error.retryable;
        if (!retryable || attempt === this.#maxAttempts) {
          break;
        }
        const delay = Math.min(this.#retryMaxMs, this.#retryBaseMs * 2 ** (attempt - 1));
        try {
          await sleepFor(delay * (0.75 + Math.random() * 0.5), undefined, context.signal === undefined ? {} : { signal: context.signal });
        } catch {
          throw cancelled();
        }
      }
    }
    const reason = lastError instanceof Error ? lastError.message : String(lastError);
    if (lastError instanceof HuggingFaceError && !lastError.retryable) {
      throw lastError;
    }
    throw new HuggingFaceError(`failed to download ${file.path} after ${this.#maxAttempts} attempts: ${reason}`, {
      retryable: false,
    });
  }

  async #attempt(
    url: string,
    file: Pick<TemplateSourceFile, "path" | "size" | "sha256">,
    destination: string,
    context: DownloadContext & { baseBytes?: number; totalBytes?: number },
  ): Promise<void> {
    const controller = new AbortController();
    const onAbort = (): void => controller.abort();
    context.signal?.addEventListener("abort", onAbort, { once: true });
    let stalled = false;
    let timer: NodeJS.Timeout | undefined;
    const touch = (): void => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        stalled = true;
        controller.abort();
      }, this.#idleTimeoutMs);
    };
    touch();
    const partial = `${destination}.partial`;
    try {
      const response = await this.#get(url, controller.signal, "application/octet-stream");
      if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        throw this.#statusError(response, url);
      }
      const declared = response.headers.get("content-length");
      if (declared !== null && !response.headers.has("content-encoding") && Number(declared) !== file.size) {
        await response.body?.cancel().catch(() => undefined);
        throw new HuggingFaceError(
          `${file.path} is ${declared} bytes on the server, the pinned revision records ${file.size}`,
          { retryable: false },
        );
      }
      if (response.body === null) {
        throw new HuggingFaceError(`empty response body for ${url}`, { retryable: true });
      }
      const hash = createHash("sha256");
      let received = 0;
      const output = await open(partial, constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC, 0o644);
      try {
        for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
          touch();
          received += chunk.length;
          if (received > file.size) {
            throw new HuggingFaceError(`${file.path} exceeds the pinned size of ${file.size} bytes`, { retryable: false });
          }
          hash.update(chunk);
          let written = 0;
          while (written < chunk.length) {
            written += (await output.write(chunk, written, chunk.length - written)).bytesWritten;
          }
          context.onProgress?.({
            file: file.path,
            fileBytes: received,
            fileTotal: file.size,
            bytes: (context.baseBytes ?? 0) + received,
            total: context.totalBytes ?? file.size,
          });
        }
        await output.sync();
      } finally {
        await output.close();
      }
      if (received !== file.size) {
        throw new HuggingFaceError(`${file.path} is ${received} bytes, the pinned revision records ${file.size}`, {
          retryable: false,
        });
      }
      const digest = hash.digest("hex");
      if (digest !== file.sha256) {
        throw new HuggingFaceError(
          `${file.path} failed verification: sha256 ${digest} does not match the pinned ${file.sha256}`,
          { retryable: false },
        );
      }
      await rename(partial, destination);
    } catch (error) {
      if (stalled && !context.signal?.aborted) {
        throw new HuggingFaceError(`no data received for ${this.#idleTimeoutMs} ms fetching ${url}`, { retryable: true });
      }
      throw error;
    } finally {
      clearTimeout(timer);
      context.signal?.removeEventListener("abort", onAbort);
    }
  }

  /** Downloads every file into `directory` (flat), sequentially; the first failure aborts the rest. */
  async downloadAll(
    repository: string,
    revision: string,
    files: readonly Pick<TemplateSourceFile, "path" | "size" | "sha256">[],
    directory: string,
    context: DownloadContext = {},
  ): Promise<void> {
    const total = files.reduce((sum, file) => sum + file.size, 0);
    let base = 0;
    for (const file of files) {
      await this.downloadFile(repository, revision, file, join(directory, file.path), {
        ...context,
        baseBytes: base,
        totalBytes: total,
      });
      base += file.size;
    }
  }

  async #json(url: string, signal?: AbortSignal): Promise<{ body: unknown; next: string | undefined }> {
    const controller = AbortSignal.any([AbortSignal.timeout(30_000), ...(signal === undefined ? [] : [signal])]);
    const response = await this.#get(url, controller, "application/json");
    if (!response.ok) {
      await response.body?.cancel().catch(() => undefined);
      throw this.#statusError(response, url);
    }
    const link = response.headers.get("link");
    const next = link === null ? undefined : /<([^>]+)>\s*;\s*rel="next"/.exec(link)?.[1];
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new HuggingFaceError(`invalid JSON from ${url}`, { retryable: false });
    }
    return { body, next };
  }

  /** `GET {endpoint}/api/models/{repository}` -> `sha`. */
  async modelSha(repository: string, signal?: AbortSignal): Promise<string> {
    const { body } = await this.#json(`${this.endpoint}/api/models/${repository}`, signal);
    const sha = typeof body === "object" && body !== null ? (body as { sha?: unknown }).sha : undefined;
    if (typeof sha !== "string" || sha === "") {
      throw new HuggingFaceError(`model ${repository} has no revision sha in the Hub response`, { retryable: false });
    }
    return sha;
  }

  /** `GET {endpoint}/api/models?author=...&search=...` -> repository ids (all pages). */
  async listModels(author: string, search: string, signal?: AbortSignal): Promise<string[]> {
    const ids: string[] = [];
    let url: string | undefined = `${this.endpoint}/api/models?${new URLSearchParams({ author, search }).toString()}`;
    for (let page = 0; url !== undefined; page += 1) {
      if (page >= MAX_CATALOG_PAGES) {
        throw new HuggingFaceError(`model listing exceeded ${MAX_CATALOG_PAGES} pages`, { retryable: false });
      }
      const { body, next } = await this.#json(url, signal);
      if (!Array.isArray(body)) {
        throw new HuggingFaceError("model listing is not a JSON array", { retryable: false });
      }
      for (const entry of body) {
        const record = entry as { id?: unknown; modelId?: unknown } | null;
        const id = record?.id ?? record?.modelId;
        if (typeof id === "string") {
          ids.push(id);
        }
      }
      url = next === undefined ? undefined : new URL(next, url).toString();
    }
    return ids;
  }
}
