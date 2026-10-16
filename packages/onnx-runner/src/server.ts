import {
  createServer as createHttpServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";

import type { BackendFactory } from "./backend.js";
import type { ArtifactCatalog } from "./catalog.js";
import type { ModelProvisioner } from "./downloads.js";
import { RequestCancelledError, RequestTooLargeError } from "./errors.js";
import { createRunnerHandler, type RunnerReply } from "./handler.js";
import type { Runtime } from "./runtime.js";
import { resolveSettings, type Settings } from "./settings.js";

export interface ServerOptions {
  catalog: ArtifactCatalog;
  /** Registered backends; a request naming any other backend is rejected without fallback. */
  backends: readonly BackendFactory[];
  settings?: Partial<Settings>;
  /**
   * Fetches and prepares models for `POST /v1/models/{id}/downloads`. Without one the server validates download
   * requests and then declines them with 409 `unsupported_configuration`.
   */
  provisioner?: ModelProvisioner;
  /** Overrides `settings.autoDownload`: inference for a missing, installable model waits for a shared download job. */
  autoDownload?: boolean;
  /** Diagnostic sink; never receives request text, schemas or outputs. */
  log?: (message: string) => void;
}

export interface RunnerServer {
  readonly runtime: Runtime;
  readonly http: Server;
  listen(): Promise<AddressInfo>;
  /** Stops listening, lets in-flight requests finish, then unloads the model. */
  close(): Promise<void>;
}

export function createServer(options: ServerOptions): RunnerServer {
  const settings = resolveSettings(options.settings);
  if (!Number.isSafeInteger(settings.maxBodyBytes) || settings.maxBodyBytes < 1) {
    throw new RangeError(`maxBodyBytes must be a positive integer, got ${settings.maxBodyBytes}`);
  }
  const log = options.log ?? (() => undefined);
  const handler = createRunnerHandler({
    catalog: options.catalog,
    backends: options.backends,
    settings,
    provisioner: options.provisioner,
    autoDownload: options.autoDownload,
    log,
  });
  const runtime = handler.runtime;

  const exchanges = new Set<Promise<void>>();
  const http = createHttpServer((request, response) => {
    void serve(request, response);
  });

  async function serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const controller = new AbortController();
    const finished = new Promise<void>((resolve) => {
      response.once("close", () => {
        if (!response.writableFinished) {
          controller.abort();
        }
        exchanges.delete(finished);
        resolve();
      });
    });
    exchanges.add(finished);
    const origin = request.headers.origin;
    const cors = origin !== undefined && settings.corsOrigins.includes(origin);
    let reply: RunnerReply;
    try {
      reply =
        cors && request.method === "OPTIONS"
          ? {
              status: 204,
              body: "",
              contentType: "text/plain",
              headers: {
                "access-control-allow-methods": "GET, POST, OPTIONS",
                "access-control-allow-headers": "content-type",
                "access-control-max-age": "600",
              },
            }
          : await handler.handle({
              method: request.method ?? "GET",
              path: request.url ?? "/",
              body: () => readBody(request, settings.maxBodyBytes),
              signal: controller.signal,
            });
    } catch {
      // The handler only rejects for a cancelled request: the client is gone, so there is nobody to answer.
      response.destroy();
      return;
    }
    if (response.destroyed) {
      return;
    }
    const body = Buffer.from(reply.body);
    response.writeHead(reply.status, {
      "content-type": reply.contentType,
      "content-length": body.length,
      ...reply.headers,
      ...(cors ? { "access-control-allow-origin": origin, vary: "Origin" } : {}),
    });
    response.end(body);
  }

  let closing: Promise<void> | undefined;
  return {
    runtime,
    http,
    listen: () =>
      new Promise<AddressInfo>((resolve, reject) => {
        http.once("error", reject);
        http.listen({ host: settings.host, port: settings.port }, () => {
          http.off("error", reject);
          resolve(http.address() as AddressInfo);
        });
      }),
    close: () => (closing ??= shutdown()),
  };

  async function shutdown(): Promise<void> {
    // Waiting inference requests fail fast instead of holding shutdown open for a download that is being cancelled.
    handler.abortJobs();
    if (http.listening) {
      const stopped = new Promise<void>((resolve) => http.close(() => resolve()));
      const drained = (async () => {
        while (exchanges.size > 0) {
          await Promise.allSettled([...exchanges]);
        }
      })();
      await Promise.race([drained, sleep(settings.shutdownTimeoutMs / 2, undefined, { ref: false })]);
      http.closeAllConnections();
      await stopped;
    }
    await handler.close();
  }
}

/**
 * Reads the whole body, failing with `RequestTooLargeError` as soon as it is known to exceed `limit` bytes (from
 * `content-length` or while streaming). The rest of an oversize body is discarded without being buffered; the
 * stream is not destroyed, because that would also tear down the socket the 413 reply must still be written to.
 */
function readBody(request: IncomingMessage, limit: number): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const declared = Number(request.headers["content-length"]);
    if (Number.isFinite(declared) && declared > limit) {
      request.resume();
      reject(new RequestTooLargeError(limit));
      return;
    }
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const fail = (error: unknown): void => {
      if (!settled) {
        settled = true;
        chunks.length = 0;
        reject(error);
      }
    };
    request.on("data", (chunk: Buffer) => {
      if (settled) {
        return;
      }
      size += chunk.length;
      if (size > limit) {
        fail(new RequestTooLargeError(limit));
      } else {
        chunks.push(chunk);
      }
    });
    request.once("end", () => {
      if (!settled) {
        settled = true;
        resolve(Buffer.concat(chunks).toString("utf8"));
      }
    });
    request.once("error", fail);
    request.once("close", () => {
      if (!request.complete) {
        fail(new RequestCancelledError("request body was not fully received"));
      }
    });
  });
}
