import { createReadStream, statSync } from "node:fs";
import { createServer, type IncomingHttpHeaders, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

export type Behavior = { status: number } | "reset" | "truncate" | "stall" | "corrupt" | undefined;

export interface MockRequest {
  server: "hub" | "cdn";
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
}

export interface MockHfOptions {
  repository: string;
  revision: string;
  /** Files served at `/{repository}/resolve/{revision}/{name}`. */
  files: Record<string, Buffer | { path: string }>;
  /** Require `Authorization: Bearer <token>` on the hub origin. */
  token?: string;
  /** Redirect file downloads to a second origin (as the real hub redirects to its CDN). */
  cdn?: boolean;
  /** Decides how an attempt to download `name` (1-based) behaves. */
  behavior?: (name: string, attempt: number) => Behavior;
  /** Response of `GET /api/models/{repository}`. */
  sha?: string | undefined;
  /** Per-repository shas for `GET /api/models/{repo}`; takes precedence over `sha`. */
  shas?: Record<string, string>;
  /** Repository ids listed by `GET /api/models`, split in pages of `pageSize` with Link headers. */
  listing?: string[];
  pageSize?: number;
  /** Per-response delay before the first byte, in ms. */
  delayMs?: number;
}

export interface MockHf {
  endpoint: string;
  requests: MockRequest[];
  attempts: Map<string, number>;
  close(): Promise<void>;
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve((server.address() as AddressInfo).port)));
}

export async function startMockHf(options: MockHfOptions): Promise<MockHf> {
  const requests: MockRequest[] = [];
  const attempts = new Map<string, number>();
  const sockets = new Set<import("node:net").Socket>();
  const track = (server: Server): void => {
    server.on("connection", (socket) => {
      sockets.add(socket);
      socket.once("close", () => sockets.delete(socket));
    });
  };
  const record = (which: "hub" | "cdn", request: IncomingMessage): void => {
    requests.push({ server: which, method: request.method ?? "", url: request.url ?? "", headers: request.headers });
  };

  const serveFile = (name: string, response: ServerResponse): void => {
    const entry = options.files[name];
    if (entry === undefined) {
      response.writeHead(404).end("not found");
      return;
    }
    if (!Buffer.isBuffer(entry)) {
      // Large real files are streamed from disk.
      response.writeHead(200, { "content-length": statSync(entry.path).size });
      createReadStream(entry.path).pipe(response);
      return;
    }
    const content = entry;
    const attempt = (attempts.get(name) ?? 0) + 1;
    attempts.set(name, attempt);
    const behavior = options.behavior?.(name, attempt);
    const send = (): void => {
      if (behavior !== undefined && typeof behavior === "object") {
        response.writeHead(behavior.status, { "retry-after": "0" }).end("error");
        return;
      }
      if (behavior === "reset") {
        response.writeHead(200, { "content-length": content.length });
        response.write(content.subarray(0, Math.floor(content.length / 2)));
        setTimeout(() => response.destroy(), 5);
        return;
      }
      if (behavior === "truncate") {
        // Ends the stream cleanly but short, without a Content-Length to give it away.
        response.writeHead(200);
        response.end(content.subarray(0, Math.max(0, content.length - 1)));
        return;
      }
      if (behavior === "stall") {
        response.writeHead(200, { "content-length": content.length });
        response.write(content.subarray(0, 1));
        return;
      }
      const body = Buffer.from(content);
      if (behavior === "corrupt") {
        body[0] = (body[0] as number) ^ 0xff;
      }
      response.writeHead(200, { "content-length": body.length });
      // Several chunks, so streaming hash and size accounting are exercised.
      const step = Math.max(1, Math.ceil(body.length / 4));
      for (let offset = 0; offset < body.length; offset += step) {
        response.write(body.subarray(offset, offset + step));
      }
      response.end();
    };
    if (options.delayMs !== undefined) {
      setTimeout(send, options.delayMs);
    } else {
      send();
    }
  };

  const cdn = createServer((request, response) => {
    record("cdn", request);
    const name = decodeURIComponent((request.url ?? "").replace(/^\/blob\//, ""));
    serveFile(name, response);
  });
  track(cdn);
  const cdnPort = options.cdn ? await listen(cdn) : 0;

  const hub = createServer((request, response) => {
    record("hub", request);
    const url = new URL(request.url ?? "/", "http://hub");
    if (options.token !== undefined && request.headers.authorization !== `Bearer ${options.token}`) {
      response.writeHead(401).end("unauthorized");
      return;
    }
    const prefix = `/${options.repository}/resolve/${options.revision}/`;
    if (url.pathname.startsWith(prefix)) {
      const name = decodeURIComponent(url.pathname.slice(prefix.length));
      if (options.cdn) {
        response.writeHead(302, { location: `http://127.0.0.1:${cdnPort}/blob/${encodeURIComponent(name)}` }).end();
      } else {
        serveFile(name, response);
      }
      return;
    }
    if (url.pathname.startsWith("/api/models/")) {
      const repository = url.pathname.slice("/api/models/".length);
      const sha = options.shas?.[repository] ?? (repository === options.repository ? options.sha : undefined);
      if (sha === undefined) {
        response.writeHead(404).end("{}");
      } else {
        response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ id: repository, sha }));
      }
      return;
    }
    if (url.pathname === "/api/models" && options.listing !== undefined) {
      const size = options.pageSize ?? options.listing.length;
      const page = Number(url.searchParams.get("page") ?? "0");
      const slice = options.listing.slice(page * size, (page + 1) * size);
      const headers: Record<string, string> = { "content-type": "application/json" };
      if ((page + 1) * size < options.listing.length) {
        const next = new URLSearchParams(url.searchParams);
        next.set("page", String(page + 1));
        headers["link"] = `<http://127.0.0.1:${(hub.address() as AddressInfo).port}/api/models?${next.toString()}>; rel="next"`;
      }
      response.writeHead(200, headers).end(JSON.stringify(slice.map((id) => ({ id, modelId: id }))));
      return;
    }
    response.writeHead(404).end("not found");
  });
  track(hub);
  const port = await listen(hub);
  return {
    endpoint: `http://127.0.0.1:${port}`,
    requests,
    attempts,
    close: async () => {
      for (const socket of sockets) {
        socket.destroy();
      }
      await Promise.all([new Promise((resolve) => hub.close(resolve)), options.cdn ? new Promise((resolve) => cdn.close(resolve)) : undefined]);
    },
  };
}
