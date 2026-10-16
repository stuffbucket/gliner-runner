import type { HostEvent, HostInit, HostRequest, HostStatus } from "./protocol.js";
import { createBrowserRunner, deleteModels, type BrowserRunner } from "./runner.js";

export interface HostOptions {
  /** Where the site serves the artifact templates. */
  templatesBaseUrl: string;
}

declare const self: DedicatedWorkerGlobalScope;

/** Why this browser cannot run the model in a tab, or `undefined` when it can. */
async function unsupportedReason(): Promise<string | undefined> {
  if (typeof navigator.storage?.getDirectory !== "function") {
    return "This browser has no private file storage for websites (OPFS).";
  }
  if (typeof FileSystemFileHandle === "undefined" || typeof FileSystemFileHandle.prototype.createSyncAccessHandle !== "function") {
    return "This browser cannot write large files from a background worker.";
  }
  try {
    await navigator.storage.getDirectory();
  } catch {
    // Safari's Private Browsing (and some privacy settings) turn website storage off.
    return "Website storage is turned off here (private browsing, or a privacy setting).";
  }
  return undefined;
}

/**
 * Runs the gliner-runner in this dedicated worker and answers the page's requests (see `client.ts`). Everything
 * stays in the worker: the 1.9 GB download, the assembly, the weights and inference never touch the page thread.
 */
export function serveRunner(options: HostOptions): void {
  const post = (event: HostEvent): void => self.postMessage(event);
  const status = (value: HostStatus): void => post({ type: "status", status: value });
  const inFlight = new Map<number, AbortController>();

  let init: HostInit = {};
  let started: () => void = () => undefined;
  const initialized = new Promise<void>((resolve) => (started = resolve));

  const start = async (): Promise<BrowserRunner | undefined> => {
    await initialized;
    status({ state: "starting" });
    const reason = await unsupportedReason();
    if (reason !== undefined) {
      status({ state: "unsupported", reason });
      return undefined;
    }
    try {
      const runner = await createBrowserRunner({
        templatesBaseUrl: options.templatesBaseUrl,
        hfEndpoint: init.hfEndpoint,
        executionProvider: init.executionProvider,
        onProgress: (progress) => post({ type: "progress", progress }),
        onExecutionProvider: (provider, note) => post({ type: "provider", provider, note }),
        log: (message) => post({ type: "log", message }),
      });
      const { quota = 0, usage = 0 } = await navigator.storage.estimate();
      const persisted = (await navigator.storage.persisted?.().catch(() => false)) ?? false;
      status({ state: "ready", storage: { quota, usage }, persisted });
      return runner;
    } catch (error) {
      status({ state: "failed", reason: error instanceof Error ? error.message : String(error) });
      return undefined;
    }
  };

  let runner = start();

  self.onmessage = async ({ data }: MessageEvent<HostRequest>) => {
    if (data.type === "init") {
      init = data.options;
      started();
      return;
    }
    if (data.type === "abort") {
      inFlight.get(data.id)?.abort();
      return;
    }
    const current = await runner;
    if (data.type === "delete-model") {
      await current?.close();
      await deleteModels();
      runner = start();
      await runner;
      post({ type: "deleted", id: data.id });
      return;
    }
    if (current === undefined) {
      post({
        type: "response",
        id: data.id,
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "the in-browser runner is not available" }),
      });
      return;
    }
    const controller = new AbortController();
    inFlight.set(data.id, controller);
    try {
      const reply = await current.handler.handle({
        method: data.method,
        path: data.path,
        body: data.body,
        signal: controller.signal,
      });
      post({ type: "response", id: data.id, status: reply.status, contentType: reply.contentType, body: reply.body });
    } catch {
      post({ type: "cancelled", id: data.id });
    } finally {
      inFlight.delete(data.id);
    }
  };
}
