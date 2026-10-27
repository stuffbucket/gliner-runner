// Finds a model this page can use without downloading anything, the way the playground (Playground.jsx) reaches
// one: a runner on this computer (`gliner-runner serve`) over HTTP, or the model already installed in this
// browser, run in a worker. It never starts a download.
//
// The local runner is only asked when the visitor has used it from the playground before (or the address is in
// the page's ?runner=), so a first visit does not reach for this computer's network.
import { HttpTransport, WorkerTransport } from "@gliner-runner/onnx-runner-web/client";

const DEFAULT_ADDRESS = "http://127.0.0.1:8090";
// The playground's remembered choices (see KEYS in Playground.jsx).
const PLAYGROUND = { mode: "gliner-runner-playground-mode", address: "gliner-runner-playground-address" };

const stored = (key) => {
  try {
    return localStorage.getItem(key) ?? undefined;
  } catch {
    return undefined;
  }
};
const models = async (transport) => {
  const reply = await transport.request("GET", "/v1/models");
  if (reply.status !== 200) return [];
  try {
    return JSON.parse(reply.body).models ?? [];
  } catch {
    return [];
  }
};
const installed = (list, model) => list.some((entry) => entry.model_id === model && entry.available_locally);

async function onThisComputer(model) {
  const params = new URLSearchParams(location.search);
  const address = params.get("runner") ?? stored(PLAYGROUND.address);
  if (address === undefined && stored(PLAYGROUND.mode) !== "machine") return undefined;
  const transport = new HttpTransport((address ?? DEFAULT_ADDRESS).trim().replace(/\/+$/, ""));
  try {
    return installed(await models(transport), model) ? { transport, where: "computer" } : undefined;
  } catch {
    return undefined;
  }
}

/** Whether this browser has any model files at all, without starting the worker. */
async function anythingInstalled() {
  try {
    const root = await navigator.storage.getDirectory();
    await root.getDirectoryHandle("models");
    return true;
  } catch {
    return false;
  }
}

async function inThisBrowser(model) {
  if (!(await anythingInstalled())) return undefined;
  const worker = new Worker(new URL("../workers/runner.worker.ts", import.meta.url), { type: "module" });
  const transport = new WorkerTransport(worker);
  const state = await new Promise((resolve) => {
    const unsubscribe = transport.subscribe((event) => {
      if (event.type !== "status" || event.status.state === "starting") return;
      unsubscribe();
      resolve(event.status.state);
    });
  });
  if (state === "ready" && installed(await models(transport).catch(() => []), model)) return { transport, where: "browser" };
  transport.terminate();
  return undefined;
}

/** A transport to `model` and where it runs, or undefined when none is ready to use. */
export async function findModel(model) {
  return (await onThisComputer(model)) ?? (await inThisBrowser(model));
}
