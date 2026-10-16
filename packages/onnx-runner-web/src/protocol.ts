import type { InstallProgress } from "@gliner-runner/onnx-runner/portable";

import type { ExecutionProvider } from "./platform.js";

/** Options the page may pass when it starts the worker (a test endpoint, a forced execution provider). */
export interface HostInit {
  hfEndpoint?: string | undefined;
  executionProvider?: "auto" | ExecutionProvider | undefined;
}

/** Page -> worker. The worker starts the runner on `init`. */
export type HostRequest =
  | { type: "init"; options: HostInit }
  | { type: "request"; id: number; method: string; path: string; body?: string | undefined }
  | { type: "abort"; id: number }
  | { type: "delete-model"; id: number };

export type Unsupported = { state: "unsupported"; reason: string };

/** What the worker reports while it starts. `storage` is the origin's quota and usage in bytes. */
export type HostStatus =
  | { state: "starting" }
  | ({ state: "ready" } & { storage: { quota: number; usage: number }; persisted: boolean })
  | Unsupported
  | { state: "failed"; reason: string };

/** Worker -> page. */
export type HostEvent =
  | { type: "status"; status: HostStatus }
  | { type: "response"; id: number; status: number; contentType: string; body: string }
  | { type: "cancelled"; id: number }
  | { type: "deleted"; id: number }
  | { type: "progress"; progress: InstallProgress }
  | { type: "provider"; provider: ExecutionProvider; note?: string | undefined }
  | { type: "log"; message: string };
