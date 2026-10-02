import { spawn, type ChildProcess } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

import { GlinerClient } from "./index.js";

export interface GlinerSidecarOptions {
  command?: string;
  commandArguments?: string[];
  host?: string;
  port?: number;
  device?: string;
  runnerArguments?: string[];
  startupTimeoutMs?: number;
  shutdownTimeoutMs?: number;
  diagnosticLimitBytes?: number;
  env?: NodeJS.ProcessEnv;
  fetch?: typeof globalThis.fetch;
}

export class GlinerSidecarError extends Error {
  readonly diagnostics: string;

  constructor(message: string, diagnostics = "") {
    super(diagnostics.length > 0 ? `${message}\n${diagnostics}` : message);
    this.name = "GlinerSidecarError";
    this.diagnostics = diagnostics;
  }
}

export class GlinerSidecar {
  readonly baseUrl: string;
  readonly client: GlinerClient;
  private readonly options: Required<
    Pick<
      GlinerSidecarOptions,
      | "command"
      | "commandArguments"
      | "host"
      | "port"
      | "device"
      | "runnerArguments"
      | "startupTimeoutMs"
      | "shutdownTimeoutMs"
      | "diagnosticLimitBytes"
    >
  > &
    Pick<GlinerSidecarOptions, "env">;
  private readonly fetchImpl: typeof globalThis.fetch;
  private child: ChildProcess | undefined;
  private diagnostics = "";
  private exit:
    | { code: number | null; signal: NodeJS.Signals | null; error?: Error }
    | undefined;

  constructor(options: GlinerSidecarOptions = {}) {
    const host = options.host ?? "127.0.0.1";
    const port = options.port ?? 8090;
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new RangeError("port must be an integer between 1 and 65535");
    }
    this.options = {
      command: options.command ?? "gliner-runner",
      commandArguments: options.commandArguments ?? [],
      host,
      port,
      device: options.device ?? "cpu",
      runnerArguments: options.runnerArguments ?? [],
      startupTimeoutMs: options.startupTimeoutMs ?? 30_000,
      shutdownTimeoutMs: options.shutdownTimeoutMs ?? 5_000,
      diagnosticLimitBytes: options.diagnosticLimitBytes ?? 16_384,
      ...(options.env === undefined ? {} : { env: options.env }),
    };
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof this.fetchImpl !== "function") {
      throw new TypeError("A Fetch API implementation is required");
    }
    this.baseUrl = `http://${host}:${port}`;
    this.client = new GlinerClient({ baseUrl: this.baseUrl, fetch: this.fetchImpl });
  }

  get running(): boolean {
    return this.child !== undefined && this.exit === undefined;
  }

  async start(): Promise<void> {
    if (this.child !== undefined) {
      throw new GlinerSidecarError("sidecar has already been started");
    }
    const child = spawn(
      this.options.command,
      [
        ...this.options.commandArguments,
        "serve",
        "--host",
        this.options.host,
        "--port",
        String(this.options.port),
        "--device",
        this.options.device,
        ...this.options.runnerArguments,
      ],
      {
        env: this.options.env ?? process.env,
        shell: false,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      },
    );
    this.child = child;
    child.stdout?.on("data", (chunk: Buffer) => this.recordDiagnostic(chunk));
    child.stderr?.on("data", (chunk: Buffer) => this.recordDiagnostic(chunk));
    child.once("error", (error) => {
      this.exit = { code: null, signal: null, error };
    });
    child.once("exit", (code, signal) => {
      this.exit = { code, signal };
    });

    const deadline = Date.now() + this.options.startupTimeoutMs;
    while (Date.now() < deadline) {
      if (this.exit !== undefined) {
        throw this.exitError("sidecar exited before becoming ready");
      }
      try {
        const response = await this.fetchImpl(`${this.baseUrl}/healthz`, {
          signal: AbortSignal.timeout(500),
        });
        if (response.ok) {
          return;
        }
      } catch {
        // Readiness polling treats transport failures as not-ready until timeout.
      }
      await delay(50);
    }
    await this.stop();
    throw new GlinerSidecarError(
      `sidecar did not become ready within ${this.options.startupTimeoutMs} ms`,
      this.diagnostics,
    );
  }

  async stop(): Promise<void> {
    const child = this.child;
    if (child === undefined || this.exit !== undefined) {
      return;
    }
    child.kill();
    const exited = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), this.options.shutdownTimeoutMs);
      child.once("exit", () => {
        clearTimeout(timeout);
        resolve(true);
      });
    });
    if (!exited && this.exit === undefined) {
      child.kill("SIGKILL");
      await new Promise<void>((resolve) => child.once("exit", () => resolve()));
    }
  }

  private recordDiagnostic(chunk: Buffer): void {
    this.diagnostics += chunk.toString("utf8");
    const excess = Buffer.byteLength(this.diagnostics) - this.options.diagnosticLimitBytes;
    if (excess > 0) {
      this.diagnostics = Buffer.from(this.diagnostics).subarray(excess).toString("utf8");
    }
  }

  private exitError(message: string): GlinerSidecarError {
    const state = this.exit;
    if (state?.error !== undefined) {
      return new GlinerSidecarError(`${message}: ${state.error.message}`, this.diagnostics);
    }
    return new GlinerSidecarError(
      `${message}: code=${String(state?.code)} signal=${String(state?.signal)}`,
      this.diagnostics,
    );
  }
}
