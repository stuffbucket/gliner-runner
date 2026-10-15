#!/usr/bin/env node
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";

import { runClassifyCommand, runDemoCommand, type ClassifyIo } from "./classify-cli.js";
import { loadComponents } from "./components.js";
import { runModelsCommand } from "./models-cli.js";
import { createServer } from "./server.js";
import {
  SERVE_USAGE,
  SettingsError,
  parseServeArguments,
  wantsHelp,
  type Settings,
} from "./settings.js";
import { runUpdateCommand } from "./update.js";
import { VERSION_FLAGS, formatVersion, readVersionInfo } from "./version.js";

const log = (message: string): void => {
  process.stderr.write(`[gliner-runner-onnx] ${message}\n`);
};

function usageError(message: string): number {
  process.stderr.write(`error: ${message}\n\n${SERVE_USAGE}`);
  return 2;
}

async function serve(argv: readonly string[]): Promise<number> {
  let settings: Settings;
  try {
    settings = parseServeArguments(argv, process.env);
  } catch (error) {
    if (error instanceof SettingsError) {
      return usageError(error.message);
    }
    throw error;
  }
  const server = createServer({ ...(await loadComponents(settings, { log })), settings, log });
  try {
    const address = await server.listen();
    log("Application startup complete.");
    log(`Listening on http://${settings.host}:${address.port}`);
  } catch (error) {
    log(`failed to listen on ${settings.host}:${settings.port}: ${(error as Error).message}`);
    await server.close();
    return 1;
  }

  return new Promise<number>((resolve) => {
    let stopping = false;
    const stop = (signal: string): void => {
      if (stopping) {
        return;
      }
      stopping = true;
      log(`Received ${signal}; shutting down`);
      setTimeout(() => {
        log("graceful shutdown timed out; forcing exit");
        process.exit(1);
      }, settings.shutdownTimeoutMs).unref();
      server.close().then(
        () => {
          log("Application shutdown complete");
          resolve(0);
        },
        (error: unknown) => {
          log(`shutdown failed: ${error instanceof Error ? error.message : String(error)}`);
          resolve(1);
        },
      );
    };
    process.on("SIGTERM", () => stop("SIGTERM"));
    process.on("SIGINT", () => stop("SIGINT"));
  });
}

async function models(argv: readonly string[]): Promise<number> {
  const controller = new AbortController();
  const interrupt = (): void => controller.abort();
  process.once("SIGINT", interrupt);
  process.once("SIGTERM", interrupt);
  try {
    return await runModelsCommand(argv, {
      env: process.env,
      interactive: process.stdin.isTTY === true,
      stdout: (text) => void process.stdout.write(text),
      stderr: (text) => void process.stderr.write(text),
      confirm: async (question) => {
        const reader = createInterface({ input: process.stdin, output: process.stderr });
        try {
          return /^\s*y(es)?\s*$/i.test(await reader.question(question));
        } finally {
          reader.close();
        }
      },
      signal: controller.signal,
    });
  } finally {
    process.off("SIGINT", interrupt);
    process.off("SIGTERM", interrupt);
  }
}

export const HELP = `gliner-runner classifies text into labels you choose. It runs on your computer.

Usage: gliner-runner <command> [options]

Commands:
  models install <id>   Download and prepare a model, for example decide-340m (one time);
                        several ids or --all install every curated model
  models list           Show installed models and their status (JSON)
  models scan           Look for model files already on this computer
  models prepare <id>   Prepare a model from a local copy of its files, offline
  classify              Classify one text into labels you choose
  demo                  Run a few example classifications
  serve                 Run a local web service for applications (http://127.0.0.1:8090)
  update check          Check whether a newer release is available
  --version             Print the version

Run 'gliner-runner <command> --help' for a command's options.
Start with: gliner-runner models install decide-340m, then gliner-runner demo
`;

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk as Uint8Array));
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function classification(run: typeof runClassifyCommand, argv: readonly string[]): Promise<number> {
  const controller = new AbortController();
  const interrupt = (): void => controller.abort();
  process.once("SIGINT", interrupt);
  process.once("SIGTERM", interrupt);
  const io: ClassifyIo = {
    env: process.env,
    stdout: (text) => void process.stdout.write(text),
    stderr: (text) => void process.stderr.write(text),
    stdinIsTerminal: process.stdin.isTTY === true,
    readStdin,
    signal: controller.signal,
  };
  try {
    return await run(argv, io);
  } finally {
    process.off("SIGINT", interrupt);
    process.off("SIGTERM", interrupt);
  }
}

async function main(argv: readonly string[]): Promise<number> {
  const [command, ...rest] = argv;
  if (command !== undefined && VERSION_FLAGS.includes(command)) {
    if (rest.length > 1 || (rest.length === 1 && rest[0] !== "--json")) {
      return usageError(`${command} only accepts --json`);
    }
    const info = readVersionInfo();
    process.stdout.write(rest.length === 1 ? `${JSON.stringify(info)}\n` : formatVersion(info));
    return 0;
  }
  if (command === "serve") {
    if (wantsHelp(rest)) {
      process.stdout.write(SERVE_USAGE);
      return 0;
    }
    return serve(rest);
  }
  if (command === "models") {
    return models(rest);
  }
  if (command === "classify") {
    return classification(runClassifyCommand, rest);
  }
  if (command === "demo") {
    return classification(runDemoCommand, rest);
  }
  if (command === "update") {
    return runUpdateCommand(
      rest,
      {
        env: process.env,
        appPath: fileURLToPath(import.meta.url),
        stdout: (text) => void process.stdout.write(text),
        stderr: (text) => void process.stderr.write(text),
      },
      readVersionInfo().version,
    );
  }
  if (command === undefined || command === "help" || wantsHelp([command])) {
    process.stdout.write(HELP);
    return command === undefined ? 2 : 0;
  }
  process.stderr.write(`error: unknown command: ${command}\n\n${HELP}`);
  return 2;
}

process.exitCode = await main(process.argv.slice(2));
