import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE = fileURLToPath(new URL("../../src/", import.meta.url));

/** Runtime import specifiers of a TypeScript module: type-only imports and exports are erased, so they are skipped. */
function runtimeImports(text: string): string[] {
  const specifiers: string[] = [];
  for (const match of text.matchAll(/^(?:import|export)\s+(type\s+)?(?:[^;]*?\sfrom\s+)?"([^"]+)"/gms)) {
    if (match[1] === undefined) {
      specifiers.push(match[2] as string);
    }
  }
  for (const match of text.matchAll(/\bimport\(\s*"([^"]+)"\s*\)/g)) {
    specifiers.push(match[1] as string);
  }
  return specifiers;
}

test("the portable entry point (the browser build) never loads node:* or onnxruntime-node", () => {
  // `#platform` resolves to the browser slot under the `browser` condition, exactly as a bundler resolves it.
  const resolve = (from: string, specifier: string): string | undefined => {
    if (specifier === "#platform") {
      return join(SOURCE, "platform", "slot.ts");
    }
    if (specifier.startsWith(".")) {
      return join(dirname(from), specifier.replace(/\.js$/, ".ts"));
    }
    return undefined;
  };
  const seen = new Set<string>();
  const offending: string[] = [];
  const queue = [join(SOURCE, "portable.ts")];
  while (queue.length > 0) {
    const file = queue.pop() as string;
    if (seen.has(file)) {
      continue;
    }
    seen.add(file);
    assert.ok(existsSync(file), `${relative(SOURCE, file)} exists`);
    for (const specifier of runtimeImports(readFileSync(file, "utf8"))) {
      if (specifier.startsWith("node:") || specifier === "onnxruntime-node") {
        offending.push(`${relative(SOURCE, file)} imports ${specifier}`);
      }
      const next = resolve(file, specifier);
      if (next !== undefined) {
        queue.push(next);
      }
    }
  }
  assert.deepEqual(offending, []);
  assert.ok(seen.has(join(SOURCE, "handler.ts")) && seen.has(join(SOURCE, "onnx", "install.ts")), "the walk reached the runner");
  assert.ok(!seen.has(join(SOURCE, "platform", "node.ts")), "the Node platform is not part of the browser build");
});
