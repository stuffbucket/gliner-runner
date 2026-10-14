import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { FACTORY_SPECIFIER, loadComponents, type RunnerComponents } from "../src/components.js";
import { defaultSettings } from "../src/settings.js";
import { FakeFactory, installedCatalog } from "./helpers.js";

const settings = { ...defaultSettings(), modelsDir: "/nowhere" };

function missingFactoryError(): Error {
  const path = fileURLToPath(new URL(`../src/${FACTORY_SPECIFIER.slice(2)}`, import.meta.url));
  return Object.assign(new Error(`Cannot find module '${path}' imported from cli.js`), {
    code: "ERR_MODULE_NOT_FOUND",
  });
}

test("without an ONNX factory the server starts with an empty catalog and no backends", async () => {
  const logs: string[] = [];
  const components = await loadComponents(settings, {
    log: (message) => logs.push(message),
    importModule: async () => {
      throw missingFactoryError();
    },
  });
  assert.deepEqual(components.backends, []);
  const inventory = await components.catalog.inventory();
  assert.equal(inventory.destination, "/nowhere");
  assert.equal(inventory.models.length, 3);
  assert.ok(inventory.models.every((model) => model.localRevisions.length === 0));
  assert.equal(logs.length, 1);
});

test("uses the components built by the factory module", async () => {
  const built: RunnerComponents = { catalog: installedCatalog(), backends: [new FakeFactory()] };
  let received: unknown;
  const components = await loadComponents(settings, {
    importModule: async (specifier) => {
      assert.equal(specifier, FACTORY_SPECIFIER);
      return {
        createComponents: async (given: unknown) => {
          received = given;
          return built;
        },
      };
    },
  });
  assert.equal(components, built);
  assert.equal(received, settings);
});

test("a factory that fails to import its own dependencies is not treated as missing", async () => {
  const failure = Object.assign(new Error("Cannot find package 'onnxruntime-node' imported from factory.js"), {
    code: "ERR_MODULE_NOT_FOUND",
  });
  await assert.rejects(
    loadComponents(settings, {
      importModule: async () => {
        throw failure;
      },
    }),
    /onnxruntime-node/,
  );
});

test("a factory module must export createComponents", async () => {
  await assert.rejects(
    loadComponents(settings, { importModule: async () => ({}) }),
    /must export createComponents/,
  );
});
