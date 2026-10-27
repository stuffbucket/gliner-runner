// The data the Be the model page shows (src/components/BeTheModel.jsx): src/data/be-the-model.json, real lines
// with the model's real scores. Until that file exists, the page falls back to the made-up test fixture
// (scripts/fixtures/be-the-model.mock.json) and says the scores are made up.
//
// The page imports "virtual:be-the-model", which this Vite plugin resolves at build time, so only one of the two
// files goes into the page's script. The per-label "multi" scores are left out: the page does not use them.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { chooseData } from "../src/components/beTheModel.js";

export const REAL = "src/data/be-the-model.json";
export const MOCK = "scripts/fixtures/be-the-model.mock.json";
const ID = "virtual:be-the-model";

const read = (file) => (existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : undefined);

/** The checked data: the real file when there is one, otherwise the fixture with `mock: true`. */
export function beTheModelData(siteDir) {
  return chooseData(read(join(siteDir, REAL)), read(join(siteDir, MOCK)));
}

/** What the page needs of it. */
export function pageData(data) {
  return {
    model: data.model,
    mode: data.mode,
    mock: data.mock,
    labels: data.labels.map(({ name, description }) => ({ name, description })),
    examples: data.examples.map(({ id, text, ask, age, scores }) => ({ id, text, ask, age, scores })),
  };
}

export function beTheModelPlugin(siteDir) {
  return {
    name: "gliner-runner:be-the-model",
    resolveId(id) {
      return id === ID ? `\0${ID}` : undefined;
    },
    load(id) {
      if (id !== `\0${ID}`) return undefined;
      // Rebuild in dev when either file changes or the real one appears.
      this.addWatchFile(join(siteDir, REAL));
      this.addWatchFile(join(siteDir, MOCK));
      return `export default ${JSON.stringify(pageData(beTheModelData(siteDir)))};`;
    },
  };
}
