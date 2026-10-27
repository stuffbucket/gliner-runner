// "Be the model": one slider per label, each set to the model's score for one example text. Moving a slider asks
// "which text did the model score like this?", and the page shows that text and its scores. These functions are
// the whole of that choice. They are pure, so the tests (site/scripts/be-the-model.test.mjs), the page
// (BeTheModel.jsx) and the Markdown copy of the docs (site/scripts/markdown-copies.mjs) agree.
//
// The data is src/data/be-the-model.json: real lines with the model's real scores (see
// site/scripts/be-the-model-data.mjs).

/**
 * Two examples whose scores for the dragged label are this close count as a tie. The tie goes to the example
 * whose other scores are nearest the ones on screen, so the text changes as little as it can while the slider
 * moves, and a small nudge keeps the text that is already there.
 */
export const TIE = 0.02;

/** The sum of squared differences between two sets of scores, over every label but `skip`. */
function otherDistance(a, b, skip) {
  let total = 0;
  for (const label of Object.keys(a)) {
    if (label === skip) continue;
    const d = (a[label] ?? 0) - (b[label] ?? 0);
    total += d * d;
  }
  return total;
}

/** The examples from the lowest score for `label` to the highest (equal scores in id order). */
export function alongLabel(examples, label) {
  return [...examples].sort((a, b) => a.scores[label] - b.scores[label] || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * The example to show when the slider for `label` is at `value`: the one whose score for that label is nearest
 * `value`. Examples within `tie` of the nearest are a tie, broken by the smallest distance from `current` (the
 * scores on screen now) on the other labels, then by id, so the answer never depends on the order of the list.
 *
 * `order` is `alongLabel(examples, label)`, worked out once per label, so each move of a slider is a binary
 * search and a look at the few examples near it.
 *
 * @param {Array<{id: string, scores: Record<string, number>}>} order
 * @param {string} label
 * @param {number} value
 * @param {Record<string, number>} current
 * @param {number} [tie]
 */
export function nearestAlong(order, label, value, current, tie = TIE) {
  if (order.length === 0) throw new Error("nearestAlong: no examples");
  // The first example scoring `value` or more.
  let lo = 0;
  let hi = order.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (order[mid].scores[label] < value) lo = mid + 1;
    else hi = mid;
  }
  const gap = (i) => Math.abs(order[i].scores[label] - value);
  const best = Math.min(lo > 0 ? gap(lo - 1) : Infinity, lo < order.length ? gap(lo) : Infinity);
  // A tiny allowance so floating-point noise never splits a real tie.
  const limit = best + tie + 1e-12;
  let winner;
  let winnerDistance = Infinity;
  const consider = (example) => {
    const distance = otherDistance(example.scores, current, label);
    if (distance < winnerDistance || (distance === winnerDistance && example.id < winner.id)) {
      winner = example;
      winnerDistance = distance;
    }
  };
  for (let i = lo - 1; i >= 0 && gap(i) <= limit; i--) consider(order[i]);
  for (let i = lo; i < order.length && gap(i) <= limit; i++) consider(order[i]);
  return winner;
}

/** `nearestAlong` for a list in any order. */
export function nearestExample(examples, label, value, current, tie = TIE) {
  return nearestAlong(alongLabel(examples, label), label, value, current, tie);
}

/**
 * The example `steps` places away from `currentId` in `order` (one label's examples, lowest score first): +1 is
 * the next higher score, -1 the next lower. It stops at the ends. `Infinity` and `-Infinity` go to the highest
 * and the lowest.
 */
export function stepAlong(order, currentId, steps) {
  const at = order.findIndex((example) => example.id === currentId);
  if (at === -1) throw new Error(`stepAlong: no example ${currentId}`);
  return order[Math.min(order.length - 1, Math.max(0, at + steps))];
}

/** `stepAlong` for a list in any order. */
export function stepExample(examples, label, currentId, steps) {
  return stepAlong(alongLabel(examples, label), currentId, steps);
}

/** How many places a key moves along a slider, or undefined for keys the slider leaves alone. */
export function keySteps(key) {
  return {
    ArrowRight: 1,
    ArrowUp: 1,
    ArrowLeft: -1,
    ArrowDown: -1,
    PageUp: 10,
    PageDown: -10,
    Home: -Infinity,
    End: Infinity,
  }[key];
}

/** The label with the highest score: the model's answer when it picks one label. */
export function topLabel(scores) {
  return Object.entries(scores).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
}

/** How many different scores (to two decimals, as the page shows them) each label's slider can stop at. */
export function stopsPerLabel(data) {
  return Object.fromEntries(
    data.labels.map(({ name }) => [name, new Set(data.examples.map((example) => example.scores[name].toFixed(2))).size]),
  );
}

/** Throws if the data is not the shape the page needs. Returns the data. */
export function checkData(data, name = "be-the-model data") {
  const fail = (message) => {
    throw new Error(`${name}: ${message}`);
  };
  if (!Array.isArray(data?.labels) || data.labels.length < 2) fail("needs at least two labels");
  if (!Array.isArray(data.examples) || data.examples.length === 0) fail("needs examples");
  const names = data.labels.map((label) => label.name);
  if (new Set(names).size !== names.length) fail("label names repeat");
  const ids = new Set();
  for (const example of data.examples) {
    if (ids.has(example.id)) fail(`example id ${example.id} repeats`);
    ids.add(example.id);
    if (typeof example.text !== "string" || !example.text) fail(`${example.id} has no text`);
    for (const label of names) {
      const score = example.scores?.[label];
      if (typeof score !== "number" || score < 0 || score > 1) fail(`${example.id} has no score from 0 to 1 for "${label}"`);
    }
    if (Object.keys(example.scores).length !== names.length) fail(`${example.id} scores labels that are not in the list`);
    if (data.mode === "single") {
      const sum = names.reduce((total, label) => total + example.scores[label], 0);
      // The scores are rounded, so they can miss 1 by a little.
      if (Math.abs(sum - 1) > 0.01) fail(`${example.id}'s scores add up to ${sum.toFixed(3)}, not 1`);
    }
  }
  return data;
}

/** The real data when there is any, otherwise the made-up fixture. `mock` tells the page to say so. */
export function chooseData(real, mock) {
  if (real) return { ...checkData(real, "src/data/be-the-model.json"), mock: false };
  if (mock) return { ...checkData(mock, "scripts/fixtures/be-the-model.mock.json"), mock: true };
  throw new Error("no be-the-model data: add src/data/be-the-model.json");
}

/** The model's request for one line: the data's labels and descriptions, one label (as the data was made). */
export function inferRequest(data, text) {
  return {
    model: data.model,
    backend: "onnx",
    precision: "fp32",
    operation: "classify",
    text,
    schema: {
      kind: "classification",
      tasks: {
        category: {
          labels: data.labels.map((label) => label.name),
          label_descriptions: Object.fromEntries(data.labels.map((label) => [label.name, label.description])),
        },
      },
    },
  };
}

/** The scores from a reply to `inferRequest`, or an error message. */
export function scoresFromReply(data, status, body) {
  let reply;
  try {
    reply = JSON.parse(body);
  } catch {
    return { error: "The model's reply could not be read." };
  }
  if (status !== 200) {
    const detail = typeof reply.detail === "string" ? reply.detail : reply.detail?.message;
    return { error: detail ?? "The model turned the request down." };
  }
  const probabilities = reply.output?.category?.probabilities ?? {};
  const scores = {};
  for (const { name } of data.labels) {
    if (typeof probabilities[name] !== "number") return { error: `The model gave no score for "${name}".` };
    scores[name] = probabilities[name];
  }
  return { scores };
}
