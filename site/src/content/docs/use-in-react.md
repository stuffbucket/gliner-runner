---
title: Use it in a React app
description: A small React page that sends text and labels to the local runner and shows the result.
sidebar:
  order: 3
---

This guide builds a web page with a text box, a place to type labels, and a Classify button. The
page calls the runner on your own computer. Nothing goes to the internet.

You need the runner and the `decide-340m` model (see the [home page](/gliner-runner/)), and
Node.js version 20.19 or newer, or 22.12 or newer. Check with `node --version`.

## 1. Start the runner

In one Terminal window:

```sh
gliner-runner serve
```

```text
[gliner-runner-onnx] Application startup complete.
[gliner-runner-onnx] Listening on http://127.0.0.1:8090
```

Leave this window open. The runner is now waiting for requests on port 8090.

## 2. Get the example app

In a second Terminal window:

```sh
git clone https://github.com/stuffbucket/gliner-runner.git
cd gliner-runner/examples/react-classifier
npm install
```

The app lives in the `examples/react-classifier` folder. It has its own `package.json` and is
separate from the rest of the project. Its dependencies are fixed to exact versions (React 19.3.0,
Vite 8.3.1).

## 3. Run it

```sh
npm run dev
```

Open <http://localhost:5173>. The text box already has a sample message and the labels
`billing, support, shipping`. Click Classify. You should see `billing` and a score for each
label.

Tick "Allow several labels" to let a text match more than one label, or none.

## How it works

The component is one file, `src/App.jsx`:

```jsx
import { useState } from "react";

// Empty means "this page's own address", which Vite forwards to the runner (see vite.config.js).
const RUNNER = import.meta.env.VITE_RUNNER_URL ?? "";

export default function App() {
  const [text, setText] = useState("I was charged twice for my last order.");
  const [labels, setLabels] = useState("billing, support, shipping");
  const [several, setSeveral] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function classify() {
    setError("");
    setResult(null);
    const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
    const task = { labels: list, ...(several ? { min_labels: 0, max_labels: null } : {}) };
    const reply = await fetch(`${RUNNER}/v1/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "decide-340m", backend: "onnx", precision: "fp32", operation: "classify",
        text, schema: { kind: "classification", tasks: { category: task } },
      }),
    }).catch(() => null);
    if (!reply) return setError("Could not reach the runner. Is `gliner-runner serve` running, and does `gliner-runner cors list` include this page's address?");
    const data = await reply.json();
    if (!reply.ok) return setError(data.detail?.message ?? "The runner rejected the request.");
    setResult(data.output.category);
  }

  const picked = result && [result.value].flat().filter(Boolean);
  return (
    <main>
      <h1>Classify text</h1>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />
      <input value={labels} onChange={(e) => setLabels(e.target.value)} aria-label="Labels, separated by commas" />
      <label><input type="checkbox" checked={several} onChange={(e) => setSeveral(e.target.checked)} /> Allow several labels</label>
      <button onClick={classify}>Classify</button>
      {error && <p className="error">{error}</p>}
      {result && (
        <div>
          <p className="answer">{picked.length ? picked.join(", ") : "No label fits."}</p>
          <ul>{Object.entries(result.probabilities).map(([name, p]) => <li key={name}>{name}: {p.toFixed(3)}</li>)}</ul>
        </div>
      )}
    </main>
  );
}
```

What it does:

- It keeps the text, the labels, and the result in `useState`.
- When you click Classify, it splits the labels at the commas and sends one request to
  `/v1/infer`.
- The request names the model (`decide-340m`) and lists your labels under `schema`. With "Allow
  several labels" ticked, it adds `min_labels: 0` and `max_labels: null`.
- The answer is in `data.output.category`: `value` is the chosen label (or a list of labels),
  and `probabilities` has a score for every label.
- If the runner is not running, the page says so.

### Why there is no CORS setup

A web page is only allowed to call the address it came from. This page comes from
`localhost:5173` and the runner is on `127.0.0.1:8090`, which is a different address. To avoid the
problem, the page calls its own address, `/v1/infer`, and Vite passes anything starting with
`/v1` on to the runner. That is the `server.proxy` line in `vite.config.js`:

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The page calls /v1/infer on its own address. Vite forwards /v1 to the local
// runner, so the browser never makes a cross-origin request and no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/v1": "http://127.0.0.1:8090" },
  },
});
```

To check the proxy without the page, send a request to the Vite address while `npm run dev` is
running. It reaches the runner and returns the answer:

```sh
curl -s http://localhost:5173/v1/infer -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx", "precision": "fp32", "operation": "classify",
  "text": "I was charged twice for my last order.",
  "schema": {"kind": "classification", "tasks": {"category": {"labels": ["billing", "support", "shipping"]}}}
}'
```

```text
{"request_id":"ce093550-04c5-40f7-8b06-bff8c4bcfc0b","model":"decide-340m","backend":"onnx","precision":"fp32","output":{"category":{"value":"billing","confidence":0.8887295281128845,"probabilities":{"billing":0.8887295281128845,"support":0.04850131418933352,"shipping":0.06276915769778203}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":1.004750370979309,"violations":[]}},"error":null,"timing":{"queue_ms":4.562040999997407,"inference_ms":21.926749999984168},"usage":{"inputTokens":22,"outputTokens":0}}
```

### If the page is served some other way

The proxy exists while `npm run dev` runs, and also for `npm run preview` (which serves the built
page at <http://localhost:4173>). Any other web server, such as GitHub Pages, nginx, or
`python3 -m http.server`, has no proxy. There the page has to call the runner directly, and the
runner has to allow the page's address.

1. Build the page with the runner's address. `VITE_RUNNER_URL` replaces the empty `RUNNER` at the
   top of `App.jsx`:

   ```sh
   VITE_RUNNER_URL=http://127.0.0.1:8090 npm run build
   ```

2. Allow the address the page is served from. Use the exact address, with the port and no
   trailing slash:

   ```sh
   gliner-runner cors add https://you.github.io
   ```

   The runner saves the list and reads it each time `gliner-runner serve` starts, so restart it
   after a change. Common local development addresses such as `http://localhost:5173` and
   `http://localhost:4173` are allowed from the start. `gliner-runner cors list` shows the list
   with a number for each address, and `gliner-runner cors delete <number>` removes one.
   Addresses that are not on the list stay blocked.

3. If the page lives in a folder of a site, such as `https://you.github.io/classifier/`, set
   `base: "/classifier/"` in `vite.config.js` so the page finds its script and styles.

Chrome asks once whether a public site may reach devices on your local network. Allow it, because
the runner on `127.0.0.1` counts as one. The [playground](/gliner-runner/playground/) on this site
works this way.

## Make it yours

- Change the default labels and text in the first lines of `App.jsx`.
- Add `label_descriptions` to the task, for example
  `label_descriptions: { billing: "Charges, invoices, or refunds" }`. Descriptions can improve the
  answers (see [how it works](/gliner-runner/how-it-works/#descriptions-help)).
- Change `model` to `decide-multilingual` to classify other languages.
- Build a static copy with `npm run build`. The result is in `dist/`. To put it on a web server,
  see [If the page is served some other way](#if-the-page-is-served-some-other-way).
