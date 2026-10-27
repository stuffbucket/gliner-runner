import { useState } from "react";

// Empty means "this page's own address", which Vite forwards to the
// runner (see vite.config.js).
const RUNNER = import.meta.env.VITE_RUNNER_URL ?? "";

const NOT_REACHED =
  "Could not reach the runner. Is `gliner-runner serve` running, and " +
  "does `gliner-runner cors list` include this page's address?";

export default function App() {
  const [text, setText] = useState(
    "I was charged twice for my last order.",
  );
  const [labels, setLabels] = useState("billing, support, shipping");
  const [several, setSeveral] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function classify() {
    setError("");
    setResult(null);
    const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
    const task = several
      ? { labels: list, min_labels: 0, max_labels: null }
      : { labels: list };
    const reply = await fetch(`${RUNNER}/v1/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "decide-340m", backend: "onnx", precision: "fp32",
        operation: "classify", text,
        schema: { kind: "classification", tasks: { category: task } },
      }),
    }).catch(() => null);
    if (!reply) return setError(NOT_REACHED);
    const data = await reply.json();
    if (!reply.ok) {
      const message = data.detail?.message;
      return setError(message ?? "The runner rejected the request.");
    }
    setResult(data.output.category);
  }

  const picked = result && [result.value].flat().filter(Boolean);
  return (
    <main>
      <h1>Classify text</h1>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
      />
      <input
        value={labels}
        onChange={(e) => setLabels(e.target.value)}
        aria-label="Labels, separated by commas"
      />
      <label>
        <input
          type="checkbox"
          checked={several}
          onChange={(e) => setSeveral(e.target.checked)}
        />{" "}
        Allow several labels
      </label>
      <button onClick={classify}>Classify</button>
      {error && <p className="error">{error}</p>}
      {result && (
        <div>
          <p className="answer">
            {picked.length ? picked.join(", ") : "No label fits."}
          </p>
          <ul>
            {Object.entries(result.probabilities).map(([name, p]) => (
              <li key={name}>
                {name}: {p.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
