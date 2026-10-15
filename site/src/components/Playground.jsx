import { useState } from "react";
import "./Playground.css";

const DEFAULT_ADDRESS = "http://127.0.0.1:8090";
const ADDRESS_KEY = "gliner-runner-playground-address";

function savedAddress() {
  try {
    return localStorage.getItem(ADDRESS_KEY) || DEFAULT_ADDRESS;
  } catch {
    return DEFAULT_ADDRESS;
  }
}

function Command({ children }) {
  return <pre className="command"><code>{children}</code></pre>;
}

// Classifies text with the runner on the visitor's own computer. The page is served from GitHub Pages and calls
// http://127.0.0.1:8090 directly, so the runner must allow this page's origin (gliner-runner cors add <origin>).
export default function Playground() {
  const [address, setAddress] = useState(savedAddress);
  const [status, setStatus] = useState("idle"); // idle | checking | ready | blocked | unreachable
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [text, setText] = useState("I was charged twice for my last order.");
  const [labels, setLabels] = useState("billing, support, shipping");
  const [several, setSeveral] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const base = address.trim().replace(/\/+$/, "");
  const origin = window.location.origin;

  async function connect() {
    setStatus("checking");
    setResult(null);
    setError("");
    try {
      localStorage.setItem(ADDRESS_KEY, base);
    } catch {
      // Remembering the address is only a convenience.
    }
    try {
      const reply = await fetch(`${base}/v1/models`);
      const data = await reply.json();
      const installed = data.models.filter((m) => m.available_locally).map((m) => m.model_id);
      setModels(installed);
      setModel(installed.includes("decide-340m") ? "decide-340m" : (installed[0] ?? ""));
      setStatus("ready");
    } catch {
      // A no-cors request succeeds whenever something answers, even if CORS would hide the reply from this page.
      const running = await fetch(`${base}/healthz`, { mode: "no-cors" }).then(() => true, () => false);
      setStatus(running ? "blocked" : "unreachable");
    }
  }

  async function classify() {
    setError("");
    setResult(null);
    setBusy(true);
    const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
    const task = { labels: list, ...(several ? { min_labels: 0, max_labels: null } : {}) };
    const reply = await fetch(`${base}/v1/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model, backend: "onnx", precision: "fp32", operation: "classify",
        text, schema: { kind: "classification", tasks: { category: task } },
      }),
    }).catch(() => null);
    setBusy(false);
    if (!reply) return setError("Lost the connection to the runner. Is it still running?");
    const data = await reply.json();
    if (!reply.ok) return setError(data.detail?.message ?? "The runner rejected the request.");
    setResult(data.output.category);
  }

  const picked = result && [result.value].flat().filter(Boolean);
  return (
    <div className="not-content playground">
      <div className="row">
        <label htmlFor="pg-address">Runner address</label>
        <input id="pg-address" value={address} onChange={(e) => setAddress(e.target.value)} spellCheck={false} />
        <button onClick={connect} disabled={status === "checking"}>
          {status === "checking" ? "Connecting…" : status === "ready" ? "Reconnect" : "Connect"}
        </button>
      </div>

      {status === "blocked" && (
        <div className="notice">
          <p>The runner is running, but it does not allow this page yet. Run this once, then restart the runner:</p>
          <Command>gliner-runner cors add {origin}</Command>
        </div>
      )}
      {status === "unreachable" && (
        <div className="notice">
          <p>No runner answered at {base}. Start it in a terminal:</p>
          <Command>gliner-runner serve</Command>
          <p>
            If it is already running, check that it allows this page (<code>gliner-runner cors list</code>), and
            allow this site to reach your local network if the browser asks.
          </p>
        </div>
      )}
      {status === "ready" && models.length === 0 && (
        <div className="notice">
          <p>Connected, but no model is installed yet. Install one, then click Reconnect:</p>
          <Command>gliner-runner models install decide-340m</Command>
        </div>
      )}

      {status === "ready" && models.length > 0 && (
        <div className="form">
          <label htmlFor="pg-model">Model</label>
          <select id="pg-model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
          <label htmlFor="pg-text">Text</label>
          <textarea id="pg-text" value={text} onChange={(e) => setText(e.target.value)} rows={4} />
          <label htmlFor="pg-labels">Labels, separated by commas</label>
          <input id="pg-labels" value={labels} onChange={(e) => setLabels(e.target.value)} />
          <label className="check">
            <input type="checkbox" checked={several} onChange={(e) => setSeveral(e.target.checked)} /> Allow several labels
          </label>
          <button onClick={classify} disabled={busy}>{busy ? "Classifying…" : "Classify"}</button>
          {error && <p className="error">{error}</p>}
          {result && (
            <div className="result">
              <p className="answer">{picked.length ? picked.join(", ") : "No label fits."}</p>
              <ul>
                {Object.entries(result.probabilities).map(([name, p]) => (
                  <li key={name}>
                    <span>{name}</span>
                    <meter min={0} max={1} value={p} />
                    <span>{p.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
