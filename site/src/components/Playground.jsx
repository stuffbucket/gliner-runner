import { useEffect, useRef, useState } from "react";
import { HttpTransport, INSTALL_PEAK_BYTES, WorkerTransport } from "@gliner-runner/onnx-runner-web/client";
import "./Playground.css";

const DEFAULT_ADDRESS = "http://127.0.0.1:8090";
const IN_TAB_MODEL = "decide-340m";
const KEYS = { mode: "gliner-runner-playground-mode", address: "gliner-runner-playground-address" };
const SETUP_COMMAND =
  "curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh -s -- --playground --yes --no-open";
const BREW_COMMAND = "brew install stuffbucket/tap/gliner-runner && gliner-runner playground --yes --no-open";

const stored = (key, fallback) => {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
};
const store = (key, value) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Remembering choices is only a convenience.
  }
};
const gb = (bytes) => `${(bytes / 1e9).toFixed(2)} GB`;
const parse = (reply) => {
  try {
    return JSON.parse(reply.body);
  } catch {
    return {};
  }
};
const message = (data, fallback) =>
  typeof data.detail === "string" ? data.detail : (data.detail?.message ?? fallback);

function Command({ children }) {
  const [copied, setCopied] = useState(false);
  const copy = () =>
    navigator.clipboard?.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  return (
    <div className="command">
      <pre>
        <code>{children}</code>
      </pre>
      <button type="button" className="secondary" onClick={copy}>
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

/** Text, labels and the answer; the same for both runners. */
function Classifier({ transport, models }) {
  const [model, setModel] = useState(models.includes(IN_TAB_MODEL) ? IN_TAB_MODEL : models[0]);
  const [text, setText] = useState("I was charged twice for my last order.");
  const [labels, setLabels] = useState("billing, support, shipping");
  const [several, setSeveral] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const loaded = useRef(false);

  async function classify() {
    setError("");
    setResult(null);
    setBusy(true);
    const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
    const task = { labels: list, ...(several ? { min_labels: 0, max_labels: null } : {}) };
    try {
      const reply = await transport.request("POST", "/v1/infer", {
        model, backend: "onnx", precision: "fp32", operation: "classify",
        text, schema: { kind: "classification", tasks: { category: task } },
      });
      const data = parse(reply);
      if (reply.status !== 200) {
        setError(message(data, "The runner rejected the request."));
      } else {
        loaded.current = true;
        setResult(data.output.category);
      }
    } catch {
      setError("Lost the connection to the runner.");
    } finally {
      setBusy(false);
    }
  }

  const picked = result && [result.value].flat().filter(Boolean);
  const working = transport.kind === "tab" && !loaded.current ? "Loading the model…" : "Classifying…";
  return (
    <div className="form">
      {models.length > 1 && (
        <>
          <label htmlFor="pg-model">Model</label>
          <select id="pg-model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </>
      )}
      <label htmlFor="pg-text">Text</label>
      <textarea id="pg-text" value={text} onChange={(e) => setText(e.target.value)} rows={4} />
      <label htmlFor="pg-labels">Labels, separated by commas</label>
      <input id="pg-labels" value={labels} onChange={(e) => setLabels(e.target.value)} />
      <label className="check">
        <input type="checkbox" checked={several} onChange={(e) => setSeveral(e.target.checked)} /> Allow several labels
      </label>
      <button type="button" onClick={classify} disabled={busy}>{busy ? working : "Classify"}</button>
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
  );
}

/** Downloads a model through the runner's own download API and waits for the job. */
async function install(transport, entry, setJob) {
  const created = await transport.request("POST", `/v1/models/${entry.model_id}/downloads`, {
    approved: true,
    destination: entry.download_destination,
  });
  let job = parse(created);
  if (created.status !== 202) {
    throw new Error(message(job, "The runner refused the download."));
  }
  setJob(job);
  while (job.state === "queued" || job.state === "running") {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    job = parse(await transport.request("GET", `/v1/model-downloads/${job.job_id}`));
    setJob(job);
  }
  if (job.state !== "succeeded") {
    throw new Error(job.error ?? "The download failed.");
  }
}

const STAGES = {
  download: "Downloading from Hugging Face",
  "verify-source": "Checking the download",
  graph: "Preparing the model",
  weights: "Preparing the model",
  copy: "Preparing the model",
  "verify-output": "Checking the prepared model",
  finalize: "Finishing",
};

function InTab({ params }) {
  const [transport, setTransport] = useState(null);
  const [status, setStatus] = useState({ state: "starting" });
  const [entry, setEntry] = useState(null);
  const [progress, setProgress] = useState(null);
  const [provider, setProvider] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [generation, setGeneration] = useState(0);
  const cancelled = useRef(false);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/runner.worker.ts", import.meta.url), { type: "module" });
    const runner = new WorkerTransport(worker, {
      hfEndpoint: params.get("hf") ?? undefined,
      executionProvider: params.get("ep") ?? undefined,
    });
    const unsubscribe = runner.subscribe((event) => {
      if (event.type === "status") setStatus(event.status);
      if (event.type === "progress") setProgress(event.progress);
      if (event.type === "provider") setProvider(event);
    });
    setTransport(runner);
    return () => {
      unsubscribe();
      runner.terminate();
    };
  }, [generation]);

  const refresh = async () => {
    const data = parse(await transport.request("GET", "/v1/models"));
    setEntry(data.models?.find((model) => model.model_id === IN_TAB_MODEL) ?? null);
  };
  useEffect(() => {
    if (status.state === "ready" && transport) refresh();
  }, [status.state, transport]);

  async function download() {
    setError("");
    setProgress(null);
    cancelled.current = false;
    // Asks the browser not to evict the 1.75 GB model under storage pressure (Firefox shows a prompt).
    await navigator.storage.persist?.().catch(() => false);
    try {
      await install(transport, entry, setJob);
      await refresh();
    } catch (failure) {
      if (!cancelled.current) setError(failure.message);
    } finally {
      setJob(null);
    }
  }

  function cancel() {
    // Stopping the worker stops the download; the next start removes what it left behind.
    cancelled.current = true;
    setJob(null);
    setProgress(null);
    setEntry(null);
    setGeneration((value) => value + 1);
  }

  async function remove() {
    setEntry(null);
    await transport.deleteModel();
    setProvider(null);
    await refresh();
  }

  if (status.state === "starting" || !transport) {
    return <p className="muted">Starting the runner in this tab…</p>;
  }
  if (status.state === "unsupported" || status.state === "failed") {
    return (
      <div className="notice">
        <p>
          {status.state === "unsupported" ? status.reason : `The runner could not start: ${status.reason}`} Switch to{" "}
          <strong>Use my machine</strong>, or open this page in a normal (not private) window of a current Chrome,
          Edge, Firefox or Safari.
        </p>
      </div>
    );
  }
  if (entry === null) {
    return <p className="muted">Checking for the model…</p>;
  }
  if (!entry.available_locally) {
    const free = status.storage.quota - status.storage.usage;
    const enough = free >= INSTALL_PEAK_BYTES;
    const fraction = progress && progress.total > 0 ? progress.done / progress.total : 0;
    return (
      <div className="setup">
        <p>
          The model ({IN_TAB_MODEL}) is downloaded once from Hugging Face, about <strong>1.95 GB</strong>, and kept
          in this browser for this site (1.75 GB after preparation). Nothing you type is sent anywhere.
        </p>
        {!enough && (
          <p className="error">
            This browser lets the site store {gb(Math.max(free, 0))}; the download needs about {gb(INSTALL_PEAK_BYTES)}.
            Free some disk space, or use <strong>Use my machine</strong>.
          </p>
        )}
        {job === null ? (
          <button type="button" onClick={download} disabled={!enough}>Download {IN_TAB_MODEL}</button>
        ) : (
          <div className="progress">
            <span>{progress ? STAGES[progress.stage] ?? progress.stage : "Starting the download"}…</span>
            <progress max={1} value={fraction} />
            <span>{progress ? `${gb(progress.done)} of ${gb(progress.total)}` : ""}</span>
            <button type="button" className="secondary" onClick={cancel}>Cancel</button>
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </div>
    );
  }
  return (
    <>
      <Classifier transport={transport} models={[IN_TAB_MODEL]} />
      <p className="muted small">
        {provider
          ? `Running on ${provider.provider === "webgpu" ? "WebGPU (your graphics card)" : "WASM (your processor)"}${provider.note ? `: ${provider.note}` : ""}.`
          : "The model loads on the first Classify (a few seconds)."}{" "}
        <button type="button" className="link" onClick={remove}>Remove the downloaded model</button>
      </p>
    </>
  );
}

function OnMyMachine({ params }) {
  const [address, setAddress] = useState(() => params.get("runner") ?? stored(KEYS.address, DEFAULT_ADDRESS));
  const [state, setState] = useState({ kind: "probing" });
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const base = address.trim().replace(/\/+$/, "");
  const transport = new HttpTransport(base);

  async function probe() {
    try {
      const data = parse(await transport.request("GET", "/v1/models"));
      store(KEYS.address, base);
      setState({ kind: "connected", models: data.models ?? [] });
      return true;
    } catch {
      // A no-cors request succeeds whenever something answers, even if CORS hides the reply from this page.
      const running = await fetch(`${base}/healthz`, { mode: "no-cors" }).then(() => true, () => false);
      setState({ kind: running ? "blocked" : "absent" });
      return false;
    }
  }

  useEffect(() => {
    let stopped = false;
    let timer;
    const loop = async () => {
      if (stopped) return;
      if (!(await probe()) && !stopped) timer = setTimeout(loop, 3000);
    };
    loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [base]);

  async function download(entry) {
    setError("");
    try {
      await install(transport, entry, setJob);
      await probe();
    } catch (failure) {
      setError(failure.message);
    } finally {
      setJob(null);
    }
  }

  const origin = window.location.origin;
  const advanced = (
    <details className="advanced">
      <summary>Runner address</summary>
      <input aria-label="Runner address" value={address} onChange={(e) => setAddress(e.target.value)} spellCheck={false} />
    </details>
  );
  if (state.kind === "probing") {
    return <p className="muted">Looking for a runner on this computer…</p>;
  }
  if (state.kind !== "connected") {
    return (
      <div className="setup">
        {state.kind === "blocked" ? (
          <p>A runner is running at {base}, but it does not allow this page yet. Run this, which allows the page and restarts nothing else:</p>
        ) : (
          <p>No runner is running on this computer yet. Run this in a terminal; it installs gliner-runner if needed, gets the model, and starts it:</p>
        )}
        {state.kind === "blocked" ? <Command>gliner-runner playground --yes --no-open</Command> : <Command>{SETUP_COMMAND}</Command>}
        {state.kind !== "blocked" && (
          <p className="muted small">
            With Homebrew: <code>{BREW_COMMAND}</code>. This page connects by itself once the runner is up; your
            browser may ask to let {origin} reach your local network. Allow it.
          </p>
        )}
        {advanced}
      </div>
    );
  }
  const installed = state.models.filter((model) => model.available_locally).map((model) => model.model_id);
  const missing = state.models.find((model) => model.model_id === IN_TAB_MODEL && !model.available_locally);
  return (
    <>
      {installed.length > 0 ? (
        <Classifier transport={transport} models={installed} />
      ) : (
        <div className="setup">
          <p>Connected to {base}, but no model is installed there yet.</p>
          {missing && (job === null ? (
            <button type="button" onClick={() => download(missing)}>Download {IN_TAB_MODEL} on this computer (1.9 GB)</button>
          ) : (
            <p className="muted">Downloading and preparing {IN_TAB_MODEL} on your computer… this takes a few minutes.</p>
          ))}
          {error && <p className="error">{error}</p>}
        </div>
      )}
      <p className="muted small">Connected to the runner at {base}.</p>
      {advanced}
    </>
  );
}

// The playground: gliner-runner itself running in this tab (a Web Worker with WebGPU or WASM), or the runner on
// this computer over HTTP. Both are driven through the runner's /v1 API.
export default function Playground() {
  const params = new URLSearchParams(window.location.search);
  const [mode, setMode] = useState(() => params.get("mode") ?? stored(KEYS.mode, "tab"));
  const choose = (next) => {
    store(KEYS.mode, next);
    setMode(next);
  };
  return (
    <div className="not-content playground">
      <div className="modes" role="tablist" aria-label="Where the runner runs">
        <button type="button" role="tab" aria-selected={mode === "tab"} onClick={() => choose("tab")}>In this tab</button>
        <button type="button" role="tab" aria-selected={mode === "local"} onClick={() => choose("local")}>Use my machine</button>
      </div>
      {mode === "local" ? <OnMyMachine params={params} /> : <InTab params={params} />}
    </div>
  );
}
