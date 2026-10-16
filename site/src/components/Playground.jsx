import { useEffect, useRef, useState } from "react";
import { HttpTransport, INSTALL_PEAK_BYTES, WorkerTransport } from "@gliner-runner/onnx-runner-web/client";
import "./Playground.css";

const BASE = "/gliner-runner";
const DEFAULT_ADDRESS = "http://127.0.0.1:8090";
const IN_TAB_MODEL = "decide-340m";
const KEYS = {
  mode: "gliner-runner-playground-mode",
  address: "gliner-runner-playground-address",
  about: "gliner-runner-playground-about",
};
const SETUP_COMMAND =
  "curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh -s -- --playground --yes --no-open";

const EXAMPLES = [
  { name: "Support ticket", text: "I was charged twice for my last order.", labels: "billing, support, shipping" },
  { name: "Review", text: "The battery lasts all day, but the screen scratches far too easily.", labels: "positive, negative, mixed" },
  { name: "Topics", text: "I will pay my rent on Friday, and then I will bake bread.", labels: "weather, cooking, sports, money", several: true },
  { name: "Urgency", text: "The production database is down and customers cannot log in.", labels: "urgent, can wait" },
];

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

function Command({ children }) {
  const [copied, setCopied] = useState(false);
  const copy = () =>
    navigator.clipboard?.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  return (
    <div className="pg-command">
      <code>{children}</code>
      <button type="button" className="pg-quiet" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

/** A centered card for everything that comes before classifying. */
function Card({ title, children }) {
  return (
    <div className="pg-card">
      {title && <h2>{title}</h2>}
      {children}
    </div>
  );
}

/** Text, labels and the answer; the same for both runners. */
function Workspace({ transport, models, firstRunNote }) {
  const [model, setModel] = useState(models.includes(IN_TAB_MODEL) ? IN_TAB_MODEL : models[0]);
  const [text, setText] = useState(EXAMPLES[0].text);
  const [labels, setLabels] = useState(EXAMPLES[0].labels);
  const [several, setSeveral] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [warm, setWarm] = useState(false);

  const pick = (example) => {
    setText(example.text);
    setLabels(example.labels);
    setSeveral(example.several ?? false);
    setResult(null);
  };

  async function classify() {
    setError("");
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
        setWarm(true);
        setResult(data.output.category);
      }
    } catch {
      setError("Lost the connection to the runner.");
    } finally {
      setBusy(false);
    }
  }

  const picked = result ? [result.value].flat().filter(Boolean) : [];
  const ranked = result ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]) : [];
  const loading = busy && !warm && firstRunNote !== undefined;
  return (
    <div className="pg-workspace">
      <section className="pg-input">
        <div className="pg-examples" aria-label="Examples">
          {EXAMPLES.map((example) => (
            <button type="button" key={example.name} className="pg-chip" onClick={() => pick(example)}>
              {example.name}
            </button>
          ))}
        </div>
        <textarea
          aria-label="Text to classify"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or type some text"
        />
        <label className="pg-field">
          <span>Labels</span>
          <input value={labels} onChange={(e) => setLabels(e.target.value)} placeholder="billing, support, shipping" />
        </label>
        <div className="pg-actions">
          <label className="pg-check">
            <input type="checkbox" checked={several} onChange={(e) => setSeveral(e.target.checked)} /> Several labels can fit
          </label>
          {models.length > 1 && (
            <select aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          )}
          <button type="button" className="pg-primary" onClick={classify} disabled={busy || !text.trim()}>
            {loading ? "Loading the model…" : busy ? "Classifying…" : "Classify"}
          </button>
        </div>
      </section>
      <section className="pg-output" aria-live="polite">
        {error && <p className="pg-error">{error}</p>}
        {!result && !error && <p className="pg-hint">{loading ? firstRunNote : "The answer appears here."}</p>}
        {result && (
          <>
            <p className="pg-answer">{picked.length ? picked.join(", ") : "No label fits"}</p>
            <ul className="pg-scores">
              {ranked.map(([name, p]) => (
                <li key={name} className={picked.includes(name) ? "picked" : ""}>
                  <span>{name}</span>
                  <span className="pg-meter"><span style={{ width: `${Math.round(p * 100)}%` }} /></span>
                  <span>{p.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}

const STAGES = {
  download: "Downloading",
  "verify-source": "Checking",
  graph: "Preparing",
  weights: "Preparing",
  copy: "Preparing",
  "verify-output": "Checking",
  finalize: "Finishing",
};

function InTab({ params, setStatus, useMachine }) {
  const [transport, setTransport] = useState(null);
  const [host, setHost] = useState({ state: "starting" });
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
      if (event.type === "status") setHost(event.status);
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
    if (host.state === "ready" && transport) refresh();
  }, [host.state, transport]);

  const blocked = host.state === "unsupported" || host.state === "failed";
  useEffect(() => {
    const engine = provider ? (provider.provider === "webgpu" ? "graphics card" : "processor") : null;
    setStatus(
      blocked
        ? { tone: "warn", text: "Not available in this window" }
        : job
          ? { tone: "busy", text: "Getting the model" }
          : entry?.available_locally
            ? { tone: "ok", text: engine ? `Ready · ${engine}` : "Ready" }
            : entry === null
              ? { tone: "idle", text: "Starting" }
              : { tone: "idle", text: "Model not downloaded" },
    );
  }, [blocked, entry, job, provider]);

  async function download() {
    setError("");
    setProgress(null);
    cancelled.current = false;
    // Asks the browser not to evict the model when disk space runs low (Firefox shows a prompt).
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
    setProvider(null);
    await transport.deleteModel();
    await refresh();
  }

  const elsewhere = <button type="button" className="pg-link" onClick={useMachine}>Run it on my computer instead</button>;
  if (blocked) {
    return (
      <Card title="This window can't keep the model">
        <p>{host.reason} Open the page in a normal (not private) window, or run the model on your computer.</p>
        <button type="button" className="pg-primary" onClick={useMachine}>Run it on my computer</button>
      </Card>
    );
  }
  if (host.state === "starting" || !transport || entry === null) {
    return <Card><p className="pg-hint">Starting…</p></Card>;
  }
  if (!entry.available_locally) {
    const free = host.storage.quota - host.storage.usage;
    const enough = free >= INSTALL_PEAK_BYTES;
    const fraction = progress && progress.total > 0 ? progress.done / progress.total : 0;
    return (
      <Card title={job ? "Getting the model ready" : "Download the model to start"}>
        {job === null ? (
          <>
            <p>One time, 1.95 GB. It stays in this browser, and your text never leaves it.</p>
            {!enough && <p className="pg-error">Not enough space: this browser allows the site {gb(Math.max(free, 0))}.</p>}
            <button type="button" className="pg-primary" onClick={download} disabled={!enough}>Download</button>
            {error && <p className="pg-error">{error}</p>}
            <p className="pg-small">{elsewhere}</p>
          </>
        ) : (
          <>
            <progress max={1} value={fraction} />
            <p className="pg-small">
              {progress ? `${STAGES[progress.stage] ?? "Working"} · ${gb(progress.done)} of ${gb(progress.total)}` : "Starting…"}
            </p>
            <button type="button" className="pg-quiet" onClick={cancel}>Cancel</button>
          </>
        )}
      </Card>
    );
  }
  return (
    <>
      <Workspace transport={transport} models={[IN_TAB_MODEL]} firstRunNote="Loading the model takes a few seconds the first time." />
      <p className="pg-footer">
        Runs in this browser{provider ? ` on your ${provider.provider === "webgpu" ? "graphics card" : "processor"}` : ""}.{" "}
        <button type="button" className="pg-link" onClick={remove}>Remove the model</button> · {elsewhere}
      </p>
    </>
  );
}

function OnMyMachine({ params, setStatus, useBrowser }) {
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

  useEffect(() => {
    setStatus(
      state.kind === "connected"
        ? { tone: "ok", text: "Connected to your computer" }
        : { tone: "busy", text: "Waiting for your computer" },
    );
  }, [state.kind]);

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

  const back = <button type="button" className="pg-link" onClick={useBrowser}>Back to this browser</button>;
  if (state.kind !== "connected") {
    return (
      <Card title="Run it on your computer">
        <p>Paste this into a terminal. It sets up the runner and connects this page.</p>
        <Command>{state.kind === "blocked" ? "gliner-runner playground --yes --no-open" : SETUP_COMMAND}</Command>
        <p className="pg-hint pg-waiting">{state.kind === "probing" ? "Looking for the runner…" : "Waiting for the runner…"}</p>
        <details className="pg-advanced">
          <summary>Runner address</summary>
          <input aria-label="Runner address" value={address} onChange={(e) => setAddress(e.target.value)} spellCheck={false} />
        </details>
        <p className="pg-small">{back}</p>
      </Card>
    );
  }
  const installed = state.models.filter((model) => model.available_locally).map((model) => model.model_id);
  const missing = state.models.find((model) => model.model_id === IN_TAB_MODEL && !model.available_locally);
  if (installed.length === 0) {
    return (
      <Card title="Connected, but no model yet">
        {job === null ? (
          <button type="button" className="pg-primary" onClick={() => download(missing)} disabled={!missing}>
            Download {IN_TAB_MODEL} to your computer
          </button>
        ) : (
          <p className="pg-hint">Downloading and preparing on your computer… a few minutes.</p>
        )}
        {error && <p className="pg-error">{error}</p>}
        <p className="pg-small">{back}</p>
      </Card>
    );
  }
  return (
    <>
      <Workspace transport={transport} models={installed} />
      <p className="pg-footer">Runs on your computer ({base}). {back}</p>
    </>
  );
}

function About({ onClose }) {
  return (
    <aside className="pg-about" aria-label="About the playground">
      <div className="pg-about-head">
        <h2>About</h2>
        <button type="button" className="pg-quiet" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <p>Type some text and the labels you care about. GLiNER Runner picks the label that fits best and scores each one from 0 to 1.</p>
      <p>Tick <strong>Several labels can fit</strong> when a text can have more than one label, or none.</p>
      <p>Your text stays on your device: the model runs in this browser, or on your own computer.</p>
      <ul>
        <li><a href={`${BASE}/how-it-works/`}>How it works</a></li>
        <li><a href={`${BASE}/examples/`}>Examples</a></li>
        <li><a href={`${BASE}/use-in-react/#the-browser-playground`}>How the playground works</a></li>
        <li><a href={`${BASE}/start/#install`}>Install the runner</a></li>
      </ul>
    </aside>
  );
}

// The playground: gliner-runner itself running in this tab (a Web Worker with WebGPU or WASM), or the runner on
// this computer over HTTP. Both are driven through the runner's /v1 API.
export default function Playground() {
  const params = new URLSearchParams(window.location.search);
  const [mode, setMode] = useState(() => params.get("mode") ?? stored(KEYS.mode, "tab"));
  // On narrow screens the panel covers the playground, so it only reopens by itself on wide ones.
  const [about, setAbout] = useState(
    () => stored(KEYS.about, "closed") === "open" && window.matchMedia("(min-width: 60rem)").matches,
  );
  const [status, setStatus] = useState({ tone: "idle", text: "Starting" });
  const choose = (next) => {
    store(KEYS.mode, next);
    setMode(next);
  };
  const toggleAbout = (open) => {
    store(KEYS.about, open ? "open" : "closed");
    setAbout(open);
  };
  return (
    <div className={`pg-app${about ? " pg-with-about" : ""}`}>
      <div className="pg-main">
        <header className="pg-bar">
          <h1>Playground</h1>
          <span className={`pg-status pg-${status.tone}`}>{status.text}</span>
          <button type="button" className="pg-quiet pg-about-toggle" aria-expanded={about} onClick={() => toggleAbout(!about)}>
            {about ? "Hide about" : "About"}
          </button>
        </header>
        {mode === "local" ? (
          <OnMyMachine params={params} setStatus={setStatus} useBrowser={() => choose("tab")} />
        ) : (
          <InTab params={params} setStatus={setStatus} useMachine={() => choose("local")} />
        )}
      </div>
      {about && <About onClose={() => toggleAbout(false)} />}
    </div>
  );
}
