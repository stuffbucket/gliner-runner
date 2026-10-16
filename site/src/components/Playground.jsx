import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { HttpTransport, INSTALL_PEAK_BYTES, WorkerTransport } from "@gliner-runner/onnx-runner-web/client";
import "./Playground.css";
import twenty from "../data/twenty-questions.json";
import { MAX_QUESTIONS, afterClassification, afterReply, guessQuestion, newGame, nextStep, startGuess, verdict } from "./twentyQuestions.js";

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
const MAC = /Mac|iPhone|iPad/.test(navigator.userAgentData?.platform ?? navigator.platform ?? navigator.userAgent);

const EXAMPLES = [
  { name: "Support ticket", text: "I was charged twice for my last order.", labels: "billing, support, shipping" },
  { name: "Review", text: "The battery lasts all day, but the screen scratches far too easily.", labels: "positive, negative, mixed" },
  { name: "Topics", text: "I will pay my rent on Friday, and then I will bake bread.", labels: "weather, cooking, sports, money", several: true },
  { name: "Urgency", text: "The production database is down and customers cannot log in.", labels: "urgent, can wait" },
  { name: "20 questions", game: true },
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
const request = (model, text, labels, several, descriptions) => ({
  model, backend: "onnx", precision: "fp32", operation: "classify", text,
  schema: {
    kind: "classification",
    tasks: {
      category: {
        labels,
        ...(descriptions ? { label_descriptions: descriptions } : {}),
        ...(several ? { min_labels: 0, max_labels: null } : {}),
      },
    },
  },
});

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

function Shortcut() {
  return <span className="kbd" aria-hidden="true">{MAC ? "⌘" : "Ctrl"} ↵</span>;
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
      <button type="button" className="btn btn-sm" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
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

/** How many answers the stack keeps; older ones are dropped. */
const STACK_LIMIT = 50;
/** Each step down the stack keeps this much of a card's opacity. */
const FADE_PER_STEP = 0.65;

const motion = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/**
 * The answer as a standalone card: the question it answered (in a game), what was sent, which labels fit, every
 * score, and how it ran. A verdict card closes a game.
 */
function Answer({ answer, depth }) {
  const ms = (value) => (value < 10 ? value.toFixed(1) : Math.round(value));
  const common = {
    className: `answer-card pg-answer${answer.kind === "verdict" ? " pg-verdict" : ""}`,
    "data-id": answer.id,
    style: { opacity: FADE_PER_STEP ** depth },
    "aria-hidden": depth > 0 ? "true" : undefined,
  };
  if (answer.kind === "verdict") {
    return (
      <article {...common}>
        <p className="pg-question">20 questions</p>
        <p className="pg-verdict-text">{answer.message}</p>
      </article>
    );
  }
  if (answer.kind === "guess") {
    return (
      <article {...common}>
        <p className="pg-question">{answer.question}</p>
        <p className="pg-guess-text">
          {guessQuestion(answer.label)}
          {answer.reply && <span className={`pg-reply pg-reply-${answer.reply}`}>{answer.reply === "yes" ? "Yes" : "No"}</span>}
        </p>
      </article>
    );
  }
  const ranked = Object.entries(answer.scores).sort((a, b) => b[1] - a[1]);
  return (
    <article {...common}>
      {answer.question && <p className="pg-question">{answer.question}</p>}
      <p className="answer-quote">“{answer.text}”</p>
      <p className="answer-labels">
        {answer.labels.map((label) => (
          <span key={label} className={answer.picked.includes(label) ? "picked" : ""}>{label}</span>
        ))}
      </p>
      {answer.picked.length === 0 && <p className="pg-none">No label fits well enough.</p>}
      <ul className="answer-scores">
        {ranked.map(([label, score]) => (
          <li key={label} className={answer.picked.includes(label) ? "picked" : ""}>
            <span>{label}</span>
            <span className="meter"><span style={{ width: `${Math.max(2, Math.round(score * 100))}%` }} /></span>
            <span>{score.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="answer-meta">
        <span className="model-badge">{answer.model}</span>
        {answer.several && <span>several labels allowed</span>}
        <span
          className="pg-elapsed"
          title={answer.modelMs === undefined ? undefined : `${ms(answer.modelMs)} ms of it in the model`}
        >
          Answered in {ms(answer.elapsed)} ms
        </span>
      </p>
    </article>
  );
}

/**
 * The answers from the card being looked at down to the oldest. When the cards change, the ones that stay slide to
 * their new places (measured before and after the update and animated between, so the move follows the shared
 * easing) and fade by a step, and a card that appears (a new answer, or a newer one shown again) eases in at the top.
 */
function AnswerStack({ answers, game }) {
  const list = useRef(null);
  const tops = useRef(new Map());
  useLayoutEffect(() => {
    const duration = parseFloat(motion("--duration-move", "420ms"));
    const easing = motion("--ease-out", "ease-out");
    const seen = new Map();
    for (const node of list.current?.children ?? []) {
      const id = node.dataset.id;
      const top = node.offsetTop;
      const before = tops.current.get(id);
      if (duration > 0 && before === undefined) {
        node.animate([{ opacity: 0, transform: "translateY(-0.75rem) scale(0.985)" }], { duration, easing });
      } else if (duration > 0 && before !== top) {
        node.animate([{ transform: `translateY(${before - top}px)` }, { transform: "none" }], { duration, easing });
      }
      seen.set(id, top);
    }
    tops.current = seen;
  }, [answers]);

  if (answers.length === 0) {
    return (
      <div className="answer-card pg-placeholder">
        <p>{game ? "The model's questions and guesses appear here." : "The answer appears here."}</p>
        <p className="pg-small">Press {game ? "Start" : "Classify"} or <Shortcut /></p>
      </div>
    );
  }
  return (
    <div className="pg-stack-cards" ref={list}>
      {answers.map((answer, depth) => <Answer key={answer.id} answer={answer} depth={depth} />)}
    </div>
  );
}

/**
 * Text, labels and the answers; the same for both runners. The model is loaded in the background as soon as the
 * workspace appears (and whenever another model is chosen), so the first Classify is as fast as the rest.
 */
function Workspace({ transport, models, onWarm }) {
  const [model, setModel] = useState(models.includes(IN_TAB_MODEL) ? IN_TAB_MODEL : models[0]);
  const [text, setText] = useState(EXAMPLES[0].text);
  const [labels, setLabels] = useState(EXAMPLES[0].labels);
  const [several, setSeveral] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answers, setAnswers] = useState([]);
  const [error, setError] = useState("");
  const [game, setGame] = useState(null);
  /** How many newer cards sit above the one being looked at (0: the newest). */
  const [back, setBack] = useState(0);
  const next = useRef(1);
  /** Settles once the background load of the current model has finished (whatever its outcome). */
  const loaded = useRef(Promise.resolve());

  useEffect(() => {
    let current = true;
    onWarm("loading", model);
    const warm = transport.request("POST", "/v1/infer", request(model, "Hello.", ["greeting", "question"], false)).then(
      (reply) => current && onWarm(reply.status === 200 ? "ready" : "failed", model),
      () => current && onWarm("failed", model),
    );
    loaded.current = warm;
    return () => {
      current = false;
    };
  }, [model]);

  /** Puts a card on top of the stack; returns its id. Someone looking back through older cards stays on the same one. */
  const push = (card) => {
    const id = String(next.current++);
    setAnswers((previous) => [{ id, ...card }, ...previous].slice(0, STACK_LIMIT));
    setBack((newer) => (newer > 0 ? newer + 1 : 0));
    return id;
  };

  const pick = (example) => {
    if (example.game) {
      clues.current = twenty.sample;
      setGame(newGame(twenty.tree, twenty.sample));
      setPlaying(false);
      return;
    }
    setGame(null);
    setPlaying(false);
    setText(example.text);
    setLabels(example.labels);
    setSeveral(example.several ?? false);
  };

  /** Classifies once and puts the answer on the stack; resolves with the answer, or nothing after an error. */
  async function run({ text, labels: list, several, descriptions, question }) {
    setError("");
    setBusy(true);
    // A request sent while the model is still loading waits for it; that wait is not the classification's time.
    await loaded.current;
    const started = performance.now();
    try {
      const reply = await transport.request("POST", "/v1/infer", request(model, text, list, several, descriptions));
      const data = parse(reply);
      if (reply.status !== 200) {
        setError(message(data, "The runner rejected the request."));
        return undefined;
      }
      const { value, probabilities } = data.output.category;
      const answer = {
        question, text, labels: list, several, model,
        picked: [value].flat().filter(Boolean),
        scores: probabilities,
        elapsed: performance.now() - started,
        modelMs: data.timing?.inference_ms,
      };
      push(answer);
      return answer;
    } catch {
      setError("Lost the connection to the runner.");
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  function classify() {
    const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
    if (busy || !text.trim() || list.length === 0) return;
    run({ text, labels: list, several });
  }

  // 20 questions: the player only writes clues and answers guesses. Once started, the game classifies on its own
  // until it has a guess to check, or runs out of questions.
  const [playing, setPlaying] = useState(false);
  const clues = useRef("");
  const guessCard = useRef(undefined);
  const cluesBox = useRef(null);

  const setClues = (value) => {
    clues.current = value;
    setGame((current) => ({ ...current, clues: value }));
  };

  /** If the game's next move is a guess, puts it on the stack and asks it (a guess counts as a question). */
  function advance(state) {
    if (state.phase !== "asking") return state;
    const step = nextStep(state);
    if (step?.kind !== "guess") return state;
    guessCard.current = push({ kind: "guess", question: `Question ${step.number} of ${MAX_QUESTIONS}`, label: step.label, model });
    return startGuess(state, step);
  }

  /** One question of 20 questions: classify the clues at the most likely unexplored level (see twentyQuestions.js). */
  async function ask() {
    if (busy || game?.phase !== "asking" || !game.clues.trim()) return;
    const step = nextStep(game);
    if (step.kind === "guess") {
      setGame(advance(game));
      return;
    }
    const answer = await run({ text: game.clues, labels: step.labels, descriptions: step.descriptions, question: step.text });
    if (answer === undefined) {
      setPlaying(false);
      return;
    }
    // Clues typed while the model was thinking are kept for the next question.
    let state = { ...afterClassification(game, step, answer.scores), clues: clues.current };
    if (state.phase === "lost") push({ kind: "verdict", message: verdict(state), model });
    else state = advance(state);
    setGame(state);
  }

  useEffect(() => {
    if (!playing || game?.phase !== "asking" || busy || !game.clues.trim()) return undefined;
    // A short pause between questions, so each card can be read as it arrives.
    const timer = setTimeout(ask, 350);
    return () => clearTimeout(timer);
  }, [playing, game, busy]);

  function start() {
    if (!game?.clues.trim()) return;
    setPlaying(true);
  }

  function playAgain() {
    clues.current = "";
    setGame(newGame(twenty.tree, ""));
    setPlaying(false);
    cluesBox.current?.focus();
  }

  /** The player's answer to a guess. */
  function reply(yes) {
    if (game?.phase !== "guessing") return;
    const card = guessCard.current;
    setAnswers((previous) => previous.map((old) => (old.id === card ? { ...old, reply: yes ? "yes" : "no" } : old)));
    let state = afterReply(game, yes);
    if (state.phase === "won" || state.phase === "lost") push({ kind: "verdict", message: verdict(state), model });
    else state = advance(state);
    setGame(state);
  }

  const onKeyDown = (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!game) classify();
      else if (!playing && game.phase === "asking") start();
    }
  };
  const shortcut = MAC ? "Meta+Enter" : "Control+Enter";
  // The card on top of the stack (it stays put when cards beyond the limit are dropped).
  const shown = Math.min(back, Math.max(answers.length - 1, 0));

  return (
    <div className="pg-workspace" onKeyDown={onKeyDown}>
      <div className="pg-examples">
        <span className="field-label">Example</span>
        {EXAMPLES.map((example) => (
          <button
            type="button"
            key={example.name}
            className={`btn btn-sm${example.game && game ? " pg-active" : ""}`}
            onClick={() => pick(example)}
          >
            {example.name}
          </button>
        ))}
      </div>
      {game ? (
        <>
          <label className="field-label pg-text-label" htmlFor="pg-text">
            Think of an animal, a plant or a thing, and describe it without naming it
          </label>
          <textarea
            id="pg-text"
            ref={cluesBox}
            className="input pg-text"
            value={game.clues}
            onChange={(e) => setClues(e.target.value)}
            placeholder="It is small, round and red, and grows on trees."
          />
          <div className="field pg-labels pg-turn" aria-live="polite">
            {game.phase === "guessing" ? (
              <>
                <span className="field-label">Question {game.asked} of {MAX_QUESTIONS} · your turn</span>
                <p className="pg-turn-guess">{guessQuestion(game.guess)}</p>
                <span className="field-hint">Not quite? You can add a clue above before you answer.</span>
              </>
            ) : game.phase === "won" || game.phase === "lost" ? (
              <p className="pg-turn-guess">{verdict(game)}</p>
            ) : playing ? (
              <>
                <span className="field-label">Question {game.asked + 1} of {MAX_QUESTIONS}</span>
                <p className="pg-turn-status">{busy ? "Thinking…" : "Choosing the next question…"}</p>
                <span className="field-hint">It works out what to ask from your clues and stops to check each guess with you.</span>
              </>
            ) : (
              <span className="field-hint">
                {game.asked === 0
                  ? game.clues === twenty.sample
                    ? "This example describes a zebra; write your own, or start with it. The model then asks itself up to 20 questions and stops to check each guess with you."
                    : "Then press Start. The model asks itself up to 20 questions and stops to check each guess with you."
                  : "Stopped. Change the clues if you like, then carry on."}
              </span>
            )}
          </div>
          <div className="pg-actions">
            {game.phase === "asking" && !playing && (
              <button type="button" className="btn btn-primary pg-classify" onClick={start} disabled={!game.clues.trim()} aria-keyshortcuts={shortcut}>
                {game.asked === 0 ? "Start" : "Carry on"} <Shortcut />
              </button>
            )}
            {game.phase === "guessing" && (
              <>
                <button type="button" className="btn btn-primary" onClick={() => reply(true)}>Yes, that's it</button>
                <button type="button" className="btn" onClick={() => reply(false)}>No</button>
              </>
            )}
            {(game.phase === "won" || game.phase === "lost") && (
              <button type="button" className="btn btn-primary" onClick={playAgain}>Play again</button>
            )}
            <button type="button" className="btn btn-link" onClick={() => pick(EXAMPLES[0])}>Leave the game</button>
          </div>
        </>
      ) : (
        <>
          <label className="field-label pg-text-label" htmlFor="pg-text">Text</label>
          <textarea id="pg-text" className="input pg-text" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste or type some text" />
          <label className="field pg-labels">
            <span className="field-label">Labels</span>
            <input className="input" value={labels} onChange={(e) => setLabels(e.target.value)} placeholder="billing, support, shipping" />
            <span className="field-hint">Separate labels with commas.</span>
          </label>
          <div className="pg-actions">
            <label className="switch">
              <input type="checkbox" role="switch" checked={several} onChange={(e) => setSeveral(e.target.checked)} />
              <span className="switch-track" aria-hidden="true" />
              Several labels can fit
            </label>
            {models.length > 1 && (
              <select className="input" aria-label="Model" value={model} onChange={(e) => setModel(e.target.value)}>
                {models.map((id) => <option key={id} value={id}>{id}</option>)}
              </select>
            )}
            <button
              type="button"
              className="btn btn-primary pg-classify"
              onClick={classify}
              disabled={busy || !text.trim()}
              aria-keyshortcuts={shortcut}
            >
              {busy ? "Classifying…" : "Classify"} <Shortcut />
            </button>
          </div>
        </>
      )}
      {error && <p className="pg-error pg-message">{error}</p>}
      {answers.length > 0 && (
        <div className="pg-history" role="group" aria-label="Answer history">
          <button type="button" className="btn btn-sm pg-history-step" aria-label="Older answer" title="Older"
            disabled={shown >= answers.length - 1} onClick={() => setBack(shown + 1)}>‹</button>
          <span className="pg-history-count">{answers.length - shown} of {answers.length}</span>
          <button type="button" className="btn btn-sm pg-history-step" aria-label="Newer answer" title="Newer"
            disabled={shown === 0} onClick={() => setBack(shown - 1)}>›</button>
        </div>
      )}
      <section className="pg-stack" aria-live="polite" aria-label="Answers">
        <AnswerStack answers={answers.slice(shown)} game={game !== null} />
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

const ENGINE = {
  webgpu: { short: "GPU", title: "Running on your graphics card (GPU) through WebGPU" },
  wasm: { short: "CPU", title: "Running on your processor (CPU) through WebAssembly" },
};

function InTab({ params, setStatus, useMachine }) {
  const [transport, setTransport] = useState(null);
  const [host, setHost] = useState({ state: "starting" });
  const [entry, setEntry] = useState(null);
  const [progress, setProgress] = useState(null);
  const [provider, setProvider] = useState(null);
  const [warm, setWarm] = useState("loading");
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
    const engine = provider ? ENGINE[provider.provider] : undefined;
    setStatus(
      blocked
        ? { tone: "warn", text: "Not available in this window" }
        : job
          ? { tone: "busy", text: "Getting the model" }
          : entry?.available_locally
            ? warm === "ready"
              ? { tone: "ok", text: engine ? `Ready · ${engine.short}` : "Ready", title: engine?.title }
              : warm === "failed"
                ? { tone: "warn", text: "The model could not load" }
                : { tone: "busy", text: "Loading the model" }
            : entry === null
              ? { tone: "idle", text: "Starting" }
              : { tone: "idle", text: "Model not downloaded" },
    );
  }, [blocked, entry, job, provider, warm]);

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
    setWarm("loading");
    await transport.deleteModel();
    await refresh();
  }

  const elsewhere = <button type="button" className="btn btn-link" onClick={useMachine}>Run it on my computer instead</button>;
  if (blocked) {
    return (
      <Card title="This window can't keep the model">
        <p>{host.reason} Open the page in a normal (not private) window, or run the model on your computer.</p>
        <button type="button" className="btn btn-primary" onClick={useMachine}>Run it on my computer</button>
      </Card>
    );
  }
  if (host.state === "starting" || !transport || entry === null) {
    return <Card><p className="pg-muted">Starting…</p></Card>;
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
            <div className="pg-card-actions">
              <button type="button" className="btn btn-primary" onClick={download} disabled={!enough}>Download</button>
              {elsewhere}
            </div>
            {error && <p className="pg-error">{error}</p>}
          </>
        ) : (
          <>
            <progress max={1} value={fraction} />
            <p className="pg-small">
              {progress ? `${STAGES[progress.stage] ?? "Working"} · ${gb(progress.done)} of ${gb(progress.total)}` : "Starting…"}
            </p>
            <div className="pg-card-actions">
              <button type="button" className="btn btn-sm" onClick={cancel}>Cancel</button>
            </div>
          </>
        )}
      </Card>
    );
  }
  return (
    <>
      <Workspace transport={transport} models={[IN_TAB_MODEL]} onWarm={(state) => setWarm(state)} />
      <p className="pg-footer">
        Runs in this browser{provider ? ` on your ${provider.provider === "webgpu" ? "graphics card (GPU)" : "processor (CPU)"}` : ""}.{" "}
        <button type="button" className="btn btn-link" onClick={remove}>Remove the model</button> · {elsewhere}
      </p>
    </>
  );
}

function OnMyMachine({ params, setStatus, useBrowser }) {
  const [address, setAddress] = useState(() => params.get("runner") ?? stored(KEYS.address, DEFAULT_ADDRESS));
  const [state, setState] = useState({ kind: "probing" });
  const [warm, setWarm] = useState({ state: "loading", model: "" });
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const base = address.trim().replace(/\/+$/, "");
  const transport = useRef(null);
  if (transport.current?.base !== base) transport.current = new HttpTransport(base);

  async function probe() {
    try {
      const data = parse(await transport.current.request("GET", "/v1/models"));
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

  const installed = state.kind === "connected" ? state.models.filter((m) => m.available_locally).map((m) => m.model_id) : [];
  useEffect(() => {
    setStatus(
      state.kind !== "connected"
        ? { tone: "busy", text: "Waiting for your computer" }
        : installed.length === 0
          ? { tone: "idle", text: "Connected · no model yet" }
          : warm.state === "ready"
            ? { tone: "ok", text: "Ready · your computer", title: `${warm.model} is loaded in the runner at ${base}` }
            : warm.state === "failed"
              ? { tone: "warn", text: "The model could not load" }
              : { tone: "busy", text: `Loading ${warm.model}` },
    );
  }, [state.kind, installed.length, warm]);

  async function download(entry) {
    setError("");
    try {
      await install(transport.current, entry, setJob);
      await probe();
    } catch (failure) {
      setError(failure.message);
    } finally {
      setJob(null);
    }
  }

  const back = <button type="button" className="btn btn-link" onClick={useBrowser}>Back to this browser</button>;
  if (state.kind !== "connected") {
    return (
      <Card title="Run it on your computer">
        <p>Paste this into a terminal. It sets up the runner and connects this page.</p>
        <Command>{state.kind === "blocked" ? "gliner-runner playground --yes --no-open" : SETUP_COMMAND}</Command>
        <p className="pg-muted pg-waiting">{state.kind === "probing" ? "Looking for the runner…" : "Waiting for the runner…"}</p>
        <details className="pg-advanced">
          <summary>Runner address</summary>
          <input className="input" aria-label="Runner address" value={address} onChange={(e) => setAddress(e.target.value)} spellCheck={false} />
        </details>
        <div className="pg-card-actions">{back}</div>
      </Card>
    );
  }
  const missing = state.models.find((model) => model.model_id === IN_TAB_MODEL && !model.available_locally);
  if (installed.length === 0) {
    return (
      <Card title="Connected, but no model yet">
        {job === null ? (
          <div className="pg-card-actions">
            <button type="button" className="btn btn-primary" onClick={() => download(missing)} disabled={!missing}>
              Download {IN_TAB_MODEL} to your computer
            </button>
          </div>
        ) : (
          <p className="pg-muted pg-waiting">Downloading and preparing on your computer… a few minutes.</p>
        )}
        {error && <p className="pg-error">{error}</p>}
        <div className="pg-card-actions">{back}</div>
      </Card>
    );
  }
  return (
    <>
      <Workspace transport={transport.current} models={installed} onWarm={(state, model) => setWarm({ state, model })} />
      <p className="pg-footer">Runs on your computer ({base}). {back}</p>
    </>
  );
}

function About({ onClose }) {
  return (
    <aside className="pg-about" aria-label="About the playground">
      <div className="pg-about-head">
        <h2>About</h2>
        <button type="button" className="btn btn-sm" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <p>Type some text and the labels you care about. GLiNER Runner picks the label that fits best and scores each one from 0 to 1.</p>
      <p>Turn on <strong>Several labels can fit</strong> when a text can have more than one label, or none.</p>
      <p>In <strong>20 questions</strong> you describe something without naming it. The model works out what it is by classifying your clues, and you only answer its guesses.</p>
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
          <span className={`pill pill-${status.tone}`} title={status.title} role="status">{status.text}</span>
          <button type="button" className="btn btn-sm pg-about-toggle" aria-expanded={about} onClick={() => toggleAbout(!about)}>
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
