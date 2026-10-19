import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { brushX } from "d3-brush";
import { scaleLinear, scaleLog } from "d3-scale";
import { select } from "d3-selection";
import { line } from "d3-shape";
import "./Trace.css";
import { allLevels, inRange, leafRanking, levelsFor, pathsTo, thingNames, trace } from "./twentyQuestionsTrace.js";

/** The smallest probability the log scale shows; anything below it sits at the left edge. */
const FLOOR = 1e-4;
/** Quick ranges for the brush, for when dragging is awkward (keyboard, small screens). */
const PRESETS = [
  { name: "All", range: [0, 1] },
  { name: "Above 0.001", range: [0.001, 1] },
  { name: "Above 0.01", range: [0.01, 1] },
  { name: "Above 0.1", range: [0.1, 1] },
];

/** "0.73", "0.008", "2e-4". */
const two = (p) => (p >= 0.0095 || p === 0 ? p.toFixed(2) : p >= 0.001 ? p.toFixed(3) : p.toExponential(0));
/** "1st", "2nd", "13th", "22nd". */
const place = (n) => {
  const suffix = ["th", "st", "nd", "rd"];
  const tens = n % 100;
  return `${n}${suffix[(tens - 20) % 10] ?? suffix[tens] ?? suffix[0]}`;
};
const times = (ratio) => `${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio)}×`;

/** A scale over [low, high] for the chosen kind, mapped onto [0, width]. */
function probabilityScale(kind, [low, high], width) {
  return kind === "log"
    ? scaleLog().domain([Math.max(low, FLOOR), Math.max(high, FLOOR * 10)]).range([0, width]).clamp(true)
    : scaleLinear().domain([low, high]).range([0, width]).clamp(true);
}

/** The width of an element, kept up to date. */
function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/**
 * The probability range: every option's probability as a tick on one axis, and a brush to pick the part to look at.
 * The bars below zoom to that range, so small differences inside it are easy to see.
 */
function RangeStrip({ kind, range, onRange, ticks }) {
  const [ref, width] = useWidth();
  const brushArea = useRef(null);
  const brush = useRef(null);
  const height = 44;
  const x = useMemo(() => probabilityScale(kind, kind === "log" ? [FLOOR, 1] : [0, 1], width), [kind, width]);

  useEffect(() => {
    if (!brushArea.current || width === 0) return;
    brush.current = brushX()
      .extent([[0, 4], [width, height - 16]])
      .on("end", (event) => {
        if (!event.sourceEvent) return; // moved by the code below
        if (!event.selection) onRange([0, 1]);
        else onRange(event.selection.map((px) => x.invert(px)).map((p, i) => (i === 0 && p <= FLOOR ? 0 : p)));
      });
    select(brushArea.current).call(brush.current);
  }, [x, width]);

  useEffect(() => {
    if (!brush.current || width === 0) return;
    const [low, high] = range;
    const whole = low <= (kind === "log" ? FLOOR : 0) && high >= 1;
    select(brushArea.current).call(brush.current.move, whole ? null : [x(Math.max(low, kind === "log" ? FLOOR : 0)), x(high)]);
  }, [range, x, width]);

  const axis = kind === "log" ? [1e-4, 1e-3, 0.01, 0.1, 1] : [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="tr-strip" ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={`Probability range ${two(range[0])} to ${two(range[1])}`}>
          <line className="tr-axis" x1={0} x2={width} y1={height - 16} y2={height - 16} />
          {axis.map((value) => (
            <g key={value} transform={`translate(${x(value)},${height - 16})`}>
              <line className="tr-axis" y2={4} />
              <text className="tr-axis-text" y={14} textAnchor={value === axis[0] ? "start" : value === 1 ? "end" : "middle"}>{two(value)}</text>
            </g>
          ))}
          {ticks.map((tick, index) => (
            <line key={index} className={`tr-tick${tick.target ? " tr-tick-target" : ""}`} x1={x(Math.max(tick.p, FLOOR))} x2={x(Math.max(tick.p, FLOOR))} y1={tick.target ? 6 : 14} y2={height - 18} />
          ))}
          <g ref={brushArea} className="tr-brush" />
        </svg>
      )}
    </div>
  );
}

/** One step on the way: its question, how it went for the option on the way, and its options as bars. */
function Step({ step, kind, range }) {
  const zoom = probabilityScale(kind, range, 100);
  const { shown, above, below } = inRange(step.options, range);
  const onPath = step.options.filter((option) => option.onPath);
  return (
    <section className="tr-step">
      <header className="tr-step-head">
        <p className="tr-step-question">{step.question}</p>
        <p className="tr-step-meta">
          {step.n} options ·{" "}
          <abbr title="1 − normalised entropy: 1 when all the probability is on one option, 0 when every option is equally likely">
            sureness {step.sureness.toFixed(2)}
          </abbr>
        </p>
        {onPath.map((option) => (
          <p key={option.label} className={`tr-status tr-${option.status}`}>
            {option.status === "ok" ? "✓" : option.status === "close" ? "~" : "✗"} {option.label}: {place(option.rank)} of {step.n},{" "}
            {option.rank === 1
              ? <abbr title="Its probability minus the next best option's">ahead by {option.lead.toFixed(2)}</abbr>
              : <abbr title={`The best option's probability over its own (a log-odds gap of ${Math.log(option.behind).toFixed(2)})`}>{times(option.behind)} less likely than {step.options[0].label}</abbr>}
          </p>
        ))}
      </header>
      {above > 0 && <p className="tr-more">+{above} above {two(range[1])}</p>}
      <ul className="tr-options">
        {shown.map((option) => (
          <li
            key={option.label}
            className={[option.onPath && "tr-on-path", option.isTarget && "tr-target", option.isWinner && "tr-winner"].filter(Boolean).join(" ")}
          >
            <span className="tr-label">
              {option.label}
              {option.ruledOut && <span className="tr-ruled-out" title="The game guessed this and you said no">✗ guessed</span>}
            </span>
            <span className="tr-bar" aria-hidden="true"><span style={{ width: `${Math.max(zoom(option.p), option.p > 0 ? 1 : 0)}%` }} /></span>
            <span className="tr-p">{two(option.p)}</span>
            {option.child && (
              <div className="tr-child">{option.child.scored ? <Step step={option.child} kind={kind} range={range} /> : <p className="tr-more">Scoring…</p>}</div>
            )}
          </li>
        ))}
      </ul>
      {below > 0 && <p className="tr-more">+{below} below {two(range[0])}</p>}
    </section>
  );
}

/** Every thing the game knows, by the chance of its whole path, with the one being traced marked. */
function Ranking({ ranking, name, range }) {
  const [ref, width] = useWidth();
  const height = 160;
  const pad = { left: 36, right: 8, top: 8, bottom: 22 };
  const x = scaleLog().domain([1, Math.max(ranking.length, 2)]).range([pad.left, width - pad.right]);
  const y = scaleLog().domain([Math.max(ranking.at(-1)?.p ?? FLOOR, 1e-12), 1]).range([height - pad.bottom, pad.top]).clamp(true);
  const curve = line().x((_, i) => x(i + 1)).y((d) => y(Math.max(d.p, 1e-12)))(ranking);
  const rank = ranking.findIndex((r) => r.label === name) + 1;
  const [hover, setHover] = useState(undefined);
  const onMove = (event) => {
    const box = event.currentTarget.getBoundingClientRect();
    const index = Math.round(x.invert(event.clientX - box.left)) - 1;
    setHover(ranking[Math.min(Math.max(index, 0), ranking.length - 1)]);
  };
  const shown = ranking.filter((r) => r.p >= range[0] && r.p <= range[1]).length;
  return (
    <section className="tr-ranking">
      <p className="tr-step-question">Everything the game knows, most likely first</p>
      <p className="tr-step-meta">
        {ranking.length} things · {shown} in the range{hover && <> · <strong>{hover.label}</strong> {two(hover.p)}</>}
      </p>
      <div ref={ref} className="tr-curve">
        {width > 0 && (
          <svg width={width} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(undefined)} role="img"
            aria-label={rank > 0 ? `${name} is ${place(rank)} of ${ranking.length}` : `${name} is not ranked`}>
            <rect className="tr-band" x={pad.left} width={width - pad.left - pad.right}
              y={y(Math.min(range[1], 1))} height={Math.max(0, y(Math.max(range[0], 1e-12)) - y(Math.min(range[1], 1)))} />
            {[1, 0.1, 0.01, 0.001].map((p) => (
              <text key={p} className="tr-axis-text" x={pad.left - 4} y={y(p) + 4} textAnchor="end">{two(p)}</text>
            ))}
            {[1, 10, 100].filter((r) => r <= ranking.length).map((r) => (
              <text key={r} className="tr-axis-text" x={x(r)} y={height - 6} textAnchor="middle">#{r}</text>
            ))}
            <path className="tr-line" d={curve} />
            {hover && <circle className="tr-hover" cx={x(ranking.indexOf(hover) + 1)} cy={y(Math.max(hover.p, 1e-12))} r={3.5} />}
            {rank > 0 && <circle className="tr-dot" cx={x(rank)} cy={y(Math.max(ranking[rank - 1].p, 1e-12))} r={5} />}
          </svg>
        )}
      </div>
      <ol className="tr-top">
        {ranking.slice(0, 10).map((r, index) => (
          <li key={r.nodeKey} className={r.label === name ? "tr-target" : ""}><span>{index + 1}.</span> {r.label} <span>{two(r.p)}</span></li>
        ))}
      </ol>
    </section>
  );
}

/**
 * "Why?": how the clues fare on the way to one thing. It scores the steps on the way (reusing the game's own
 * classifications for the same clues), never as questions of the game; "Score everything" scores every step, for
 * the overall rank.
 */
export default function Trace({ tree, game, infer, target: initial, onClose }) {
  const names = useMemo(() => thingNames(tree), [tree]);
  const [typed, setTyped] = useState(initial);
  const target = typed.trim().toLowerCase();
  const [kind, setKind] = useState("log");
  const [range, setRange] = useState([0.01, 1]);
  const [levels, setLevels] = useState(() => new Map());
  const [progress, setProgress] = useState(undefined);
  const [error, setError] = useState("");
  const scored = useRef({ clues: undefined, levels: new Map() });
  const panel = useRef(null);

  useEffect(() => setTyped(initial), [initial]);
  useEffect(() => panel.current?.scrollIntoView({ behavior: "smooth", block: "start" }), []);

  /** Scores `wanted` levels for the current clues, reusing what is known; `count` reports progress. */
  async function score(wanted, count) {
    const clues = game.clues;
    if (scored.current.clues !== clues) scored.current = { clues, levels: new Map() };
    const known = scored.current.levels;
    for (const [nodeKey, level] of game.levels) if (level.clues === clues) known.set(nodeKey, level.probabilities);
    const missing = wanted.filter((level) => !known.has(level.nodeKey));
    setLevels(new Map(known));
    setError("");
    for (const [index, level] of missing.entries()) {
      if (scored.current.clues !== clues) return;
      count?.(index, missing.length);
      try {
        known.set(level.nodeKey, await infer({ text: clues, labels: level.labels, descriptions: level.descriptions }));
      } catch (problem) {
        setError(problem.message);
        return;
      }
      setLevels(new Map(known));
    }
  }

  // The steps on the way to the target, again whenever the clues or the target change (after a pause in typing).
  useEffect(() => {
    if (!game.clues.trim() || pathsTo(tree, target).length === 0) return undefined;
    const timer = setTimeout(() => score(levelsFor(tree, pathsTo(tree, target))), 400);
    return () => clearTimeout(timer);
  }, [target, game.clues]);

  const result = useMemo(() => trace(tree, target, levels, { ruledOut: game.ruledOut }), [tree, target, levels, game.ruledOut]);
  const everything = allLevels(tree);
  const complete = everything.every((level) => levels.has(level.nodeKey));
  const ranking = useMemo(() => (complete ? leafRanking(tree, levels) : undefined), [complete, levels]);
  const rank = ranking ? ranking.findIndex((r) => r.label === target) + 1 : 0;
  const best = result.best;

  // Keep the target in view when it first scores: the range starts just below its lowest step.
  const lowest = best?.complete ? Math.min(...best.steps.map((s) => s.option.p)) : undefined;
  useEffect(() => {
    if (lowest !== undefined && range[0] > lowest) setRange([Math.max(0, lowest / 2), 1]);
  }, [lowest]);

  const ticks = [];
  (function collect(step) {
    if (!step?.scored) return;
    for (const option of step.options) {
      ticks.push({ p: option.p, target: option.onPath });
      collect(option.child);
    }
  })(result.root);

  async function scoreEverything() {
    await score(everything, (index, total) => setProgress(`${index + 1} of ${total}`));
    setProgress(undefined);
  }

  return (
    <section className="tr" ref={panel} aria-label="Why">
      <header className="tr-head">
        <h2>Why?</h2>
        <label className="tr-target-field">
          <span className="field-label">The way to</span>
          <input className="input" list="tr-names" value={typed} placeholder="What was it?" onChange={(e) => setTyped(e.target.value)} />
          <datalist id="tr-names">{names.map((name) => <option key={name} value={name} />)}</datalist>
        </label>
        <button type="button" className="btn btn-sm" onClick={onClose} aria-label="Close">✕</button>
      </header>

      {!target ? (
        <p className="field-hint">Name the thing you were describing to see each step on the way to it.</p>
      ) : (
        <>
          <p className="tr-verdict">{result.verdict}</p>
          {best?.complete && (
            <dl className="tr-chips">
              <div title="The product of every step's probability (as the game tempers it). The game ranks things by this.">
                <dt>Path chance</dt><dd>{two(best.pathP)}</dd>
              </div>
              <div title="The lowest probability on the path (the minimum): the step most likely to send the game elsewhere.">
                <dt>Weakest step</dt><dd>{two(best.weakest.option.p)} <span>{best.weakest.step.question}</span></dd>
              </div>
              <div title="The geometric mean of the steps' probabilities: what a typical step gave, fairly for paths of any length.">
                <dt>Typical step</dt><dd>{two(best.geoMean)}</dd>
              </div>
              <div title="Where the thing ranks among everything the game knows, by path chance. Needs every step scored.">
                <dt>Overall</dt>
                <dd>
                  {ranking ? (rank > 0 ? `${place(rank)} of ${ranking.length}` : "—") : (
                    <button type="button" className="btn btn-sm" onClick={scoreEverything} disabled={progress !== undefined}>
                      {progress ? `Scoring ${progress}` : "Score everything"}
                    </button>
                  )}
                </dd>
              </div>
            </dl>
          )}
          {result.root && (
            <>
              <div className="tr-range">
                <span className="field-label">Probability range</span>
                <div className="tr-range-controls">
                  {PRESETS.map((preset) => (
                    <button key={preset.name} type="button" className={`chip${preset.range[0] === range[0] && preset.range[1] === range[1] ? " tr-chip-on" : ""}`}
                      aria-pressed={preset.range[0] === range[0] && preset.range[1] === range[1]}
                      onClick={() => setRange(preset.range)}>{preset.name}</button>
                  ))}
                  <span className="tr-scale" role="group" aria-label="Scale" title="A log scale spreads out small probabilities; a linear one shows them as they are">
                    {["log", "linear"].map((option) => (
                      <button key={option} type="button" className={`chip${kind === option ? " tr-chip-on" : ""}`} aria-pressed={kind === option}
                        onClick={() => setKind(option)}>{option === "log" ? "Log" : "Linear"}</button>
                    ))}
                  </span>
                </div>
                <RangeStrip kind={kind} range={range} onRange={setRange} ticks={ticks} />
                <span className="field-hint">Drag across the axis to zoom the bars into part of it. Options outside the range are counted, not drawn.</span>
              </div>
              {result.root.scored ? <Step step={result.root} kind={kind} range={range} /> : <p className="tr-more">Scoring…</p>}
              {ranking && <Ranking ranking={ranking} name={target} range={range} />}
            </>
          )}
          {error && <p className="pg-error">{error}</p>}
        </>
      )}
    </section>
  );
}
