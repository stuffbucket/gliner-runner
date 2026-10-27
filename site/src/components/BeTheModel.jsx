import { memo, useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import data from "virtual:be-the-model";
import "../styles/tokens.css";
import "./BeTheModel.css";
import { alongLabel, inferRequest, keySteps, nearestAlong, scoresFromReply, stepAlong, topLabel } from "./beTheModel.js";

// "Be the model": one slider per label, each at the model's score for the kid's line on show. Drag one, and the
// line becomes the example the model scored nearest that value, while the other sliders move to its scores. Let
// go, and the dragged slider settles on the example's own score, so the page only shows numbers the model gave.
// The arrow keys step from one example to the next along a slider. The choice of example is in beTheModel.js.
//
// Keeping a drag smooth: React renders only when the example on show changes, not on every move of the pointer.
// The dragged slider is moved directly (a CSS variable on its row, once per frame), and the other sliders glide
// to their new scores with CSS transitions on transforms. The range inputs are uncontrolled: the page sets their
// value when the example changes, so React never fights the pointer.
//
// When a model is ready to use (a runner on this computer, or the model already in this browser), the kid's line
// can be edited: the model scores the new line, and it becomes one more example the sliders can stop at.

const { labels, model, mock } = data;
const WIDE = "(min-width: 52rem)";
const ANNOUNCE_AFTER_MS = 700;
const two = (n) => n.toFixed(2);
const html = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const Row = memo(function Row({ name, description, score, isTop, noteOpen, inputId, noteId, actions }) {
  const ref = useRef(null);
  useLayoutEffect(() => actions.register(name, ref.current), []);
  // A new score: the input says so to assistive technology. A slider being dragged is left to the pointer.
  useLayoutEffect(() => {
    if (!ref.current.hasAttribute("data-dragging")) ref.current.querySelector("input").value = String(score);
  }, [score]);
  return (
    <li ref={ref} className="btm-row" data-top={isTop ? "" : undefined} style={{ "--at": String(score) }}>
      <span className="btm-name">
        <label htmlFor={inputId}>{name}</label>
        <button
          type="button"
          className="btm-info"
          aria-expanded={noteOpen}
          aria-controls={noteId}
          onClick={() => actions.toggleNote(name)}
          onKeyDown={(event) => event.key === "Escape" && actions.toggleNote(null)}
        >
          <span aria-hidden="true">?</span>
          <span className="btm-sr">What “{name}” means</span>
        </button>
        <span className="btm-note-pop" id={noteId} data-open={noteOpen ? "" : undefined}>
          {description}
        </span>
      </span>
      <span className="btm-value" aria-hidden="true">
        {two(score)}
      </span>
      <span className="btm-slider">
        <input
          id={inputId}
          type="range"
          min="0"
          max="1"
          step="0.001"
          defaultValue={score}
          aria-valuetext={`${name} ${two(score)}`}
          aria-describedby={noteId}
          onPointerDown={(event) => actions.grab(name, event)}
          onChange={(event) => actions.move(name, event.currentTarget.valueAsNumber)}
          onKeyDown={(event) => actions.key(name, event)}
          onFocus={() => actions.follow(name)}
        />
        <span className="btm-rail" aria-hidden="true">
          <span className="btm-fill" />
        </span>
        <span className="btm-mover" aria-hidden="true">
          <span className="btm-thumb" />
        </span>
      </span>
    </li>
  );
});

export default function BeTheModel() {
  const id = useId();
  const [examples, setExamples] = useState(data.examples);
  const [currentId, setCurrentId] = useState(data.examples[0].id);
  const [openNote, setOpenNote] = useState(null);
  const [announcement, setAnnouncement] = useState("");
  const [runner, setRunner] = useState({ state: "checking" });
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");

  const current = useMemo(() => examples.find((example) => example.id === currentId), [examples, currentId]);
  const orders = useMemo(() => Object.fromEntries(labels.map(({ name }) => [name, alongLabel(examples, name)])), [examples]);
  const top = topLabel(current.scores);

  // What the event handlers read, so they can stay the same functions for the life of the page.
  const live = useRef({});
  live.current = { current, orders };
  const rows = useRef(new Map());
  const card = useRef(null);
  const side = useRef(null);
  const line = useRef(null);
  const drag = useRef({ label: null, value: 0, frame: 0 });
  const followed = useRef(null);
  const nextMessage = useRef(undefined);
  const edit = useRef({ cancel: false, before: "" });
  const mine = useRef(0);

  // On a wide screen the kid's line eases up or down to sit beside the slider in use, inside the window.
  const place = useCallback(() => {
    const name = followed.current;
    const box = card.current;
    if (!name || !box || !side.current) return;
    if (!window.matchMedia(WIDE).matches) {
      box.style.setProperty("--follow", "0px");
      return;
    }
    const row = rows.current.get(name).getBoundingClientRect();
    const column = side.current.getBoundingClientRect();
    const height = box.offsetHeight;
    // Where the card sits without the shift: stuck under the header, or at the top of its column.
    const stuck = parseFloat(getComputedStyle(box).top) || 0;
    const natural = Math.min(Math.max(column.top, stuck), column.bottom - height);
    const want = row.top + row.height / 2 - height / 2 - natural;
    const room = Math.min(window.innerHeight - 16, column.bottom) - natural - height;
    box.style.setProperty("--follow", `${Math.round(Math.max(0, Math.min(want, room)))}px`);
  }, []);

  const actions = useMemo(() => {
    const frame = () => {
      const d = drag.current;
      d.frame = 0;
      if (d.label === null) return;
      rows.current.get(d.label).style.setProperty("--drag", String(d.value));
      const { current: now, orders: along } = live.current;
      const next = nearestAlong(along[d.label], d.label, d.value, now.scores);
      if (next.id !== now.id) setCurrentId(next.id);
    };
    const follow = (name) => {
      followed.current = name;
      place();
    };
    return {
      register(name, element) {
        rows.current.set(name, element);
      },
      toggleNote(name) {
        setOpenNote((open) => (name === null || open === name ? null : name));
      },
      follow,
      grab(name, event) {
        if (event.button !== 0) return;
        const row = rows.current.get(name);
        drag.current.label = name;
        row.style.setProperty("--drag", String(live.current.current.scores[name]));
        row.setAttribute("data-dragging", "");
        follow(name);
      },
      move(name, value) {
        const d = drag.current;
        if (d.label === name) {
          d.value = value;
          d.frame ||= requestAnimationFrame(frame);
          return;
        }
        // Moved some other way (a screen reader's swipe, say): step to the next example in that direction.
        const { current: now, orders: along } = live.current;
        const score = now.scores[name];
        if (value !== score) setCurrentId(stepAlong(along[name], now.id, value > score ? 1 : -1).id);
      },
      letGo() {
        const d = drag.current;
        if (d.label === null) return;
        if (d.frame) {
          cancelAnimationFrame(d.frame);
          frame();
        }
        const row = rows.current.get(d.label);
        row.removeAttribute("data-dragging");
        // Settle on the real score; if the example just changed, the row's next render does it.
        row.querySelector("input").value = String(live.current.current.scores[d.label]);
        d.label = null;
      },
      key(name, event) {
        if (event.key === "Escape") {
          setOpenNote(null);
          return;
        }
        const steps = keySteps(event.key);
        if (steps === undefined) return;
        event.preventDefault();
        const { current: now, orders: along } = live.current;
        setCurrentId(stepAlong(along[name], now.id, steps).id);
        follow(name);
      },
    };
  }, [place]);

  useEffect(() => {
    let frame = 0;
    const replace = () => {
      if (followed.current) frame ||= requestAnimationFrame(() => ((frame = 0), place()));
    };
    window.addEventListener("pointerup", actions.letGo);
    window.addEventListener("pointercancel", actions.letGo);
    window.addEventListener("scroll", replace, { passive: true });
    window.addEventListener("resize", replace);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointerup", actions.letGo);
      window.removeEventListener("pointercancel", actions.letGo);
      window.removeEventListener("scroll", replace);
      window.removeEventListener("resize", replace);
    };
  }, [actions, place]);

  // A new example: the card may change height, and the new line is read out once things are still.
  const first = useRef(true);
  useLayoutEffect(() => {
    place();
    if (first.current) {
      first.current = false;
      return undefined;
    }
    const message = nextMessage.current ?? `“${current.text}” The model's answer: ${top}.`;
    nextMessage.current = undefined;
    const timer = setTimeout(() => setAnnouncement(message), ANNOUNCE_AFTER_MS);
    return () => clearTimeout(timer);
  }, [current]);

  // Is a model ready to use? Only asked once the page is up, and never starts a download.
  useEffect(() => {
    let stopped = false;
    import("./beTheModelRunner.js")
      .then(({ findModel }) => findModel(model))
      .then(
        (found) => !stopped && setRunner(found ? { state: "ready", ...found } : { state: "none" }),
        () => !stopped && setRunner({ state: "none" }),
      );
    return () => {
      stopped = true;
    };
  }, []);

  const editable = runner.state === "ready" && !busy;

  async function score(text, before) {
    setBusy(true);
    setProblem("");
    try {
      let reply;
      try {
        reply = await runner.transport.request("POST", "/v1/infer", inferRequest(data, text));
      } catch {
        throw new Error("Lost the connection to the model.");
      }
      const { scores, error } = scoresFromReply(data, reply.status, reply.body);
      if (error) throw new Error(error);
      const from = live.current.current;
      const example = { id: `mine-${++mine.current}`, text, ask: from.ask, age: from.age, scores, mine: true };
      const answer = topLabel(scores);
      nextMessage.current = `New scores for your line. The model's answer: ${answer}, ${two(scores[answer])}.`;
      setExamples((list) => [...list, example]);
      setCurrentId(example.id);
    } catch (failure) {
      line.current.textContent = before;
      setProblem(failure.message);
    } finally {
      setBusy(false);
    }
  }

  const lineEvents = editable
    ? {
        onFocus() {
          edit.current = { cancel: false, before: live.current.current.text };
          setProblem("");
        },
        onInput(event) {
          // An emptied line can keep a stray <br>, which would hide the "Plead your case" hint.
          const element = event.currentTarget;
          if (element.textContent === "") element.replaceChildren();
        },
        onKeyDown(event) {
          if (event.key === "Enter") {
            event.preventDefault();
            event.currentTarget.blur();
          } else if (event.key === "Escape") {
            event.preventDefault();
            edit.current.cancel = true;
            event.currentTarget.blur();
          }
        },
        onBlur(event) {
          const element = event.currentTarget;
          const { cancel, before } = edit.current;
          const text = element.textContent.replace(/\s+/g, " ").trim();
          if (cancel || !text || text === before) {
            element.textContent = before;
            return;
          }
          score(text, before);
        },
      }
    : {};

  const hintId = `${id}-hint`;
  const hint = problem ? (
    <span className="btm-problem">{problem}</span>
  ) : busy ? (
    "The model is reading your line."
  ) : runner.state === "ready" ? (
    "Click the line to write your own. Enter scores it, Esc puts the old one back."
  ) : runner.state === "none" ? (
    <>
      Load the model in the <a href={`${import.meta.env.BASE_URL.replace(/\/?$/, "/")}playground/`}>playground</a> to write
      your own.
    </>
  ) : null;

  return (
    <div className="btm not-content">
      <div className="btm-side" ref={side}>
        <div className="btm-card" ref={card} aria-busy={busy}>
          <blockquote className="btm-quote">
            <p>
              “
              <span
                ref={line}
                className="btm-line"
                contentEditable={editable ? "plaintext-only" : undefined}
                suppressContentEditableWarning
                role={editable ? "textbox" : undefined}
                aria-label={editable ? "The kid's line" : undefined}
                aria-describedby={editable ? hintId : undefined}
                spellCheck={editable ? true : undefined}
                data-placeholder="Plead your case"
                dangerouslySetInnerHTML={{ __html: html(current.text) }}
                {...lineEvents}
              />
              ”
            </p>
          </blockquote>
          <p className="btm-meta">
            <span>
              Wants: <strong>{current.ask}</strong>
            </span>
            {current.mine ? <span>Your line</span> : current.age ? <span>Age {current.age}</span> : null}
            <span>
              The model's answer: <strong className="btm-answer">{top}</strong>
            </span>
          </p>
          <p className="btm-hint" id={hintId}>
            {hint}
          </p>
        </div>
      </div>

      <div className="btm-panel">
        <ul className="btm-sliders">
          {labels.map(({ name, description }, index) => (
            <Row
              key={name}
              name={name}
              description={description}
              score={current.scores[name]}
              isTop={name === top}
              noteOpen={openNote === name}
              inputId={`${id}-s${index}`}
              noteId={`${id}-n${index}`}
              actions={actions}
            />
          ))}
        </ul>
        <p className="btm-sum">
          The scores always add up to 1. The model is sure the line is one of these pitches, and splits that certainty
          between them, like one bet spread across the options. More on one leaves less for the rest, so moving one
          slider moves the others.
        </p>
        <p className="btm-source">
          {mock ? (
            "The scores on this page are made up."
          ) : (
            <>
              Scores from <code>{model}</code>, choosing one label.
            </>
          )}
        </p>
      </div>

      <div className="btm-sr" aria-live="polite">
        {announcement}
      </div>
    </div>
  );
}
