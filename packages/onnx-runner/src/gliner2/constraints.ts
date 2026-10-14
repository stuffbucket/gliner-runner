/**
 * Constraint AST with Kleene three-valued evaluation (gliner2/classification/constraints.py).
 * `true` = satisfied, `false` = violated, `null` = undetermined.
 */
import { Gliner2MalformedSchemaError, Gliner2SchemaError } from "./errors.js";
import { pyNum, pyRepr, pyReprValue, union } from "./pyparity.js";
import type { TaskLookup, TaskSpec } from "./types.js";

export type Kleene = boolean | null;

/** Assignment protocol (constraints.py:30-97, decoding/base.py:29-70). */
export interface Assignment {
  isDecided(task: string): boolean;
  /** Chosen labels if decided else the labels in EVERY local assignment. */
  selected(task: string): ReadonlySet<string>;
  /** Chosen labels if decided else the labels in SOME local assignment. */
  domain(task: string): ReadonlySet<string>;
  holds(task: string, label: string): Kleene;
  levels(task: string): ReadonlySet<number>;
  index(task: string, label: string): number;
  defaultLabel(task: string): string | null;
}

export interface Constraint {
  readonly type: string;
  evaluate(a: Assignment): Kleene;
  references(): ReadonlySet<string>;
  labelReferences(): ReadonlyArray<readonly [string, string]>;
  setReferences(): ReadonlySet<string>;
  countReferences(): ReadonlySet<string>;
  /** Compile-time validation (`check_schema`); throws {@link Gliner2SchemaError}. */
  checkSchema(schema: TaskLookup): void;
  /** Python dataclass `repr()`. */
  repr(): string;
}

const EMPTY: ReadonlySet<string> = new Set<string>();

const kNot = (v: Kleene): Kleene => (v === null ? null : !v);
function kAnd(vs: readonly Kleene[]): Kleene {
  let undetermined = false;
  for (const v of vs) {
    if (v === false) return false;
    if (v === null) undetermined = true;
  }
  return undetermined ? null : true;
}
function kOr(vs: readonly Kleene[]): Kleene {
  let undetermined = false;
  for (const v of vs) {
    if (v === true) return true;
    if (v === null) undetermined = true;
  }
  return undetermined ? null : false;
}

function taskSpec(schema: TaskLookup, name: string): TaskSpec {
  const t = schema.tasks.get(name);
  if (t === undefined) throw new Gliner2SchemaError(`unknown task ${pyRepr(name)}`);
  return t;
}

/** Assignment used by the compile-time static feasibility check (`DictAssignment`). */
export class DictAssignment implements Assignment {
  private readonly sel: Map<string, Set<string>>;
  private readonly dec: Set<string>;

  constructor(
    private readonly schema: TaskLookup,
    selected: Readonly<Record<string, readonly string[]>>,
    decided: readonly string[],
  ) {
    this.sel = new Map(Object.entries(selected).map(([k, v]) => [k, new Set(v)]));
    this.dec = new Set(decided);
  }

  isDecided(task: string): boolean {
    return this.dec.has(task);
  }
  selected(task: string): ReadonlySet<string> {
    return this.sel.get(task) ?? new Set();
  }
  domain(task: string): ReadonlySet<string> {
    if (this.dec.has(task)) return this.sel.get(task) ?? new Set();
    return new Set(taskSpec(this.schema, task).labelNames);
  }
  holds(task: string, label: string): Kleene {
    if (this.selected(task).has(label)) return true;
    if (this.domain(task).has(label)) return this.isDecided(task) ? false : null;
    return false;
  }
  levels(task: string): ReadonlySet<number> {
    const names = taskSpec(this.schema, task).labelNames;
    return new Set(
      [...this.domain(task)].filter((l) => names.includes(l)).map((l) => names.indexOf(l)),
    );
  }
  index(task: string, label: string): number {
    return taskSpec(this.schema, task).labelNames.indexOf(label);
  }
  defaultLabel(task: string): string | null {
    return taskSpec(this.schema, task).default;
  }
}

const selectable = (a: Assignment, task: string): Set<string> => union(a.selected(task), a.domain(task));

// ---------------------------------------------------------------------------
// Leaf nodes
// ---------------------------------------------------------------------------

class LabelRef implements Constraint {
  readonly type = "LabelRef";
  constructor(
    readonly task: string,
    readonly label: string,
  ) {}
  evaluate(a: Assignment): Kleene {
    return a.holds(this.task, this.label);
  }
  references(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return [[this.task, this.label]];
  }
  setReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  countReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  checkSchema(schema: TaskLookup): void {
    if (!taskSpec(schema, this.task).labelNames.includes(this.label)) {
      throw new Gliner2SchemaError(`${pyRepr(this.label)} is not a label of ${pyRepr(this.task)}`);
    }
  }
  repr(): string {
    return `LabelRef(task=${pyRepr(this.task)}, label=${pyRepr(this.label)})`;
  }
}

abstract class TaskOnly implements Constraint {
  abstract readonly type: string;
  constructor(readonly task: string) {}
  abstract evaluate(a: Assignment): Kleene;
  references(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return [];
  }
  setReferences(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  countReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  checkSchema(schema: TaskLookup): void {
    taskSpec(schema, this.task);
  }
  repr(): string {
    return `${this.type}(task=${pyRepr(this.task)})`;
  }
}

class AnySelected extends TaskOnly {
  readonly type = "AnySelected";
  evaluate(a: Assignment): Kleene {
    return kOr([...selectable(a, this.task)].map((l) => a.holds(this.task, l)));
  }
}

class AnyOtherSelected extends TaskOnly {
  readonly type = "AnyOtherSelected";
  evaluate(a: Assignment): Kleene {
    const d = a.defaultLabel(this.task);
    return kOr(
      [...selectable(a, this.task)].filter((l) => l !== d).map((l) => a.holds(this.task, l)),
    );
  }
  override checkSchema(schema: TaskLookup): void {
    if (taskSpec(schema, this.task).default === null) {
      throw new Gliner2SchemaError(`any_other_selected requires task ${pyRepr(this.task)} to declare a default`);
    }
  }
}

class IsDefault extends TaskOnly {
  readonly type = "IsDefault";
  evaluate(a: Assignment): Kleene {
    const d = a.defaultLabel(this.task);
    return d === null ? false : a.holds(this.task, d);
  }
  override checkSchema(schema: TaskLookup): void {
    if (taskSpec(schema, this.task).default === null) {
      throw new Gliner2SchemaError(`is_default requires task ${pyRepr(this.task)} to declare a default`);
    }
  }
}

class Cardinality implements Constraint {
  readonly type = "Cardinality";
  constructor(
    readonly task: string,
    readonly minimum: number,
    readonly maximum: number | null,
  ) {}
  evaluate(a: Assignment): Kleene {
    const sel = a.selected(this.task);
    const dom = a.domain(this.task);
    const lo = sel.size;
    const hi = union(sel, dom).size;
    const mx = this.maximum !== null ? this.maximum : hi;
    if (lo > mx) return false;
    if (hi < this.minimum) return false;
    if (lo >= this.minimum && hi <= mx) return true;
    return null;
  }
  references(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return [];
  }
  setReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  countReferences(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  checkSchema(schema: TaskLookup): void {
    const n = taskSpec(schema, this.task).labelNames.length;
    if (!(0 <= this.minimum && this.minimum <= n)) {
      throw new Gliner2SchemaError(`cardinality minimum ${this.minimum} outside [0, ${n}] for ${pyRepr(this.task)}`);
    }
    if (this.maximum !== null) {
      if (!(0 <= this.maximum && this.maximum <= n)) {
        throw new Gliner2SchemaError(`cardinality maximum ${this.maximum} outside [0, ${n}] for ${pyRepr(this.task)}`);
      }
      if (this.minimum > this.maximum) {
        throw new Gliner2SchemaError(
          `cardinality minimum ${this.minimum} exceeds maximum ${this.maximum} for ${pyRepr(this.task)}`,
        );
      }
    }
  }
  repr(): string {
    return `Cardinality(task=${pyRepr(this.task)}, minimum=${pyNum(this.minimum)}, maximum=${pyNum(this.maximum)})`;
  }
}

abstract class Ordinal implements Constraint {
  abstract readonly type: "MinLevel" | "MaxLevel" | "AtLevel";
  constructor(
    readonly task: string,
    readonly level: string,
  ) {}
  abstract evaluate(a: Assignment): Kleene;
  references(): ReadonlySet<string> {
    return new Set([this.task]);
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return [];
  }
  setReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  countReferences(): ReadonlySet<string> {
    return EMPTY;
  }
  checkSchema(schema: TaskLookup): void {
    const sp = taskSpec(schema, this.task);
    if (!sp.ordered) {
      throw new Gliner2SchemaError(`ordinal op requires an ordered task; ${pyRepr(this.task)} is unordered`);
    }
    if (!sp.labelNames.includes(this.level)) {
      throw new Gliner2SchemaError(`${pyRepr(this.level)} is not a label of ${pyRepr(this.task)}`);
    }
  }
  repr(): string {
    return `${this.type}(task=${pyRepr(this.task)}, level=${pyRepr(this.level)})`;
  }
}

function minMax(levels: ReadonlySet<number>): [number, number] {
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of levels) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return [mn, mx];
}

class MinLevel extends Ordinal {
  readonly type = "MinLevel";
  evaluate(a: Assignment): Kleene {
    const floor = a.index(this.task, this.level);
    const lv = a.levels(this.task);
    if (lv.size === 0) return false;
    const [mn, mx] = minMax(lv);
    if (mn >= floor) return true;
    if (mx < floor) return false;
    return null;
  }
}
class MaxLevel extends Ordinal {
  readonly type = "MaxLevel";
  evaluate(a: Assignment): Kleene {
    const ceil = a.index(this.task, this.level);
    const lv = a.levels(this.task);
    if (lv.size === 0) return false;
    const [mn, mx] = minMax(lv);
    if (mx <= ceil) return true;
    if (mn > ceil) return false;
    return null;
  }
}
class AtLevel extends Ordinal {
  readonly type = "AtLevel";
  evaluate(a: Assignment): Kleene {
    const target = a.index(this.task, this.level);
    const lv = a.levels(this.task);
    if (lv.size === 0) return false;
    if (lv.size === 1 && lv.has(target)) return true;
    if (!lv.has(target)) return false;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Composite nodes
// ---------------------------------------------------------------------------

class Not implements Constraint {
  readonly type = "Not";
  constructor(readonly child: Constraint) {}
  evaluate(a: Assignment): Kleene {
    return kNot(this.child.evaluate(a));
  }
  references(): ReadonlySet<string> {
    return this.child.references();
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return this.child.labelReferences();
  }
  setReferences(): ReadonlySet<string> {
    return this.child.setReferences();
  }
  countReferences(): ReadonlySet<string> {
    return this.child.countReferences();
  }
  checkSchema(schema: TaskLookup): void {
    this.child.checkSchema(schema);
  }
  repr(): string {
    return `Not(child=${this.child.repr()})`;
  }
}

abstract class Variadic implements Constraint {
  abstract readonly type: "And" | "Or" | "ExactlyOneOf";
  constructor(readonly children: readonly Constraint[]) {}
  abstract evaluate(a: Assignment): Kleene;
  references(): ReadonlySet<string> {
    return union(...this.children.map((c) => c.references()));
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return this.children.flatMap((c) => c.labelReferences());
  }
  setReferences(): ReadonlySet<string> {
    return union(...this.children.map((c) => c.setReferences()));
  }
  countReferences(): ReadonlySet<string> {
    return union(...this.children.map((c) => c.countReferences()));
  }
  checkSchema(schema: TaskLookup): void {
    for (const c of this.children) c.checkSchema(schema);
  }
  repr(): string {
    const inner = this.children.map((c) => c.repr());
    return `${this.type}(children=(${inner.join(", ")}${inner.length === 1 ? "," : ""}))`;
  }
}

class And extends Variadic {
  readonly type = "And";
  evaluate(a: Assignment): Kleene {
    return kAnd(this.children.map((c) => c.evaluate(a)));
  }
}
class Or extends Variadic {
  readonly type = "Or";
  evaluate(a: Assignment): Kleene {
    return kOr(this.children.map((c) => c.evaluate(a)));
  }
}
class ExactlyOneOf extends Variadic {
  readonly type = "ExactlyOneOf";
  evaluate(a: Assignment): Kleene {
    const vals = this.children.map((c) => c.evaluate(a));
    const trues = vals.filter((v) => v === true).length;
    const nones = vals.filter((v) => v === null).length;
    if (trues >= 2) return false;
    if (trues === 1) return nones === 0 ? true : null;
    return nones === 0 ? false : null;
  }
}

abstract class Binary implements Constraint {
  abstract readonly type: "Implies" | "Iff" | "Excludes";
  abstract readonly leftName: string;
  abstract readonly rightName: string;
  constructor(
    readonly first: Constraint,
    readonly second: Constraint,
  ) {}
  abstract evaluate(a: Assignment): Kleene;
  references(): ReadonlySet<string> {
    return union(this.first.references(), this.second.references());
  }
  labelReferences(): ReadonlyArray<readonly [string, string]> {
    return [...this.first.labelReferences(), ...this.second.labelReferences()];
  }
  setReferences(): ReadonlySet<string> {
    return union(this.first.setReferences(), this.second.setReferences());
  }
  countReferences(): ReadonlySet<string> {
    return union(this.first.countReferences(), this.second.countReferences());
  }
  checkSchema(schema: TaskLookup): void {
    this.first.checkSchema(schema);
    this.second.checkSchema(schema);
  }
  repr(): string {
    return `${this.type}(${this.leftName}=${this.first.repr()}, ${this.rightName}=${this.second.repr()})`;
  }
}

class Implies extends Binary {
  readonly type = "Implies";
  readonly leftName = "cond";
  readonly rightName = "then";
  evaluate(a: Assignment): Kleene {
    return kOr([kNot(this.first.evaluate(a)), this.second.evaluate(a)]);
  }
}
class Iff extends Binary {
  readonly type = "Iff";
  readonly leftName = "left";
  readonly rightName = "right";
  evaluate(a: Assignment): Kleene {
    const l = this.first.evaluate(a);
    const r = this.second.evaluate(a);
    return l === null || r === null ? null : l === r;
  }
}
class Excludes extends Binary {
  readonly type = "Excludes";
  readonly leftName = "left";
  readonly rightName = "right";
  evaluate(a: Assignment): Kleene {
    return kNot(kAnd([this.first.evaluate(a), this.second.evaluate(a)]));
  }
}

// ---------------------------------------------------------------------------
// Deserialization (constraint_from_dict)
// ---------------------------------------------------------------------------

type Field = "task" | "label" | "level" | "child" | "children" | "cond" | "then" | "left" | "right" | "minimum" | "maximum";

/** Python signature: accepted keyword names in order, and which are required. */
const SIGNATURES: Readonly<Record<string, { readonly fields: readonly Field[]; readonly required: readonly Field[] }>> = {
  LabelRef: { fields: ["task", "label"], required: ["task", "label"] },
  Not: { fields: ["child"], required: ["child"] },
  And: { fields: ["children"], required: ["children"] },
  Or: { fields: ["children"], required: ["children"] },
  Implies: { fields: ["cond", "then"], required: ["cond", "then"] },
  Iff: { fields: ["left", "right"], required: ["left", "right"] },
  Excludes: { fields: ["left", "right"], required: ["left", "right"] },
  ExactlyOneOf: { fields: ["children"], required: ["children"] },
  Cardinality: { fields: ["task", "minimum", "maximum"], required: ["task"] },
  AtLevel: { fields: ["task", "level"], required: ["task", "level"] },
  MinLevel: { fields: ["task", "level"], required: ["task", "level"] },
  MaxLevel: { fields: ["task", "level"], required: ["task", "level"] },
  IsDefault: { fields: ["task"], required: ["task"] },
  AnySelected: { fields: ["task"], required: ["task"] },
  AnyOtherSelected: { fields: ["task"], required: ["task"] },
};

const isMapping = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

function missingArgumentsMessage(kind: string, missing: readonly string[]): string {
  const quoted = missing.map((m) => `'${m}'`);
  let list: string;
  if (quoted.length === 1) list = quoted[0] as string;
  else if (quoted.length === 2) list = `${quoted[0]} and ${quoted[1]}`;
  else list = `${quoted.slice(0, -1).join(", ")}, and ${quoted[quoted.length - 1]}`;
  return `${kind}.__init__() missing ${missing.length} required positional argument${missing.length === 1 ? "" : "s"}: ${list}`;
}

function malformed(message: string): Gliner2MalformedSchemaError {
  return new Gliner2MalformedSchemaError(message, "TypeError");
}

function rebuild(value: unknown): unknown {
  if (isMapping(value) && "type" in value) return constraintFromDict(value);
  if (Array.isArray(value)) return value.map(rebuild);
  return value;
}

/**
 * `constraint_from_dict`. Mappings that contain `"type"` become constraints, lists become lists of rebuilt values.
 * Unknown kinds raise `Gliner2SchemaError`; missing/extra/ill-typed fields raise `Gliner2MalformedSchemaError`
 * (Python `TypeError`).
 */
export function constraintFromDict(data: unknown): Constraint {
  if (!isMapping(data)) throw malformed(`constraint must be a mapping, got ${pyReprValue(data)}`);
  const { type: kind, ...values } = data;
  const sig = typeof kind === "string" && Object.hasOwn(SIGNATURES, kind) ? SIGNATURES[kind] : undefined;
  if (sig === undefined || typeof kind !== "string") {
    throw new Gliner2SchemaError(`unknown constraint type ${kind === undefined ? "None" : pyReprValue(kind)}`);
  }
  // Null prototype so a `__proto__` key is an ordinary (and then rejected) keyword, as in Python.
  const rebuilt: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const [k, v] of Object.entries(values)) rebuilt[k] = rebuild(v);
  for (const k of Object.keys(rebuilt)) {
    if (!(sig.fields as readonly string[]).includes(k)) {
      throw malformed(`${kind}.__init__() got an unexpected keyword argument '${k}'`);
    }
  }
  const missing = sig.required.filter((f) => !(f in rebuilt));
  if (missing.length > 0) throw malformed(missingArgumentsMessage(kind, missing));

  const str = (field: Field): string => {
    const v = rebuilt[field];
    if (typeof v !== "string") throw malformed(`${kind}.${field} must be a string, got ${pyReprValue(v)}`);
    return v;
  };
  const node = (field: Field): Constraint => {
    const v = rebuilt[field];
    if (!isConstraint(v)) throw malformed(`${kind}.${field} must be a constraint, got ${pyReprValue(v)}`);
    return v;
  };
  const nodes = (field: Field): Constraint[] => {
    const v = rebuilt[field];
    if (!Array.isArray(v) || !v.every(isConstraint)) {
      throw malformed(`${kind}.${field} must be a list of constraints`);
    }
    return v as Constraint[];
  };
  const count = (field: Field, fallback: number | null): number | null => {
    const v = rebuilt[field];
    if (v === undefined || v === null) return fallback;
    if (typeof v !== "number") throw malformed(`${kind}.${field} must be a number, got ${pyReprValue(v)}`);
    return v;
  };

  switch (kind) {
    case "LabelRef":
      return new LabelRef(str("task"), str("label"));
    case "AnySelected":
      return new AnySelected(str("task"));
    case "AnyOtherSelected":
      return new AnyOtherSelected(str("task"));
    case "IsDefault":
      return new IsDefault(str("task"));
    case "Cardinality":
      return new Cardinality(str("task"), count("minimum", 0) as number, count("maximum", null));
    case "MinLevel":
      return new MinLevel(str("task"), str("level"));
    case "MaxLevel":
      return new MaxLevel(str("task"), str("level"));
    case "AtLevel":
      return new AtLevel(str("task"), str("level"));
    case "Not":
      return new Not(node("child"));
    case "And":
      return new And(nodes("children"));
    case "Or":
      return new Or(nodes("children"));
    case "ExactlyOneOf":
      return new ExactlyOneOf(nodes("children"));
    case "Implies":
      return new Implies(node("cond"), node("then"));
    case "Iff":
      return new Iff(node("left"), node("right"));
    case "Excludes":
      return new Excludes(node("left"), node("right"));
    default:
      throw new Gliner2SchemaError(`unknown constraint type ${pyRepr(kind)}`);
  }
}

function isConstraint(v: unknown): v is Constraint {
  return (
    v !== null &&
    typeof v === "object" &&
    typeof (v as { evaluate?: unknown }).evaluate === "function" &&
    typeof (v as { checkSchema?: unknown }).checkSchema === "function"
  );
}

/** Lowered default rule: `Iff(IsDefault(t), Not(AnyOtherSelected(t)))`. */
export function defaultRule(task: string): Constraint {
  return new Iff(new IsDefault(task), new Not(new AnyOtherSelected(task)));
}
