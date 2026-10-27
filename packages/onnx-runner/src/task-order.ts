import { OrderedObject, parseOrderedJson } from "./gliner2/ordered-json.js";

/**
 * JSON object member order of `schema.tasks`. Task order drives the prompt, logit order and output key order,
 * but `JSON.parse` stores integer-like names ("1", "10") in ascending order ahead of every other name, which
 * differs from the declaration order Python preserves. The HTTP layer records the declared order here, keyed by the
 * parsed `tasks` object, and consumers read it through {@link taskOrderOf}.
 */
const declared = new WeakMap<object, readonly string[]>();

export function taskOrderOf(tasks: object): readonly string[] {
  return declared.get(tasks) ?? Object.keys(tasks);
}

export function orderedTaskEntries<T>(tasks: Readonly<Record<string, T>>): Array<readonly [string, T]> {
  return taskOrderOf(tasks).map((name) => [name, tasks[name] as T] as const);
}

/** Records the declared order of a `tasks` object the runner built itself (see `shapes.ts`). */
export function declareTaskOrder(tasks: object, order: readonly string[]): void {
  const actual = Object.keys(tasks);
  if (order.length === actual.length && order.every((key) => Object.hasOwn(tasks, key))) {
    declared.set(tasks, [...order]);
  }
}

interface WithTasks {
  schema: { kind: string } & { tasks?: object };
}

/**
 * Records the declared `tasks` order of parsed requests. `requests` must be in the order they appear in `rawBody`
 * (the body itself when `batched` is false, otherwise its `requests` array). The order always comes from an
 * order-preserving parse of the raw text, so escaped names ("\u0031"), `__proto__` and repeated names behave like
 * Python's `json`: a repeated name keeps its first position (and its last value, which the parsed request holds).
 */
export function recordTaskOrder(rawBody: string, requests: readonly WithTasks[], batched: boolean): void {
  let root;
  try {
    root = parseOrderedJson(rawBody);
  } catch {
    return;
  }
  const rawRequests: unknown = batched && root instanceof OrderedObject ? root.get("requests") : [root];
  if (!Array.isArray(rawRequests)) {
    return;
  }
  requests.forEach((request, index) => {
    const raw = rawRequests[index];
    const tasks = request.schema.tasks;
    const rawTasks = raw instanceof OrderedObject ? raw.get("schema") : undefined;
    const order = rawTasks instanceof OrderedObject ? rawTasks.get("tasks") : undefined;
    if (tasks === undefined || !(order instanceof OrderedObject)) {
      return;
    }
    const keys = order.keys();
    const actual = Object.keys(tasks);
    // Both sides list the same names (the parsed record keeps every key); anything else is not the same object.
    if (keys.length === actual.length && keys.every((key) => Object.hasOwn(tasks, key))) {
      declared.set(tasks, keys);
    }
  });
}
