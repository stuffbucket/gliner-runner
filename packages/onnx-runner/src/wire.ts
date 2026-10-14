import { OrderedObject } from "./gliner2/ordered-json.js";

/**
 * `JSON.stringify` that also understands {@link OrderedObject}: ordered objects keep their insertion order even
 * when keys are integer-like (`"1"`, `"10"`), which a plain JS object would reorder ascending. Everything else
 * follows `JSON.stringify` semantics (`toJSON`, dropped `undefined` members, non-finite numbers become `null`).
 */
export function stringifyWire(value: unknown): string {
  return write(value) ?? "null";
}

function write(value: unknown): string | undefined {
  if (value instanceof OrderedObject) {
    const members: string[] = [];
    for (const [key, member] of value.entries) {
      const text = write(member);
      if (text !== undefined) {
        members.push(`${JSON.stringify(key)}:${text}`);
      }
    }
    return `{${members.join(",")}}`;
  }
  if (typeof value === "object" && value !== null) {
    const candidate = value as { toJSON?: unknown };
    if (typeof candidate.toJSON === "function") {
      return write((candidate.toJSON as () => unknown)());
    }
    if (Array.isArray(value)) {
      return `[${value.map((item) => write(item) ?? "null").join(",")}]`;
    }
    const members: string[] = [];
    for (const [key, member] of Object.entries(value)) {
      const text = write(member);
      if (text !== undefined) {
        members.push(`${JSON.stringify(key)}:${text}`);
      }
    }
    return `{${members.join(",")}}`;
  }
  return JSON.stringify(value);
}
