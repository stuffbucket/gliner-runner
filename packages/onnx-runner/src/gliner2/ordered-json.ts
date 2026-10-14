/**
 * Order-preserving JSON values. JS plain objects reorder integer-like keys ("1", "10") ascending, which would break
 * byte-level key-order parity with Python's `to_dict()`; result objects are therefore kept as ordered pairs and are
 * serialized by {@link serializeOrdered}.
 */

export type OrderedJson = string | number | boolean | null | readonly OrderedJson[] | OrderedObject;

/** An insertion-ordered JSON object (duplicate keys are not allowed; see {@link parseOrderedJson}). */
export class OrderedObject {
  readonly entries: ReadonlyArray<readonly [string, OrderedJson]>;
  constructor(entries: ReadonlyArray<readonly [string, OrderedJson]>) {
    this.entries = entries;
  }

  get(key: string): OrderedJson | undefined {
    return this.entries.find(([k]) => k === key)?.[1];
  }

  keys(): string[] {
    return this.entries.map(([k]) => k);
  }

  /** JSON text with keys in insertion order (see {@link serializeOrdered}). */
  serialize(): string {
    return serializeOrdered(this);
  }

  /**
   * Plain-object view for `JSON.stringify` of an enclosing envelope. Throws when the object (or a nested one) has
   * integer-like keys, because a plain object cannot preserve their order; use {@link serializeOrdered} then.
   */
  toJSON(): Record<string, unknown> {
    if (!canUsePlainObject(this)) {
      throw new RangeError("ordered object has integer-like keys; serialize it with serializeOrdered()");
    }
    return toPlain(this) as Record<string, unknown>;
  }
}

/** True for property names that JS orders numerically ahead of other keys (canonical array indices). */
export function isIntegerLikeKey(key: string): boolean {
  return /^(?:0|[1-9][0-9]*)$/.test(key) && Number(key) < 4294967295;
}

/** True when converting to plain objects keeps the key order (no integer-like keys anywhere). */
export function canUsePlainObject(value: OrderedJson): boolean {
  if (value instanceof OrderedObject) {
    return value.entries.every(([k, v]) => !isIntegerLikeKey(k) && canUsePlainObject(v));
  }
  if (Array.isArray(value)) return (value as readonly OrderedJson[]).every(canUsePlainObject);
  return true;
}

/** Plain JS value (objects, arrays, scalars). Key order is only faithful when {@link canUsePlainObject} holds. */
export function toPlain(value: OrderedJson): unknown {
  if (value instanceof OrderedObject) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of value.entries) out[k] = toPlain(v);
    return out;
  }
  if (Array.isArray(value)) return (value as readonly OrderedJson[]).map(toPlain);
  return value;
}

/** Inverse of {@link toPlain} for tests: plain value -> ordered value (key order = JS enumeration order). */
export function fromPlain(value: unknown): OrderedJson {
  if (Array.isArray(value)) return value.map(fromPlain);
  if (value !== null && typeof value === "object") {
    return new OrderedObject(Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, fromPlain(v)] as const));
  }
  return value as OrderedJson;
}

/** Compact JSON text; object keys keep insertion order, scalars follow `JSON.stringify` (non-finite numbers throw). */
export function serializeOrdered(value: OrderedJson): string {
  if (value instanceof OrderedObject) {
    return "{" + value.entries.map(([k, v]) => `${JSON.stringify(k)}:${serializeOrdered(v)}`).join(",") + "}";
  }
  if (Array.isArray(value)) return "[" + (value as readonly OrderedJson[]).map(serializeOrdered).join(",") + "]";
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new RangeError("cannot serialize a non-finite number as JSON");
  }
  return JSON.stringify(value);
}

const LITERAL = /true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y;

/**
 * Parses JSON text keeping object key order (also for integer-like keys and `__proto__`). A repeated key keeps the
 * position of its first occurrence and the value of its last, which is what Python's `json` module produces.
 */
export function parseOrderedJson(text: string): OrderedJson {
  let index = 0;
  const fail = (message: string): never => {
    throw new SyntaxError(`${message} at offset ${index}`);
  };
  const skip = (): void => {
    while (index < text.length && " \t\r\n".includes(text[index] as string)) index += 1;
  };
  const string = (): string => {
    const start = index;
    index += 1;
    while (index < text.length && text[index] !== '"') index += text[index] === "\\" ? 2 : 1;
    if (text[index] !== '"') fail("unterminated string");
    index += 1;
    return JSON.parse(text.slice(start, index)) as string;
  };
  const value = (): OrderedJson => {
    skip();
    const char = text[index];
    if (char === "{") {
      index += 1;
      const entries: Array<readonly [string, OrderedJson]> = [];
      const positions = new Map<string, number>();
      skip();
      if (text[index] === "}") {
        index += 1;
        return new OrderedObject(entries);
      }
      for (;;) {
        skip();
        if (text[index] !== '"') fail("expected a key");
        const key = string();
        skip();
        if (text[index] !== ":") fail("expected ':'");
        index += 1;
        const member = value();
        const existing = positions.get(key);
        if (existing === undefined) {
          positions.set(key, entries.length);
          entries.push([key, member]);
        } else entries[existing] = [key, member];
        skip();
        if (text[index] === ",") {
          index += 1;
        } else if (text[index] === "}") {
          index += 1;
          return new OrderedObject(entries);
        } else fail("expected ',' or '}'");
      }
    }
    if (char === "[") {
      index += 1;
      const items: OrderedJson[] = [];
      skip();
      if (text[index] === "]") {
        index += 1;
        return items;
      }
      for (;;) {
        items.push(value());
        skip();
        if (text[index] === ",") {
          index += 1;
        } else if (text[index] === "]") {
          index += 1;
          return items;
        } else fail("expected ',' or ']'");
      }
    }
    if (char === '"') return string();
    LITERAL.lastIndex = index;
    const literal = LITERAL.exec(text);
    if (literal === null) return fail("unexpected token");
    index += literal[0].length;
    return JSON.parse(literal[0]) as OrderedJson;
  };
  const result = value();
  skip();
  if (index !== text.length) fail("trailing content");
  return result;
}
