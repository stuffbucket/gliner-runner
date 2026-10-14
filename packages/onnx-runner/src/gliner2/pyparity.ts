/** Python-parity helpers: `sum()` over floats, `str` ordering, `repr()`, `str.strip()` semantics. */

/**
 * CPython 3.12 builtin `sum()` over floats (start=0): Neumaier compensated summation.
 * Returns 0 for an empty list, like Python's int 0.
 */
export function pySum(values: readonly number[]): number {
  const first = values[0];
  if (first === undefined) return 0;
  let f = 0 + first;
  let c = 0.0;
  for (let i = 1; i < values.length; i++) {
    const x = values[i] as number;
    const t = f + x;
    if (Math.abs(f) >= Math.abs(x)) c += f - t + x;
    else c += x - t + f;
    f = t;
  }
  if (c !== 0 && Number.isFinite(c)) f += c;
  return f;
}

/** Maps a UTF-16 code unit so that numeric order equals code point order of the encoded strings. */
function fixUnit(u: number): number {
  if (u >= 0xe000) return u - 0x800;
  if (u >= 0xd800) return u + 0x2000;
  return u;
}

/** Python `str` ordering: by Unicode code point (JS `<` compares UTF-16 code units). */
export function cmpStr(a: string, b: string): number {
  if (a === b) return 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a.charCodeAt(i);
    const y = b.charCodeAt(i);
    if (x !== y) return fixUnit(x) < fixUnit(y) ? -1 : 1;
  }
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

/** `Array.from(iter).sort(cmpStr)` */
export function sortedStrs(values: Iterable<string>): string[] {
  return Array.from(values).sort(cmpStr);
}

// Approximation of str.isprintable() for non-ASCII: everything except Cc, Cf, Cs, Co, Cn, Zl, Zp, Zs.
const NON_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/** Python `repr()` of a `str`. */
export function pyRepr(s: string): string {
  const hasSingle = s.includes("'");
  const hasDouble = s.includes('"');
  const quote = hasSingle && !hasDouble ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const cp = ch.codePointAt(0) as number;
    if (ch === quote || ch === "\\") out += "\\" + ch;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (cp < 0x20 || cp === 0x7f) out += "\\x" + cp.toString(16).padStart(2, "0");
    else if (cp >= 0x80 && NON_PRINTABLE.test(ch)) {
      out +=
        cp <= 0xff
          ? "\\x" + cp.toString(16).padStart(2, "0")
          : cp <= 0xffff
            ? "\\u" + cp.toString(16).padStart(4, "0")
            : "\\U" + cp.toString(16).padStart(8, "0");
    } else out += ch;
  }
  return out + quote;
}

/** Python `repr()` of a JSON value (used only for error messages). */
export function pyReprValue(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "string") return pyRepr(value);
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return "[" + value.map(pyReprValue).join(", ") + "]";
  if (typeof value === "object") {
    return (
      "{" +
      Object.entries(value as Record<string, unknown>)
        .map(([k, v]) => `${pyRepr(k)}: ${pyReprValue(v)}`)
        .join(", ") +
      "}"
    );
  }
  return String(value);
}

/** Python `repr()` of an int/float/None as used inside dataclass reprs. */
export function pyNum(value: number | null | undefined): string {
  return value === null || value === undefined ? "None" : String(value);
}

// Python str.strip()/isspace() treat \x1c-\x1f,\x85 as whitespace and not ﻿; JS trim() is the opposite.
const PY_WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_BLANK = new RegExp(`^[${PY_WS}]*$`);
/** True if Python `not s.strip()`. */
export function pyIsBlank(s: string): boolean {
  return PY_BLANK.test(s);
}

/** Python whitespace class (inner part of a regex character class), also used by the word splitter. */
export const PY_WS_CLASS = PY_WS;

/** Ordered union of sets (insertion order of first appearance). */
export function union<T>(...sets: ReadonlyArray<Iterable<T>>): Set<T> {
  const out = new Set<T>();
  for (const s of sets) for (const x of s) out.add(x);
  return out;
}
