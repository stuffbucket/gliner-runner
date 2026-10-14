import { OrderedObject, type OrderedJson } from "../../src/gliner2/index.js";

/** Absolute tolerance on numeric output fields (confidence, probabilities, objective). */
export const NUMERIC_TOLERANCE = 1e-4;

export { parseOrderedJson as parseOrdered } from "../../src/gliner2/ordered-json.js";

export interface NumericStats {
  compared: number;
  /** Values whose raw difference is exactly zero. */
  exactZero: number;
  /** Non-zero raw differences below the tolerance (shown as zero in the synthetic-zero view). */
  syntheticZero: number;
  overTolerance: number;
  maxAbsDiff: number;
}

export function emptyStats(): NumericStats {
  return { compared: 0, exactZero: 0, syntheticZero: 0, overTolerance: 0, maxAbsDiff: 0 };
}

export function mergeStats(into: NumericStats, from: NumericStats): void {
  into.compared += from.compared;
  into.exactZero += from.exactZero;
  into.syntheticZero += from.syntheticZero;
  into.overTolerance += from.overTolerance;
  into.maxAbsDiff = Math.max(into.maxAbsDiff, from.maxAbsDiff);
}

export interface Comparison {
  /** Semantic mismatches; empty when labels, key order, metadata and numeric bounds all agree. */
  mismatches: string[];
  numeric: NumericStats;
}

const NUMERIC_KEYS = new Set(["confidence", "objective"]);

/**
 * Compares a node `output` with the PyTorch golden. Strings, booleans, array lengths, key sequences and
 * non-numeric-field values must be identical; `confidence`, `objective` and `probabilities` values must lie
 * within {@link NUMERIC_TOLERANCE}. Raw differences are always recorded.
 */
export function compareOutputs(actual: OrderedJson | undefined, expected: OrderedJson | undefined): Comparison {
  const mismatches: string[] = [];
  const numeric = emptyStats();
  const walk = (a: OrderedJson | undefined, e: OrderedJson | undefined, path: string, tolerant: boolean): void => {
    if (a instanceof OrderedObject && e instanceof OrderedObject) {
      const ak = a.keys();
      const ek = e.keys();
      if (JSON.stringify(ak) !== JSON.stringify(ek)) {
        mismatches.push(`${path}: key sequence ${JSON.stringify(ak)} != ${JSON.stringify(ek)}`);
        return;
      }
      for (const [key, member] of e.entries) {
        walk(a.get(key), member, `${path}.${key}`, tolerant || key === "probabilities");
      }
      return;
    }
    if (Array.isArray(a) && Array.isArray(e)) {
      if (a.length !== e.length) {
        mismatches.push(`${path}: array length ${a.length} != ${e.length}`);
        return;
      }
      e.forEach((member, i) => walk((a as OrderedJson[])[i], member, `${path}[${i}]`, false));
      return;
    }
    const numericField = tolerant || NUMERIC_KEYS.has(path.slice(path.lastIndexOf(".") + 1));
    if (numericField && typeof a === "number" && typeof e === "number") {
      const diff = Math.abs(a - e);
      numeric.compared += 1;
      numeric.maxAbsDiff = Math.max(numeric.maxAbsDiff, diff);
      if (diff === 0) numeric.exactZero += 1;
      else if (diff < NUMERIC_TOLERANCE) numeric.syntheticZero += 1;
      else {
        numeric.overTolerance += 1;
        mismatches.push(`${path}: ${a} differs from ${e} by ${diff} (> ${NUMERIC_TOLERANCE})`);
      }
      return;
    }
    if (a === e || (typeof a === "number" && typeof e === "number" && a === e)) return;
    mismatches.push(`${path}: ${JSON.stringify(a)} != ${JSON.stringify(e)}`);
  };
  walk(actual, expected, "output", false);
  return { mismatches, numeric };
}
