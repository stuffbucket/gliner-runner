/** Public decoding entry points: flat logits -> scores -> problem -> decoder ladder -> result. */
import { decodeLadder, resolveConfig } from "./decoders.js";
import type { DecodeConfig } from "./decoders.js";
import { Gliner2ConfigError } from "./errors.js";
import { buildProblem } from "./problem.js";
import { buildResult, violationString } from "./result.js";
import type { Gliner2Result } from "./result.js";
import type { Scores } from "./scoring.js";
import type { CompiledSchema } from "./types.js";

/**
 * Splits one row of flat logits (`[task0 labels..., task1 labels..., ...]` in declaration order) into per-task maps.
 * The row must already be sliced to its real label count (`Gliner2Batch.labelCounts[row]`), never include padding.
 */
export function scoresFromFlatLogits(compiled: CompiledSchema, rowLogits: ArrayLike<number>): Scores {
  if (rowLogits.length !== compiled.labelCount) {
    throw new Gliner2ConfigError(
      `logit/label count mismatch: got ${rowLogits.length} logits for ${compiled.labelCount} labels`,
    );
  }
  const scores = new Map<string, Map<string, number>>();
  let k = 0;
  for (const spec of compiled.specs) {
    const m = new Map<string, number>();
    for (const label of spec.labelNames) {
      const v = Number(rowLogits[k++]);
      if (!Number.isFinite(v)) throw new Gliner2ConfigError("logits must be finite");
      m.set(label, v);
    }
    scores.set(spec.name, m);
  }
  return scores;
}

/** Decodes task -> label -> logit scores (declaration order) into a result. */
export function decodeScores(compiled: CompiledSchema, scores: Scores, config: DecodeConfig): Gliner2Result {
  const resolved = resolveConfig(config);
  const problem = buildProblem(compiled, scores, resolved);
  const widen = () => buildProblem(compiled, scores, resolved, problem.taskOrder);
  const solution = decodeLadder(problem, resolved, widen, violationString);
  return buildResult(compiled, scores, solution, problem.taskOrder, resolved.includeConfidence);
}

/**
 * Decodes one row of ONNX logits (float32 widened to double) for the compiled schema.
 * `result.output` is the order-preserving equivalent of Python `ClassificationResult.to_dict(include_confidence)`;
 * `result.serialize()` produces its JSON text. Throws `Gliner2InfeasibleError` only with `onInfeasible: "raise"`.
 */
export function decode(
  compiled: CompiledSchema,
  rowLogits: Float32Array | readonly number[],
  config: DecodeConfig,
): Gliner2Result {
  return decodeScores(compiled, scoresFromFlatLogits(compiled, rowLogits), config);
}
