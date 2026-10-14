/**
 * Error classes of the GLiNER2 classification port.
 *
 * Python mapping (see dist/port-spec/gliner2-classification.md section 2.2):
 *  - `SchemaError` (ValueError)           -> {@link Gliner2SchemaError}; the PyTorch runner maps it to InvalidSchemaError.
 *  - `TypeError` / `KeyError` raised for malformed dicts (unknown task key, missing `labels`, unknown or missing
 *    constraint fields) escape the PyTorch runner unmapped; the port reports them as
 *    {@link Gliner2MalformedSchemaError}, a subclass of {@link Gliner2SchemaError}, so callers can map every
 *    schema-shaped failure to the same user-facing InvalidSchemaError with `instanceof Gliner2SchemaError`
 *    (or {@link isGliner2SchemaError}).
 *  - `InfeasibleError`                    -> {@link Gliner2InfeasibleError} (only with `onInfeasible: "raise"`, never
 *    produced by the runner configuration).
 *  - `ValueError` from `ClassificationConfig` validation -> {@link Gliner2ConfigError} (a programming error, not a
 *    user schema error).
 */

/** Python `SchemaError`: invalid schema content. `message` is byte-identical to the Python message. */
export class Gliner2SchemaError extends Error {
  /** Python exception class this error corresponds to. */
  readonly pythonClass: "SchemaError" | "TypeError" | "KeyError";

  constructor(message: string, pythonClass: "SchemaError" | "TypeError" | "KeyError" = "SchemaError") {
    super(message);
    this.name = "Gliner2SchemaError";
    this.pythonClass = pythonClass;
  }
}

/**
 * Python `TypeError`/`KeyError` raised while turning a malformed schema dict into a schema (unknown/missing keys,
 * wrongly typed constraint fields). Still a schema problem from the caller's perspective.
 */
export class Gliner2MalformedSchemaError extends Gliner2SchemaError {
  constructor(message: string, pythonClass: "TypeError" | "KeyError" = "TypeError") {
    super(message, pythonClass);
    this.name = "Gliner2MalformedSchemaError";
  }
}

/** Python `InfeasibleError`: no assignment satisfies the constraints (`onInfeasible: "raise"` only). */
export class Gliner2InfeasibleError extends Error {
  readonly violations: readonly string[];

  constructor(message: string, violations: readonly string[] = []) {
    super(message);
    this.name = "Gliner2InfeasibleError";
    this.violations = violations;
  }
}

/** Invalid decode configuration (Python `ValueError` from `ClassificationConfig`), or invalid logits shape. */
export class Gliner2ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "Gliner2ConfigError";
  }
}

/** True for every error that the runner must report as InvalidSchemaError. */
export function isGliner2SchemaError(error: unknown): error is Gliner2SchemaError {
  return error instanceof Gliner2SchemaError;
}

/** Internal: node budget of the exact / min-violations search was exhausted. */
export class BudgetExceeded extends Error {
  constructor() {
    super("node budget exceeded");
    this.name = "BudgetExceeded";
  }
}
