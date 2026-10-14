/**
 * GLiNER2 constrained-classification port (schema, preprocessing, decoding). Pure: no HTTP, no ONNX Runtime.
 * Reference spec: dist/port-spec/gliner2-classification.md.
 */
export {
  BudgetExceeded as Gliner2BudgetExceeded,
  Gliner2ConfigError,
  Gliner2InfeasibleError,
  Gliner2MalformedSchemaError,
  Gliner2SchemaError,
  isGliner2SchemaError,
} from "./errors.js";
export { cmpStr, pyRepr, pySum } from "./pyparity.js";
export type { Assignment, Constraint, Kleene } from "./constraints.js";
export { constraintFromDict } from "./constraints.js";
export { RESERVED_TOKENS, makeTaskSpec, schemaFromDict } from "./schema.js";
export type { LabelsInput, SchemaInput, TasksInput } from "./schema.js";
export { compileSchema } from "./compile.js";
export type {
  Activation,
  ClassificationSchema as Gliner2Schema,
  CompiledSchema,
  LabelSpec,
  ModelTaskEntry,
  TaskSpec,
} from "./types.js";
export { compileRunnerSchema, oracleTask, runnerSchemaToGliner2 } from "./runner-schema.js";
export type { RunnerClassificationSchema, RunnerClassificationTask } from "./runner-schema.js";
export {
  Gliner2Preprocessor,
  Gliner2TokenizerError,
  SPECIAL_TOKENS,
  collateRows,
  normalizeText,
  splitWords,
} from "./preprocess.js";
export type {
  Gliner2Batch,
  Gliner2PreprocessorOptions,
  Gliner2Row,
  Gliner2Tokenizer,
  SpecialToken,
  SpecialTokenIds,
} from "./preprocess.js";
export { decode, decodeScores, scoresFromFlatLogits } from "./decode.js";
export type { DecodeConfig, DecoderName, DecoderRequest, InfeasibleMode, Solution } from "./decoders.js";
export { Gliner2Result, buildResult, violationString } from "./result.js";
export type { TaskOutcome } from "./result.js";
export {
  OrderedObject,
  canUsePlainObject,
  fromPlain,
  isIntegerLikeKey,
  parseOrderedJson,
  serializeOrdered,
  toPlain,
} from "./ordered-json.js";
export type { OrderedJson } from "./ordered-json.js";
export type { Scores, TaskLogits } from "./scoring.js";
