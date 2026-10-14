import type { z } from "zod";

import { unescapeRecordKey } from "./contracts.js";

/** One element of a FastAPI-compatible `422` `detail` array. */
export interface ValidationErrorItem {
  type: string;
  loc: (string | number)[];
  msg: string;
  input: unknown;
  ctx?: Record<string, unknown>;
}

export type ParseResult<T> =
  | { success: true; data: T }
  | { success: false; errors: ValidationErrorItem[] };

type Root = "body" | "path";
type Issue = z.core.$ZodIssue;

const SCHEMA_KINDS = new Set(["classification", "entities", "relations", "structured"]);
const DICT_FIELDS = new Set(["tasks", "label_descriptions", "constraints"]);

export function parseJsonBody<T>(schema: z.ZodType<T>, raw: string): ParseResult<T> {
  if (raw.trim().length === 0) {
    return failure([
      { type: "missing", loc: ["body"], msg: "Field required", input: null },
    ]);
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch (error) {
    const message = error instanceof Error ? error.message : "invalid JSON";
    const offset = /position (\d+)/.exec(message)?.[1];
    return failure([
      {
        type: "json_invalid",
        loc: ["body", offset === undefined ? 0 : Number(offset)],
        msg: "JSON decode error",
        input: {},
        ctx: { error: message },
      },
    ]);
  }
  return parseValue(schema, value, "body");
}

export function parseValue<T>(schema: z.ZodType<T>, value: unknown, root: Root): ParseResult<T> {
  const result = schema.safeParse(value);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return failure(result.error.issues.flatMap((issue) => convertIssue(issue, value, root)));
}

function failure<T>(errors: ValidationErrorItem[]): ParseResult<T> {
  return { success: false, errors };
}

interface Located {
  loc: (string | number)[];
  found: boolean;
  value: unknown;
  parent: unknown;
}

/** Resolves a zod path against the raw input, inserting pydantic's union tag for schemas. */
function locate(path: readonly PropertyKey[], input: unknown, root: Root): Located {
  const loc: (string | number)[] = [root];
  let value: unknown = input;
  let parent: unknown = undefined;
  let found = true;
  path.forEach((segment, index) => {
    // Record keys carry an escape for `__proto__` (see contracts.ts); report the key as sent.
    const key =
      typeof segment === "symbol" ? String(segment) : typeof segment === "string" ? unescapeRecordKey(segment) : segment;
    loc.push(key);
    parent = value;
    const container = value as Record<string | number, unknown> | null | undefined;
    if (found && typeof container === "object" && container !== null && key in container) {
      value = container[key];
    } else {
      found = false;
      value = undefined;
    }
    const previous = path[index - 1];
    const atRequestRoot = index === 0 || (index === 2 && path[0] === "requests");
    const kind = (value as { kind?: unknown } | null | undefined)?.kind;
    if (
      key === "schema" &&
      (atRequestRoot || typeof previous === "number") &&
      index < path.length - 1 &&
      typeof kind === "string" &&
      SCHEMA_KINDS.has(kind)
    ) {
      loc.push(kind);
    }
  });
  return { loc, found, value, parent };
}

function convertIssue(issue: Issue, input: unknown, root: Root): ValidationErrorItem[] {
  if (issue.code === "unrecognized_keys") {
    const base = locate(issue.path, input, root);
    return issue.keys.map((key) => ({
      type: "extra_forbidden",
      loc: [...base.loc, key],
      msg: "Extra inputs are not permitted",
      input: (base.value as Record<string, unknown>)[key],
    }));
  }
  if (issue.code === "invalid_union" && issue.path.at(-1) === "kind") {
    const where = locate(issue.path.slice(0, -1), input, root);
    const expected = (issue as { options?: readonly string[] }).options ?? [...SCHEMA_KINDS];
    const quoted = expected.map((tag) => `'${tag}'`).join(", ");
    const tag = (where.value as { kind?: unknown } | null | undefined)?.kind;
    if (tag === undefined) {
      return [
        {
          type: "union_tag_not_found",
          loc: where.loc,
          msg: "Unable to extract tag using discriminator 'kind'",
          input: where.value,
          ctx: { discriminator: "'kind'" },
        },
      ];
    }
    return [
      {
        type: "union_tag_invalid",
        loc: where.loc,
        msg: `Input tag '${String(tag)}' found using 'kind' does not match any of the expected tags: ${quoted}`,
        input: where.value,
        ctx: { discriminator: "'kind'", tag, expected_tags: quoted },
      },
    ];
  }
  const where = locate(issue.path, input, root);
  const item = describe(issue, where);
  return [{ ...item, loc: where.loc, input: item.input === undefined ? where.value : item.input }];
}

type Described = Omit<ValidationErrorItem, "loc" | "input"> & { input?: unknown };

function describe(issue: Issue, where: Located): Described {
  if (issue.code === "invalid_type" && !where.found) {
    return { type: "missing", msg: "Field required", input: where.parent };
  }
  switch (issue.code) {
    case "invalid_type":
      return describeInvalidType(issue, where);
    case "too_small":
      return describeBound(issue, where, "min");
    case "too_big":
      return describeBound(issue, where, "max");
    case "invalid_value":
      return describeInvalidValue(issue, where);
    case "invalid_format":
      return issue.format === "guid" || issue.format === "uuid"
        ? { type: "uuid_parsing", msg: "Input should be a valid UUID" }
        : {
            type: "string_pattern_mismatch",
            msg: `String should match pattern '${String((issue as { pattern?: unknown }).pattern ?? "")}'`,
          };
    default:
      return {
        type: "value_error",
        msg: `Value error, ${issue.message}`,
        ctx: { error: {} },
      };
  }
}

function describeInvalidType(
  issue: Extract<Issue, { code: "invalid_type" }>,
  where: Located,
): Described {
  switch (issue.expected) {
    case "string":
      return { type: "string_type", msg: "Input should be a valid string" };
    case "number":
      return { type: "float_type", msg: "Input should be a valid number" };
    case "int":
      return typeof where.value === "number"
        ? {
            type: "int_from_float",
            msg: "Input should be a valid integer, got a number with a fractional part",
          }
        : { type: "int_type", msg: "Input should be a valid integer" };
    case "boolean":
      return { type: "bool_type", msg: "Input should be a valid boolean" };
    case "array":
      return { type: "list_type", msg: "Input should be a valid list" };
    case "record":
      return { type: "dict_type", msg: "Input should be a valid dictionary" };
    default: {
      const last = where.loc.at(-1);
      const isDict =
        (typeof last === "string" && DICT_FIELDS.has(last)) ||
        (typeof last === "number" && where.loc.at(-2) === "constraints");
      return isDict
        ? { type: "dict_type", msg: "Input should be a valid dictionary" }
        : {
            type: "model_attributes_type",
            msg: "Input should be a valid dictionary or object to extract fields from",
          };
    }
  }
}

function describeBound(
  issue: Extract<Issue, { code: "too_small" | "too_big" }>,
  where: Located,
  side: "min" | "max",
): Described {
  const limit = Number(side === "min" ? (issue as { minimum: unknown }).minimum : (issue as { maximum: unknown }).maximum);
  const origin = String(issue.origin);
  if (origin === "string") {
    const noun = limit === 1 ? "character" : "characters";
    return side === "min"
      ? {
          type: "string_too_short",
          msg: `String should have at least ${limit} ${noun}`,
          ctx: { min_length: limit },
        }
      : {
          type: "string_too_long",
          msg: `String should have at most ${limit} ${noun}`,
          ctx: { max_length: limit },
        };
  }
  if (origin === "array" || origin === "record" || origin === "set") {
    // Every schema array is a pydantic tuple except the batch `requests` list.
    const fieldType =
      origin !== "array" ? "Dictionary" : where.loc.at(-1) === "requests" ? "List" : "Tuple";
    const size = lengthOf(where.value);
    const noun = limit === 1 ? "item" : "items";
    return side === "min"
      ? {
          type: "too_short",
          msg: `${fieldType} should have at least ${limit} ${noun} after validation, not ${size}`,
          ctx: { field_type: fieldType, min_length: limit, actual_length: size },
        }
      : {
          type: "too_long",
          msg: `${fieldType} should have at most ${limit} ${noun} after validation, not ${size}`,
          ctx: { field_type: fieldType, max_length: limit, actual_length: size },
        };
  }
  const inclusive = issue.inclusive !== false;
  if (side === "min") {
    return inclusive
      ? {
          type: "greater_than_equal",
          msg: `Input should be greater than or equal to ${limit}`,
          ctx: { ge: limit },
        }
      : { type: "greater_than", msg: `Input should be greater than ${limit}`, ctx: { gt: limit } };
  }
  return inclusive
    ? {
        type: "less_than_equal",
        msg: `Input should be less than or equal to ${limit}`,
        ctx: { le: limit },
      }
    : { type: "less_than", msg: `Input should be less than ${limit}`, ctx: { lt: limit } };
}

function describeInvalidValue(
  issue: Extract<Issue, { code: "invalid_value" }>,
  where: Located,
): Described {
  const values = issue.values.map((value) => `'${String(value)}'`);
  const expected =
    values.length > 1 ? `${values.slice(0, -1).join(", ")} or ${values.at(-1)}` : (values[0] ?? "");
  return {
    type: where.loc.at(-1) === "activation" ? "literal_error" : "enum",
    msg: `Input should be ${expected}`,
    ctx: { expected },
  };
}

function lengthOf(value: unknown): number {
  if (Array.isArray(value)) {
    return value.length;
  }
  return typeof value === "object" && value !== null ? Object.keys(value).length : 0;
}
