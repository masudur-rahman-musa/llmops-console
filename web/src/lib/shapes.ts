/**
 * Shared check helpers for the per-page guards, plus the source-status types
 * every page body carries. The guards live in features/<page>/guards.ts and
 * are assembled into GUARDS by console-api.ts. Pinned by tests/guards.test.ts.
 */

export type SourceState = "ok" | "unconfigured" | "error";
export type Sources = { postgres: SourceState; prometheus: SourceState };

export type Check = (value: unknown) => boolean;

const SOURCE_STATES: ReadonlyArray<unknown> = ["ok", "unconfigured", "error"];

export const str: Check = (value) => typeof value === "string";
export const num: Check = (value) => typeof value === "number" && Number.isFinite(value);
export const int: Check = (value) => Number.isInteger(value);
export const bool: Check = (value) => typeof value === "boolean";

export function nullable(check: Check): Check {
  return (value) => value === null || check(value);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Every named field present and passing; extra fields are allowed. */
export function shape(fields: Record<string, Check>): Check {
  return (value) =>
    isRecord(value) && Object.entries(fields).every(([name, check]) => name in value && check(value[name]));
}

export function listOf(check: Check): Check {
  return (value) => Array.isArray(value) && value.every(check);
}

export const pair: Check = (value) => Array.isArray(value) && value.length === 2 && num(value[0]) && num(value[1]);

export const sources = shape({
  postgres: (value) => SOURCE_STATES.includes(value),
  prometheus: (value) => SOURCE_STATES.includes(value),
});
