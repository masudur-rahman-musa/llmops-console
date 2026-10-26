import type { Params } from "@/lib/paging";

/**
 * The filters the Requests and Daily cost forms keep in the page address. Each reader
 * takes the address params and drops any value console-api's own checks would refuse
 * (the same tenant and model rules), so a hand-typed bad link shows the list unfiltered
 * instead of an error page. Pinned by tests/filters.test.tsx.
 */

const MAX_TENANT = 64;
const MAX_MODEL = 128;
const TENANT = /^[a-z0-9]+(?:[-_][a-z0-9]+)*$/;
const MODEL = /^[A-Za-z0-9][A-Za-z0-9._:/-]*$/;
const DAY = /^(\d{4})-(\d{2})-(\d{2})$/;

export type RequestFilters = {
  day?: string;
  tenant?: string;
  model?: string;
  errors: boolean;
  slow: boolean;
  cached: boolean;
};

export type CostFilters = { from?: string; to?: string; tenant?: string };

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

/** A real calendar day as YYYY-MM-DD, else undefined. */
export function checkedDay(value: string | undefined): string | undefined {
  const match = value === undefined ? null : DAY.exec(value);
  if (value === undefined || match === null) {
    return undefined;
  }
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return date.toISOString().slice(0, 10) === value ? value : undefined;
}

function checkedTenant(value: string | undefined): string | undefined {
  return value !== undefined && value.length <= MAX_TENANT && TENANT.test(value) ? value : undefined;
}

function checkedModel(value: string | undefined): string | undefined {
  return value !== undefined && value.length <= MAX_MODEL && MODEL.test(value) ? value : undefined;
}

export function readRequestFilters(params: Params): RequestFilters {
  return {
    day: checkedDay(first(params.day)),
    tenant: checkedTenant(first(params.tenant)),
    model: checkedModel(first(params.model)),
    errors: first(params.errors) === "true",
    slow: first(params.slow) === "true",
    cached: first(params.cached) === "true",
  };
}

/** A `from` after `to` is swapped. */
export function readCostFilters(params: Params): CostFilters {
  const a = checkedDay(first(params.from));
  const b = checkedDay(first(params.to));
  const [from, to] = a !== undefined && b !== undefined && a > b ? [b, a] : [a, b];
  return { from, to, tenant: checkedTenant(first(params.tenant)) };
}
