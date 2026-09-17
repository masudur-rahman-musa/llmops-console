import type { PageName } from "@/lib/console-api";

/**
 * One hand-written guard per console-api body, checking only the fields a page
 * renders (decision C: no schema dependency). Pinned by tests/guards.test.ts.
 */

export type SourceState = "ok" | "unconfigured" | "error";
export type Sources = { postgres: SourceState; prometheus: SourceState };

export type HomeBody = {
  sources: Sources;
  requests_today: number | null;
  cost_usd_today: number | null;
  p95_seconds: number | null;
  error_rate: number | null;
  residency_violations_today: number | null;
  last_eval: { finished_at: string; passed: boolean; decision_accuracy: number | null } | null;
  open_incidents: number | null;
};

export type CostBody = {
  sources: Sources;
  rollup_populated: boolean | null;
  days: Array<{ day: string; tenant: string; provider: string; requests: number; cached_requests: number; cost_usd: number }>;
  savings: Array<{ day: string; tenant: string; saved_usd: number }>;
  savings_usd_total: number | null;
  budgets: Array<{ tenant: string; cap_usd: number; spent_usd: number | null; ratio: number | null }>;
  cost_per_request: Array<{ alias: string; cost_usd: number }>;
};

export type QualityBody = {
  sources: Sources;
  runs: Array<{ id: string; started_at: string; passed: boolean; metrics: Record<string, unknown> }>;
  release_markers: Array<{ git_sha: string; deployed_at: string; status: string }>;
  categories: Array<{ category: string; passed: number; total: number }>;
  taxonomy: Array<{ tag: string; count: number }>;
  blocked: Array<{
    git_sha: string;
    deployed_at: string;
    pr_url: string | null;
    regressed: Array<{ metric: string; value: number; previous: number }>;
  }>;
};

export type ResidencyBody = {
  sources: Sources;
  by_region: Array<{ tenant: string; region: string; provider: string; requests: number }>;
  violations: number | null;
  redactions: Array<{ tenant: string; kind: string; count: number }>;
};

export type Slo = {
  name: string;
  objective: number;
  window: string;
  observed: number | null;
  budget_remaining: number | null;
};

export type ReliabilityBody = {
  sources: Sources;
  slos: Array<Slo>;
  latency_p95: Array<[number, number]>;
  error_rate: Array<[number, number]>;
  circuits: Array<{ provider: string; state: string }>;
  fallbacks: Array<{ tenant: string; alias: string; provider: string; count: number }>;
};

export type IncidentsBody = {
  sources: Sources;
  incidents: Array<{
    id: string;
    started_at: string;
    resolved_at: string | null;
    severity: string;
    title: string;
    summary: string;
    postmortem_url: string | null;
    open: boolean;
  }>;
};

export type ReleasesBody = {
  sources: Sources;
  releases: Array<{ git_sha: string; deployed_at: string; status: string; pr_url: string | null; eval_passed: boolean | null }>;
};

export type PageBodies = {
  home: HomeBody;
  cost: CostBody;
  quality: QualityBody;
  residency: ResidencyBody;
  reliability: ReliabilityBody;
  incidents: IncidentsBody;
  releases: ReleasesBody;
};

type Check = (value: unknown) => boolean;

const SOURCE_STATES: ReadonlyArray<unknown> = ["ok", "unconfigured", "error"];

const str: Check = (value) => typeof value === "string";
const num: Check = (value) => typeof value === "number" && Number.isFinite(value);
const int: Check = (value) => Number.isInteger(value);
const bool: Check = (value) => typeof value === "boolean";

function nullable(check: Check): Check {
  return (value) => value === null || check(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Every named field present and passing; extra fields are allowed. */
function shape(fields: Record<string, Check>): Check {
  return (value) =>
    isRecord(value) && Object.entries(fields).every(([name, check]) => name in value && check(value[name]));
}

function listOf(check: Check): Check {
  return (value) => Array.isArray(value) && value.every(check);
}

const pair: Check = (value) => Array.isArray(value) && value.length === 2 && num(value[0]) && num(value[1]);

const sources = shape({
  postgres: (value) => SOURCE_STATES.includes(value),
  prometheus: (value) => SOURCE_STATES.includes(value),
});

const CHECKS: { [N in PageName]: Check } = {
  home: shape({
    sources,
    requests_today: nullable(int),
    cost_usd_today: nullable(num),
    p95_seconds: nullable(num),
    error_rate: nullable(num),
    residency_violations_today: nullable(int),
    last_eval: nullable(shape({ finished_at: str, passed: bool, decision_accuracy: nullable(num) })),
    open_incidents: nullable(int),
  }),
  cost: shape({
    sources,
    rollup_populated: nullable(bool),
    days: listOf(shape({ day: str, tenant: str, provider: str, requests: int, cached_requests: int, cost_usd: num })),
    savings: listOf(shape({ day: str, tenant: str, saved_usd: num })),
    savings_usd_total: nullable(num),
    budgets: listOf(shape({ tenant: str, cap_usd: num, spent_usd: nullable(num), ratio: nullable(num) })),
    cost_per_request: listOf(shape({ alias: str, cost_usd: num })),
  }),
  quality: shape({
    sources,
    runs: listOf(shape({ id: str, started_at: str, passed: bool, metrics: isRecord })),
    release_markers: listOf(shape({ git_sha: str, deployed_at: str, status: str })),
    categories: listOf(shape({ category: str, passed: int, total: int })),
    taxonomy: listOf(shape({ tag: str, count: int })),
    blocked: listOf(
      shape({
        git_sha: str,
        deployed_at: str,
        pr_url: nullable(str),
        regressed: listOf(shape({ metric: str, value: num, previous: num })),
      }),
    ),
  }),
  residency: shape({
    sources,
    by_region: listOf(shape({ tenant: str, region: str, provider: str, requests: int })),
    violations: nullable(int),
    redactions: listOf(shape({ tenant: str, kind: str, count: num })),
  }),
  reliability: shape({
    sources,
    slos: listOf(
      shape({ name: str, objective: num, window: str, observed: nullable(num), budget_remaining: nullable(num) }),
    ),
    latency_p95: listOf(pair),
    error_rate: listOf(pair),
    circuits: listOf(shape({ provider: str, state: str })),
    fallbacks: listOf(shape({ tenant: str, alias: str, provider: str, count: num })),
  }),
  incidents: shape({
    sources,
    incidents: listOf(
      shape({
        id: str,
        started_at: str,
        resolved_at: nullable(str),
        severity: str,
        title: str,
        summary: str,
        postmortem_url: nullable(str),
        open: bool,
      }),
    ),
  }),
  releases: shape({
    sources,
    releases: listOf(
      shape({ git_sha: str, deployed_at: str, status: str, pr_url: nullable(str), eval_passed: nullable(bool) }),
    ),
  }),
};

function guard<N extends PageName>(name: N): (value: unknown) => value is PageBodies[N] {
  return (value): value is PageBodies[N] => CHECKS[name](value);
}

export const GUARDS: { [N in PageName]: (value: unknown) => value is PageBodies[N] } = {
  home: guard("home"),
  cost: guard("cost"),
  quality: guard("quality"),
  residency: guard("residency"),
  reliability: guard("reliability"),
  incidents: guard("incidents"),
  releases: guard("releases"),
};
