import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { CONSOLE_PATHS, GUARDS, type PageName } from "@/lib/console-api";

const PAGES: PageName[] = ["home", "cost", "quality", "reliability", "incidents", "releases"];

type Path = ReadonlyArray<string | number>;

function fixture(name: PageName): Record<string, unknown> {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Record<string, unknown>;
}

function passes(name: PageName, value: unknown): boolean {
  const guard = GUARDS[name] as (value: unknown) => boolean;
  return guard(value);
}

// Walks to the parent of the last step; a negative index counts from the end, so -1 is the last element.
function parentOf(root: unknown, path: Path): { parent: Record<string, unknown> | unknown[]; key: string | number } {
  let node = root;
  for (const step of path.slice(0, -1)) {
    const container = node as Record<string | number, unknown> & unknown[];
    node = typeof step === "number" && step < 0 ? container[container.length + step] : container[step];
  }
  const parent = node as Record<string, unknown> | unknown[];
  const last = path[path.length - 1];
  const key = Array.isArray(parent) && typeof last === "number" && last < 0 ? parent.length + last : last;
  return { parent, key };
}

function without(name: PageName, path: Path): Record<string, unknown> {
  const root = fixture(name);
  const { parent, key } = parentOf(root, path);
  // An element is spliced out rather than deleted, so a pair really gets shorter instead of holding a hole.
  if (Array.isArray(parent)) parent.splice(key as number, 1);
  else delete parent[key as string];
  return root;
}

function withValue(name: PageName, path: Path, value: unknown): Record<string, unknown> {
  const root = fixture(name);
  const { parent, key } = parentOf(root, path);
  (parent as Record<string | number, unknown>)[key] = value;
  return root;
}

// Each rendered field the contract names; nested ones are removed from the LAST element, so a guard reading only the first is caught.
const RENDERED: Record<PageName, Path[]> = {
  home: [
    ["sources"],
    ["requests_today"],
    ["cost_usd_today"],
    ["p95_seconds"],
    ["error_rate"],
    ["last_eval"],
    ["open_incidents"],
    ["last_eval", "finished_at"],
    ["last_eval", "passed"],
    ["last_eval", "check_score"],
  ],
  cost: [
    ["sources"],
    ["rollup_populated"],
    ["days"],
    ["savings"],
    ["savings_usd_total"],
    ["budgets"],
    ["cost_per_request"],
    ["days", -1, "day"],
    ["days", -1, "tenant"],
    ["days", -1, "provider"],
    ["days", -1, "requests"],
    ["days", -1, "cached_requests"],
    ["days", -1, "cost_usd"],
    ["savings", -1, "day"],
    ["savings", -1, "tenant"],
    ["savings", -1, "saved_usd"],
    ["budgets", -1, "tenant"],
    ["budgets", -1, "cap_usd"],
    ["budgets", -1, "spent_usd"],
    ["budgets", -1, "ratio"],
    ["cost_per_request", -1, "alias"],
    ["cost_per_request", -1, "cost_usd"],
  ],
  quality: [
    ["sources"],
    ["runs"],
    ["release_markers"],
    ["categories"],
    ["taxonomy"],
    ["blocked"],
    ["runs", -1, "id"],
    ["runs", -1, "started_at"],
    ["runs", -1, "passed"],
    ["runs", -1, "metrics"],
    ["release_markers", -1, "git_sha"],
    ["release_markers", -1, "deployed_at"],
    ["release_markers", -1, "status"],
    ["categories", -1, "category"],
    ["categories", -1, "passed"],
    ["categories", -1, "total"],
    ["taxonomy", -1, "tag"],
    ["taxonomy", -1, "count"],
    ["blocked", -1, "git_sha"],
    ["blocked", -1, "deployed_at"],
    ["blocked", -1, "pr_url"],
    ["blocked", -1, "regressed"],
    ["blocked", -1, "regressed", -1, "metric"],
    ["blocked", -1, "regressed", -1, "value"],
    ["blocked", -1, "regressed", -1, "previous"],
    ["gate"],
    ["gate", -1, "name"],
    ["gate", -1, "label"],
    ["gate", -1, "value"],
    ["gate", -1, "threshold"],
    ["gate", -1, "passed"],
  ],
  reliability: [
    ["sources"],
    ["latency_p95"],
    ["error_rate"],
    ["circuits"],
    ["fallbacks"],
    ["latency_p95", -1, 1],
    ["error_rate", -1, 1],
    ["circuits", -1, "provider"],
    ["circuits", -1, "state"],
    ["fallbacks", -1, "tenant"],
    ["fallbacks", -1, "alias"],
    ["fallbacks", -1, "provider"],
    ["fallbacks", -1, "count"],
    ["slos"],
    ["slos", -1, "name"],
    ["slos", -1, "objective"],
    ["slos", -1, "window"],
    ["slos", -1, "observed"],
    ["slos", -1, "budget_remaining"],
  ],
  incidents: [
    ["sources"],
    ["incidents"],
    ["incidents", -1, "id"],
    ["incidents", -1, "started_at"],
    ["incidents", -1, "resolved_at"],
    ["incidents", -1, "severity"],
    ["incidents", -1, "title"],
    ["incidents", -1, "summary"],
    ["incidents", -1, "postmortem_url"],
    ["incidents", -1, "open"],
  ],
  releases: [
    ["sources"],
    ["releases"],
    ["releases", -1, "git_sha"],
    ["releases", -1, "deployed_at"],
    ["releases", -1, "status"],
    ["releases", -1, "pr_url"],
    ["releases", -1, "eval_passed"],
  ],
};

describe("page body guards", () => {
  it("each page fixture passes its guard", () => {
    // Catches a guard stricter than console-api's output: one refusing an extra field, a null where the contract allows null, or an absent check_score.
    expect(Object.keys(GUARDS).sort()).toEqual(Object.keys(CONSOLE_PATHS).sort());
    for (const name of PAGES) {
      expect(passes(name, fixture(name)), name).toBe(true);
    }

    const nullHome = fixture("home");
    for (const field of ["requests_today", "cost_usd_today", "p95_seconds", "error_rate", "last_eval", "open_incidents"]) {
      nullHome[field] = null;
    }
    expect(passes("home", nullHome), "home all null").toBe(true);
    expect(passes("home", withValue("home", ["last_eval", "check_score"], null)), "home accuracy null").toBe(true);

    expect(passes("quality", withValue("quality", ["runs", -1, "metrics", "check_score"], null)), "accuracy null").toBe(true);
    expect(passes("quality", without("quality", ["runs", -1, "metrics", "check_score"])), "accuracy absent").toBe(true);
    expect(passes("quality", withValue("quality", ["blocked", -1, "regressed"], [])), "regressed empty").toBe(true);

    // Catches a quality guard that refuses a gate row whose value, threshold and verdict are all null (no run, no baseline).
    const nullGate = fixture("quality");
    Object.assign((nullGate.gate as Array<Record<string, unknown>>)[0], { value: null, threshold: null, passed: null });
    expect(passes("quality", nullGate), "gate nulls").toBe(true);

    const nullCost = withValue("cost", ["rollup_populated"], null);
    nullCost.savings_usd_total = null;
    expect(passes("cost", nullCost), "cost nulls").toBe(true);
    // Catches a reliability guard that requires a number where a budget row answers null for no traffic.
    const noTraffic = withValue("reliability", ["slos", 0, "observed"], null);
    (noTraffic.slos as Array<Record<string, unknown>>)[0].budget_remaining = null;
    expect(passes("reliability", noTraffic), "slo observed and budget null").toBe(true);
    expect(passes("reliability", withValue("reliability", ["slos"], [])), "slos empty").toBe(true);
  });

  it("a missing rendered field fails the guard", () => {
    // Catches a guard that skips a rendered field, accepts undefined where only null is allowed, or checks only the first element of a list.
    for (const name of PAGES) {
      for (const path of RENDERED[name]) {
        expect(passes(name, without(name, path)), `${name}.${path.join(".")}`).toBe(false);
      }
    }
    for (const name of PAGES) {
      expect(passes(name, null), `${name} null`).toBe(false);
      expect(passes(name, "{}"), `${name} string`).toBe(false);
    }
  });

  it("a sources value outside ok unconfigured error fails the guard", () => {
    // Catches a guard that checks only that sources exists, only one of its two members, or only for the value ok.
    for (const name of PAGES) {
      for (const source of ["postgres", "prometheus"]) {
        for (const state of ["ok", "unconfigured", "error"]) {
          expect(passes(name, withValue(name, ["sources", source], state)), `${name} ${source} ${state}`).toBe(true);
        }
        for (const bad of ["down", "OK", "", null, 1]) {
          expect(passes(name, withValue(name, ["sources", source], bad)), `${name} ${source} ${String(bad)}`).toBe(false);
        }
      }
    }
  });
});
