import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { CONSOLE_PATHS, GUARDS, type PageName } from "@/lib/console-api";

const PAGES: PageName[] = ["home", "requests", "cost", "quality"];

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
    ["routing"],
    ["grafana_url"],
    ["last_eval", "finished_at"],
    ["last_eval", "passed"],
    ["last_eval", "check_score"],
    ["routing", "loaded_at"],
    ["routing", "aliases"],
    ["routing", "aliases", -1, "name"],
    ["routing", "aliases", -1, "purpose"],
    ["routing", "aliases", -1, "candidates"],
    ["routing", "aliases", -1, "candidates", -1, "provider"],
    ["routing", "aliases", -1, "candidates", -1, "model"],
  ],
  cost: [
    ["sources"],
    ["rollup_populated"],
    ["days"],
    ["savings_usd_total"],
    ["month_spend_usd"],
    ["budgets"],
    ["cost_per_request"],
    ["days", -1, "day"],
    ["days", -1, "tenant"],
    ["days", -1, "provider"],
    ["days", -1, "requests"],
    ["days", -1, "cached_requests"],
    ["days", -1, "cost_usd"],
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
    ["release_markers", -1, "eval_run_id"],
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
  requests: [
    ["sources"],
    ["redactions"],
    ["refusals"],
    ["guardrail_blocks"],
    ["tokens"],
    ["cache_hit_rate"],
    ["per_day"],
    ["requests"],
    ["per_day", -1, "day"],
    ["per_day", -1, "requests"],
    ["per_day", -1, "errors"],
    ["redactions", -1, "kind"],
    ["redactions", -1, "count"],
    ["refusals", -1, "reason"],
    ["refusals", -1, "count"],
    ["tokens", -1, "model"],
    ["tokens", -1, "input"],
    ["tokens", -1, "output"],
    ["requests", -1, "model"],
    ["requests", -1, "input_tokens"],
    ["requests", -1, "output_tokens"],
    ["requests", -1, "cached"],
    ["requests", -1, "cost_usd"],
    ["requests", -1, "prompt_version"],
    ["requests", -1, "trace_url"],
    ["requests", -1, "status"],
    ["requests", -1, "latency_ms"],
    ["slow_ms"],
    ["tenants"],
    ["models"],
    ["langfuse_url"],
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
    for (const field of ["requests_today", "cost_usd_today", "p95_seconds", "error_rate", "last_eval", "grafana_url"]) {
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
    nullCost.month_spend_usd = null;
    expect(passes("cost", nullCost), "cost nulls").toBe(true);

    // Catches a requests guard that refuses a day whose errors are null (Prometheus down), which would blank the page.
    expect(passes("requests", withValue("requests", ["per_day", -1, "errors"], null)), "errors null").toBe(true);
    expect(passes("requests", withValue("requests", ["per_day"], [])), "per_day empty").toBe(true);
  });

  it("cost guard no longer checks the savings rows", () => {
    // Catches a cost guard that still demands `savings` (the page stopped reading it, so a body without it must still render), and one that stopped checking the total the Cache saved card shows.
    const noSavings = fixture("cost");
    delete noSavings.savings;
    expect(passes("cost", noSavings)).toBe(true);
    expect(passes("cost", withValue("cost", ["savings_usd_total"], "4.55"))).toBe(false);
  });

  it("top_users is required and shape-checked", () => {
    // Catches a cost guard that ignores `top_users` (a body without it, or with a row lacking `user_id`, would reach the page and crash it), one that lets a text count or cost through, and one that refuses an empty list (a tenant with no user ids yet is a normal body).
    expect(passes("cost", without("cost", ["top_users"])), "key missing").toBe(false);
    expect(passes("cost", without("cost", ["top_users", -1, "user_id"])), "row without user_id").toBe(false);
    expect(passes("cost", without("cost", ["top_users", -1, "tenant"])), "row without tenant").toBe(false);
    expect(passes("cost", withValue("cost", ["top_users", -1, "requests"], "30")), "text requests").toBe(false);
    expect(passes("cost", withValue("cost", ["top_users", -1, "cost_usd"], "0.5")), "text cost").toBe(false);
    expect(passes("cost", withValue("cost", ["top_users"], [])), "empty list").toBe(true);
  });

  it("requests guard checks the per day types", () => {
    // Catches a guard that lets a text count or a text day through as a per day entry.
    expect(passes("requests", withValue("requests", ["per_day", -1, "requests"], "7"))).toBe(false);
    expect(passes("requests", withValue("requests", ["per_day", -1, "errors"], "2"))).toBe(false);
    expect(passes("requests", withValue("requests", ["per_day", -1, "day"], 20261026))).toBe(false);
  });

  it("requests guard checks the filter fields", () => {
    // Catches a requests guard that lets a text status or latency through, accepts a null status (every row has one), refuses the nulls console-api may send (no latency, no latency line, no Langfuse project), or lets a non-list of tenants or models through.
    expect(passes("requests", withValue("requests", ["requests", -1, "status"], "502"))).toBe(false);
    expect(passes("requests", withValue("requests", ["requests", -1, "status"], null))).toBe(false);
    expect(passes("requests", withValue("requests", ["requests", -1, "status"], 502))).toBe(true);
    expect(passes("requests", withValue("requests", ["requests", -1, "latency_ms"], "412"))).toBe(false);
    expect(passes("requests", withValue("requests", ["requests", -1, "latency_ms"], null))).toBe(true);
    expect(passes("requests", withValue("requests", ["slow_ms"], null))).toBe(true);
    expect(passes("requests", withValue("requests", ["slow_ms"], "1905"))).toBe(false);
    expect(passes("requests", withValue("requests", ["langfuse_url"], null))).toBe(true);
    expect(passes("requests", withValue("requests", ["langfuse_url"], 7))).toBe(false);
    expect(passes("requests", withValue("requests", ["tenants"], []))).toBe(true);
    expect(passes("requests", withValue("requests", ["tenants"], "acme"))).toBe(false);
    expect(passes("requests", withValue("requests", ["models"], [7]))).toBe(false);
  });

  it("quality guard takes a marker eval_run_id", () => {
    // Catches a quality guard that refuses a marker with no run (null), or one that lets a non-string run id through.
    const path: Path = ["release_markers", -1, "eval_run_id"];
    expect(passes("quality", withValue("quality", path, null))).toBe(true);
    expect(passes("quality", withValue("quality", path, "run-2026-09-14-nightly"))).toBe(true);
    expect(passes("quality", withValue("quality", path, 7))).toBe(false);
  });

  it("home guard takes a null verdict", () => {
    // Catches a home guard that still demands a boolean `passed` (a null verdict, no baseline, would blank the page) or one that stops checking the type (a string passes).
    expect(passes("home", withValue("home", ["last_eval", "passed"], null))).toBe(true);
    expect(passes("home", withValue("home", ["last_eval", "passed"], false))).toBe(true);
    expect(passes("home", withValue("home", ["last_eval", "passed"], "yes"))).toBe(false);
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
        // A null prometheus is a source the page does not have, so it is not a bad value there; postgres has no such case.
        const bad = source === "prometheus" ? ["down", "OK", "", 1] : ["down", "OK", "", null, 1];
        for (const value of bad) {
          expect(passes(name, withValue(name, ["sources", source], value)), `${name} ${source} ${String(value)}`).toBe(false);
        }
      }
    }
  });

  it("a null prometheus passes the guard and a null postgres fails it", () => {
    // Catches a guard that still refuses the null Cost and Quality now send, one that lets null through for postgres too, and one that lets null through by dropping the check on sources altogether.
    for (const name of PAGES) {
      expect(passes(name, withValue(name, ["sources", "prometheus"], null)), `${name} prometheus null`).toBe(true);
      expect(passes(name, withValue(name, ["sources", "postgres"], null)), `${name} postgres null`).toBe(false);
      expect(passes(name, withValue(name, ["sources", "prometheus"], "down")), `${name} prometheus down`).toBe(false);
    }
  });
});
