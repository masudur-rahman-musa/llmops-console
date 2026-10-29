/**
 * Real-looking messy page bodies for all four tabs, the kind console-api really
 * sends: Prometheus fractions, 40-character shas, 64-hex prompt versions, 32-hex
 * trace ids, zero rows, empty days and nulls. Each passes its page guard, so a
 * page renders it in full. Used by tests/messy-pages.test.tsx with the junk check.
 */
import type { CostBody } from "@/features/cost/guards";
import type { HomeBody } from "@/features/home/guards";
import type { QualityBody } from "@/features/quality/guards";
import type { RequestsBody } from "@/features/requests/guards";

const SHA_A = "1f621019c4e8b2a7d3f5e6c0b9a8d7e6f5c4b3a2";
const SHA_B = "8799c18e2d4f6a1b3c5e7d9f0a2b4c6d8e0f1a3b";
const SHA_C = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678";
const PROMPT_VERSION = "9f2c4be07a1d3c58e6b0a4d2f1c7e9835b6a0d4c2e8f1a7b3c5d9e0f2a4b6c8d";

// Not fresh literals, so the extra fields console-api also sends (trace_id, run_kind) are allowed.
const home = {
  sources: { postgres: "ok" as const, prometheus: "ok" as const },
  requests_today: 0,
  cost_usd_today: 0.000028123456789,
  p95_seconds: 0.41237612345678,
  error_rate: 0.013012805401607991,
  last_eval: {
    id: "run-2026-10-27-nightly",
    finished_at: "2026-10-27T06:41:09.482913+00:00",
    passed: true,
    check_score: 48.484848484848484,
  },
  routing: {
    loaded_at: "2026-10-27T09:00:00.123456+00:00",
    aliases: [
      {
        name: "agent-default",
        purpose: "Everyday help desk chat replies on the cheap model.",
        candidates: [
          { provider: "anthropic", model: "claude-haiku-4-5-20251001" },
          { provider: "openai", model: "gpt-5-mini" },
        ],
      },
      { name: "embed", purpose: null, candidates: [{ provider: "local_embed", model: "bge-small-en-v1.5" }] },
    ],
  },
  grafana_url: "http://localhost:3002/d/gateway-overview",
};

const requests = {
  sources: { postgres: "ok" as const, prometheus: "ok" as const },
  // Prometheus increase() over a window: never quite whole.
  redactions: [
    { kind: "email", count: 40.99999999999999 },
    { kind: "iban", count: 7.000000000000001 },
  ],
  refusals: [{ reason: "out_of_scope", count: 12.333333333333334 }],
  guardrail_blocks: 5.000000000000001,
  tokens: [
    { model: "claude-haiku-4-5-20251001", input: 120450.33333333333, output: 30210.666666666668 },
    { model: "gpt-5-mini", input: 0, output: 0 },
  ],
  cache_hit_rate: 0.31254321987654,
  prompt_cache_share: 0.42131313131313,
  // Empty days, a day with only failures, and the Oct 26 and 27 counts the ledger really holds.
  per_day: [
    { day: "2026-10-23", requests: 0, errors: 0 },
    { day: "2026-10-24", requests: 0, errors: 2 },
    { day: "2026-10-25", requests: 0, errors: 0 },
    { day: "2026-10-26", requests: 14, errors: 0 },
    { day: "2026-10-27", requests: 31, errors: 3 },
  ],
  requests_page: 1,
  requests_pages: 1,
  requests: [
    {
      ts: "2026-10-27T08:30:00.918273+00:00",
      tenant: "nordlicht",
      alias: "agent-default",
      provider: "anthropic",
      model: "claude-haiku-4-5-20251001",
      input_tokens: 137,
      output_tokens: 29,
      cached: false,
      cost_usd: 0.000028123456789,
      prompt_version: PROMPT_VERSION,
      trace_id: "0af7651916cd43dd8448eb211c80319c",
      trace_url: "https://cloud.langfuse.com/project/demo/traces/0af7651916cd43dd8448eb211c80319c",
      status: 503,
      latency_ms: 30017,
    },
    {
      ts: "2026-10-27T08:12:00Z",
      tenant: "cuan",
      alias: "agent-default",
      provider: "anthropic",
      model: "claude-haiku-4-5-20251001",
      input_tokens: 0,
      output_tokens: 0,
      cached: true,
      cost_usd: 0,
      prompt_version: null,
      trace_id: "b7ad6b7169203331b7ad6b7169203331",
      trace_url: null,
      status: 500,
      latency_ms: null,
    },
  ],
  slow_ms: 1905.0000000000002,
  tenants: ["cuan", "nordlicht"],
  models: ["claude-haiku-4-5-20251001", "gpt-5-mini"],
  langfuse_url: "https://cloud.langfuse.com/project/demo/traces",
};

const cost = {
  sources: { postgres: "ok" as const, prometheus: null },
  days: [
    { day: "2026-10-27", tenant: "nordlicht", provider: "anthropic", requests: 31, cached_requests: 4, cost_usd: 18.250000000000004 },
    { day: "2026-10-27", tenant: "cuan", provider: "anthropic", requests: 0, cached_requests: 0, cost_usd: 0 },
    { day: "2026-10-26", tenant: "nordlicht", provider: "local", requests: 14, cached_requests: 14, cost_usd: 0.000123456789 },
  ],
  savings_usd_total: 4.550000000000001,
  prompt_cache_saved_usd: 0.00012345678912,
  month_spend_usd: 61.00000000000001,
  estimated: true,
  budgets: [
    { tenant: "nordlicht", cap_usd: 15, spent_usd: 15.070000000000002, ratio: 1.0046666666666666 },
    { tenant: "cuan", cap_usd: 50, spent_usd: 0, ratio: 0 },
    { tenant: "initech-eu", cap_usd: 20, spent_usd: null, ratio: null },
  ],
  cost_per_request: [
    { alias: "agent-default", cost_usd: 0.004212345678912 },
    { alias: "agent-strong", cost_usd: 0 },
  ],
  top_users: [
    { tenant: "nordlicht", user_id: "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c", requests: 120, cost_usd: 12.499999999999998 },
    { tenant: "cuan", user_id: "u_00000000000000000000000000000002", requests: 1, cost_usd: 0.000028123456789 },
  ],
};

const quality = {
  sources: { postgres: "ok" as const, prometheus: null },
  runs: [
    {
      id: "run-2026-10-27-nightly",
      started_at: "2026-10-27T06:00:00.482913+00:00",
      run_kind: "nightly",
      git_sha: SHA_A,
      passed: true,
      metrics: { check_score: 48.484848484848484, judge_score: 27.272727272727273 },
    },
    {
      id: "run-2026-10-26-nightly",
      started_at: "2026-10-26T06:00:00Z",
      run_kind: "nightly",
      git_sha: null,
      passed: false,
      metrics: { judge_score: null },
    },
    {
      id: "run-2026-10-25-pr",
      started_at: "2026-10-25T08:30:00Z",
      run_kind: "pr",
      git_sha: SHA_B,
      passed: true,
      metrics: { check_score: 100, judge_score: 0 },
    },
  ],
  release_markers: [
    { git_sha: SHA_A, deployed_at: "2026-10-27T07:00:00Z", status: "deployed", eval_run_id: "run-2026-10-27-nightly" },
    { git_sha: SHA_B, deployed_at: "2026-10-25T09:00:00Z", status: "deployed", eval_run_id: "run-2026-10-25-pr" },
    { git_sha: SHA_C, deployed_at: "2026-10-20T09:00:00Z", status: "deployed", eval_run_id: null },
  ],
  categories: [
    { category: "refusal", passed: 16, total: 33 },
    { category: "citation", passed: 0, total: 12 },
  ],
  taxonomy: [
    { tag: "wrong_tool", count: 4 },
    { tag: "missed_refusal", count: 1 },
  ],
  blocked: [
    {
      git_sha: SHA_B,
      deployed_at: "2026-10-25T09:00:00Z",
      pr_url: "https://github.com/example/helpdesk/pull/41",
      eval_run_id: "run-2026-10-25-pr",
      regressed: [{ metric: "check_score", value: 48.484848484848484, previous: 52.121212121212125 }],
    },
    {
      git_sha: SHA_C,
      deployed_at: "2026-10-20T09:00:00Z",
      pr_url: null,
      eval_run_id: null,
      regressed: [],
    },
  ],
  gate: [
    {
      name: "check_score",
      label: "Check score",
      value: 48.484848484848484,
      threshold: 48.48484848484848,
      passed: true,
      note: null,
    },
    {
      name: "judge_score",
      label: "Judge score",
      value: 27.272727272727273,
      threshold: null,
      passed: null,
      note: "Counts only while the judge agrees with people (kappa 0.7 or better).",
    },
  ],
};

export const MESSY: { home: HomeBody; requests: RequestsBody; cost: CostBody; quality: QualityBody } = {
  home,
  requests,
  cost,
  quality,
};
