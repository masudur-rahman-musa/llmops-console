import { bool, int, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The requests page body and its guard, checking only the fields the page renders. */

export type RequestsBody = {
  sources: Sources;
  redactions: Array<{ kind: string; count: number }>;
  refusals: Array<{ reason: string; count: number }>;
  guardrail_blocks: number | null;
  tokens: Array<{ model: string; input: number; output: number }>;
  cache_hit_rate: number | null;
  per_day: Array<{ day: string; requests: number; errors: number | null }>;
  requests: Array<{
    ts: string;
    tenant: string;
    model: string;
    input_tokens: number;
    output_tokens: number;
    cached: boolean;
    cost_usd: number;
    prompt_version: string | null;
    trace_url: string | null;
    status: number;
    latency_ms: number | null;
  }>;
  requests_page: number;
  requests_pages: number;
  slow_ms: number | null;
  tenants: string[];
  models: string[];
  langfuse_url: string | null;
};

const check = shape({
  sources,
  redactions: listOf(shape({ kind: str, count: num })),
  refusals: listOf(shape({ reason: str, count: num })),
  guardrail_blocks: nullable(num),
  tokens: listOf(shape({ model: str, input: num, output: num })),
  cache_hit_rate: nullable(num),
  per_day: listOf(shape({ day: str, requests: int, errors: nullable(num) })),
  requests: listOf(
    shape({
      ts: str,
      tenant: str,
      model: str,
      input_tokens: num,
      output_tokens: num,
      cached: bool,
      cost_usd: num,
      prompt_version: nullable(str),
      trace_url: nullable(str),
      status: int,
      latency_ms: nullable(int),
    }),
  ),
  requests_page: num,
  requests_pages: num,
  slow_ms: nullable(num),
  tenants: listOf(str),
  models: listOf(str),
  langfuse_url: nullable(str),
});

export function requestsGuard(value: unknown): value is RequestsBody {
  return check(value);
}
