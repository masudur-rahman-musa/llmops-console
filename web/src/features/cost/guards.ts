import { bool, int, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The cost page body and its guard, checking only the fields the page renders. */

export type CostBody = {
  sources: Sources;
  rollup_populated: boolean | null;
  days: Array<{ day: string; tenant: string; provider: string; requests: number; cached_requests: number; cost_usd: number }>;
  savings_usd_total: number | null;
  budgets: Array<{ tenant: string; cap_usd: number; spent_usd: number | null; ratio: number | null }>;
  month_spend_usd: number | null;
  cost_per_request: Array<{ alias: string; cost_usd: number }>;
  top_users: Array<{ tenant: string; user_id: string; requests: number; cost_usd: number }>;
};

const check = shape({
  sources,
  rollup_populated: nullable(bool),
  days: listOf(shape({ day: str, tenant: str, provider: str, requests: int, cached_requests: int, cost_usd: num })),
  savings_usd_total: nullable(num),
  budgets: listOf(shape({ tenant: str, cap_usd: num, spent_usd: nullable(num), ratio: nullable(num) })),
  month_spend_usd: nullable(num),
  cost_per_request: listOf(shape({ alias: str, cost_usd: num })),
  top_users: listOf(shape({ tenant: str, user_id: str, requests: int, cost_usd: num })),
});

export function costGuard(value: unknown): value is CostBody {
  return check(value);
}
