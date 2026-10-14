import { listOf, nullable, num, pair, shape, sources, str, type Sources } from "@/lib/shapes";

/** The reliability page body and its guard, checking only the fields the page renders. */

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

const check = shape({
  sources,
  slos: listOf(shape({ name: str, objective: num, window: str, observed: nullable(num), budget_remaining: nullable(num) })),
  latency_p95: listOf(pair),
  error_rate: listOf(pair),
  circuits: listOf(shape({ provider: str, state: str })),
  fallbacks: listOf(shape({ tenant: str, alias: str, provider: str, count: num })),
});

export function reliabilityGuard(value: unknown): value is ReliabilityBody {
  return check(value);
}
