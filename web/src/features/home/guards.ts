import { bool, int, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The home page body and its guard, checking only the fields the page renders. */

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

const check = shape({
  sources,
  requests_today: nullable(int),
  cost_usd_today: nullable(num),
  p95_seconds: nullable(num),
  error_rate: nullable(num),
  residency_violations_today: nullable(int),
  last_eval: nullable(shape({ finished_at: str, passed: bool, decision_accuracy: nullable(num) })),
  open_incidents: nullable(int),
});

export function homeGuard(value: unknown): value is HomeBody {
  return check(value);
}
