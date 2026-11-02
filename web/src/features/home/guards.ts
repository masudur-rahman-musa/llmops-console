import { bool, int, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The Overview page body and its guard, checking only the fields the page renders. */

export type RoutingAlias = {
  name: string;
  purpose: string | null;
  candidates: ReadonlyArray<{ provider: string; model: string }>;
};

export type HomeBody = {
  sources: Sources;
  requests_today: number | null;
  cost_usd_today: number | null;
  p95_seconds: number | null;
  error_rate: number | null;
  last_eval: { finished_at: string; passed: boolean | null; outcome_every_play: number | null } | null;
  routing: { loaded_at: string; aliases: ReadonlyArray<RoutingAlias> };
  grafana_url: string | null;
};

const check = shape({
  sources,
  requests_today: nullable(int),
  cost_usd_today: nullable(num),
  p95_seconds: nullable(num),
  error_rate: nullable(num),
  last_eval: nullable(shape({ finished_at: str, passed: nullable(bool), outcome_every_play: nullable(num) })),
  routing: shape({
    loaded_at: str,
    aliases: listOf(
      shape({ name: str, purpose: nullable(str), candidates: listOf(shape({ provider: str, model: str })) }),
    ),
  }),
  grafana_url: nullable(str),
});

export function homeGuard(value: unknown): value is HomeBody {
  return check(value);
}
