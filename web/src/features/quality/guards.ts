import { bool, int, isRecord, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The quality page body and its guard, checking only the fields the page renders. */

export type GateScore = {
  name: string;
  label: string;
  value: number | null;
  threshold: number | null;
  passed: boolean | null;
  note: string | null;
};

export type QualityBody = {
  sources: Sources;
  gate: Array<GateScore>;
  runs: Array<{ id: string; started_at: string; passed: boolean; metrics: Record<string, unknown> }>;
  release_markers: Array<{ git_sha: string; deployed_at: string; status: string; eval_run_id: string | null }>;
  categories: Array<{ category: string; passed: number; total: number }>;
  taxonomy: Array<{ tag: string; count: number }>;
  needs_a_look: Array<{ case_id: string; play: number; why: string }>;
  blocked: Array<{
    git_sha: string;
    deployed_at: string;
    pr_url: string | null;
    regressed: Array<{ metric: string; value: number; previous: number }>;
  }>;
};

const check = shape({
  sources,
  gate: listOf(
    shape({
      name: str,
      label: str,
      value: nullable(num),
      threshold: nullable(num),
      passed: nullable(bool),
      note: nullable(str),
    }),
  ),
  runs: listOf(shape({ id: str, started_at: str, passed: bool, metrics: isRecord })),
  release_markers: listOf(shape({ git_sha: str, deployed_at: str, status: str, eval_run_id: nullable(str) })),
  categories: listOf(shape({ category: str, passed: int, total: int })),
  taxonomy: listOf(shape({ tag: str, count: int })),
  needs_a_look: listOf(shape({ case_id: str, play: int, why: str })),
  blocked: listOf(
    shape({
      git_sha: str,
      deployed_at: str,
      pr_url: nullable(str),
      regressed: listOf(shape({ metric: str, value: num, previous: num })),
    }),
  ),
});

export function qualityGuard(value: unknown): value is QualityBody {
  return check(value);
}
