import { bool, int, isRecord, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The quality page body and its guard, checking only the fields the page renders. */

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

const check = shape({
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
});

export function qualityGuard(value: unknown): value is QualityBody {
  return check(value);
}
