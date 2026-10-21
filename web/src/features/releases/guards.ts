import { bool, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The releases page body and its guard, checking only the fields the page renders. */

export type ReleasesBody = {
  sources: Sources;
  releases: Array<{ git_sha: string; deployed_at: string; status: string; pr_url: string | null; eval_passed: boolean | null; check_score: number | null }>;
};

const check = shape({
  sources,
  releases: listOf(
    shape({ git_sha: str, deployed_at: str, status: str, pr_url: nullable(str), eval_passed: nullable(bool), check_score: nullable(num) }),
  ),
});

export function releasesGuard(value: unknown): value is ReleasesBody {
  return check(value);
}
