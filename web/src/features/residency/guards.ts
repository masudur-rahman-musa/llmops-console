import { int, listOf, nullable, num, shape, sources, str, type Sources } from "@/lib/shapes";

/** The residency page body and its guard, checking only the fields the page renders. */

export type ResidencyBody = {
  sources: Sources;
  by_region: Array<{ tenant: string; region: string; provider: string; requests: number }>;
  violations: number | null;
  redactions: Array<{ tenant: string; kind: string; count: number }>;
};

const check = shape({
  sources,
  by_region: listOf(shape({ tenant: str, region: str, provider: str, requests: int })),
  violations: nullable(int),
  redactions: listOf(shape({ tenant: str, kind: str, count: num })),
});

export function residencyGuard(value: unknown): value is ResidencyBody {
  return check(value);
}
