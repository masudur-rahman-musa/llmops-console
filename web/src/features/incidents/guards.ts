import { bool, listOf, nullable, shape, sources, str, type Sources } from "@/lib/shapes";

/** The incidents page body and its guard, checking only the fields the page renders. */

export type IncidentsBody = {
  sources: Sources;
  incidents: Array<{
    id: string;
    started_at: string;
    resolved_at: string | null;
    severity: string;
    title: string;
    summary: string;
    postmortem_url: string | null;
    open: boolean;
  }>;
};

const check = shape({
  sources,
  incidents: listOf(
    shape({
      id: str,
      started_at: str,
      resolved_at: nullable(str),
      severity: str,
      title: str,
      summary: str,
      postmortem_url: nullable(str),
      open: bool,
    }),
  ),
});

export function incidentsGuard(value: unknown): value is IncidentsBody {
  return check(value);
}
