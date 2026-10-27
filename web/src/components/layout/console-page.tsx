import type { ReactElement, ReactNode } from "react";

import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { readCostFilters, readRequestFilters, type CostFilters } from "@/lib/filters";
import { getConsolePage, type PageBodies, type PageName, type PageQuery } from "@/lib/console-api";
import { PAGE_PARAMS, parsePage, type Params } from "@/lib/paging";

type PageProps = { searchParams?: Promise<Params> };

/** The most days console-api sums for Daily cost, its `MAX_DAYS`; both ends count. */
const MAX_COST_DAYS = 90;
/** The days Daily cost shows when only `to` is chosen, console-api's own default window. */
const DEFAULT_COST_DAYS = 30;
const DAY_MS = 24 * 60 * 60 * 1000;

function shiftDay(day: string, days: number): string {
  return new Date(Date.parse(`${day}T00:00:00Z`) + days * DAY_MS).toISOString().slice(0, 10);
}

/**
 * The Daily cost range to ask console-api for, or none when the address names neither end.
 * A missing `to` is today (UTC), a missing `from` the default window before `to`, and a span
 * over `MAX_COST_DAYS` keeps the newest days, so a wide pick never turns the page into a 422.
 */
export function costRange(filters: CostFilters, today: string): { from: string; to: string } | undefined {
  if (filters.from === undefined && filters.to === undefined) {
    return undefined;
  }
  const to = filters.to ?? (filters.from !== undefined && filters.from > today ? filters.from : today);
  const earliest = shiftDay(to, 1 - MAX_COST_DAYS);
  const from = filters.from ?? shiftDay(to, 1 - DEFAULT_COST_DAYS);
  return { from: from < earliest ? earliest : from, to };
}

/** Callable bare in tests; Next reads the last shape, which takes its props. */
type PageComponent = {
  (): Promise<ReactElement>;
  (props: PageProps): Promise<ReactElement>;
};

/**
 * The opening every console page shares: read the page body, show the
 * Unavailable notice when it fails, else print the `<h1>`, the one-line
 * question the page answers, an optional `window` line saying what time
 * the page covers (Requests and Quality), and the SourceNotice once when the body has
 * sources. The page passes only
 * its body builder, which may be async so the cost page can still read the
 * tradeoff file, and which gets the URL params so a panel can read its page.
 * Requests asks console-api for the page in `requests_page`, parsed to a whole
 * number first, with the filters from the address (bad values dropped). Cost asks for
 * the Daily cost From and To as `costRange` builds them. Pinned by tests/shared-parts.test.tsx, tests/clarity.test.tsx
 * and tests/paging-pages.test.tsx.
 */
export function consolePage<N extends PageName>(
  name: N,
  title: string,
  question: string,
  render: (body: PageBodies[N], params: Params) => ReactNode | Promise<ReactNode>,
  window?: string,
): PageComponent {
  return async function Page(props?: PageProps): Promise<ReactElement> {
    const params = (await props?.searchParams) ?? {};
    const today = new Date().toISOString().slice(0, 10);
    const query: PageQuery | undefined =
      name === "requests"
        ? { page: parsePage(params[PAGE_PARAMS.requests]), filters: readRequestFilters(params) }
        : name === "cost"
          ? { range: costRange(readCostFilters(params), today) }
          : undefined;
    const result = await getConsolePage(name, {}, query);
    if (!result.ok) {
      return <Unavailable title={title} reason={result.reason} />;
    }
    return (
      <>
        <h1>{title}</h1>
        <p className="question">{question}</p>
        {window !== undefined && <p className="window">{window}</p>}
        {"sources" in result.body && <SourceNotice sources={result.body.sources} />}
        {await render(result.body, params)}
      </>
    );
  };
}
