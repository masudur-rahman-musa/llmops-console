/**
 * Paging for the panels that grow day by day: 20 rows a page, the page number
 * kept in the URL under one name per panel. Requests is paged by console-api;
 * the rest are sliced here. Pinned by tests/pager.test.tsx.
 */

export const PAGE_SIZE = 20;

export type Params = Record<string, string | string[] | undefined>;

/** The URL names, one per paged panel. */
export const PAGE_PARAMS = {
  requests: "requests_page",
  days: "days_page",
  taxonomy: "taxonomy_page",
  blocked: "blocked_page",
} as const;

/**
 * A URL value as a page number: the first of a repeated value, a bad one
 * (missing, 0, negative, not a whole number) as 1, and past `pages` as the last.
 */
export function parsePage(value: string | string[] | undefined, pages?: number): number {
  const raw = Array.isArray(value) ? value[0] : value;
  const page = raw !== undefined && /^[0-9]+$/.test(raw) ? Number(raw) : 1;
  const clamped = Number.isSafeInteger(page) && page >= 1 ? page : 1;
  return pages === undefined ? clamped : Math.min(clamped, Math.max(pages, 1));
}

/** How many pages `total` rows fill; at least 1. */
export function pageCount(total: number): number {
  return Math.max(1, Math.ceil(total / PAGE_SIZE));
}

export function slicePage<T>(rows: T[], page: number): T[] {
  return rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
}

/** `?...` for `page` under `name`, keeping every other panel's value as it is. */
export function pageHref(params: Params, name: string, page: number): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (key === name || value === undefined) {
      continue;
    }
    for (const item of Array.isArray(value) ? value : [value]) {
      query.append(key, item);
    }
  }
  query.set(name, String(page));
  return `?${query.toString()}`;
}

/** One panel's rows in the browser: its page from the URL, that page's rows, the page count. */
export function pageOf<T>(rows: T[], params: Params, name: string): { rows: T[]; page: number; pages: number } {
  const pages = pageCount(rows.length);
  const page = parsePage(params[name], pages);
  return { rows: slicePage(rows, page), page, pages };
}
