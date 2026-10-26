import { readFileSync } from "node:fs";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";

type Fixture = Record<string, unknown>;
type Params = Record<string, string | string[] | undefined>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const PAGER_TEXT = /Page \d+ of \d+/g;

function fixture(name: string): Fixture {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Fixture;
}

function stubFetch(body: unknown): ReturnType<typeof vi.fn> {
  const fetch = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function rowTexts(html: string): string[] {
  return Array.from(html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) => text(match[1]));
}

function rowsMatching(html: string, pattern: RegExp): string[] {
  return rowTexts(html).filter((row) => pattern.test(row));
}

function pagerTexts(html: string): string[] {
  return text(html).match(PAGER_TEXT) ?? [];
}

function props(params: Params): { searchParams: Promise<Params> } {
  return { searchParams: Promise.resolve(params) };
}

function requestedUrl(fetch: ReturnType<typeof vi.fn>): string {
  expect(fetch).toHaveBeenCalledTimes(1);
  return String(fetch.mock.calls[0][0]);
}

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.stubEnv("CONSOLE_TRADEOFF_PATH", "");
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("paged panels", () => {
  it("cost pages daily cost", async () => {
    // Catches a daily cost that ignores its page number, a browser page that shows every row, a pager whose text counts the wrong total, or a second pager left behind for the removed cache savings table.
    const body = fixture("cost");
    body.days = Array.from({ length: 45 }, (_, i) => ({
      day: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
      tenant: `dt${i}`,
      provider: "mistral",
      requests: 1,
      cached_requests: 0,
      cost_usd: 1,
    }));
    body.savings = Array.from({ length: 21 }, (_, i) => ({
      day: "2026-08-01",
      tenant: `st${i}`,
      saved_usd: 1,
    }));
    // The rows stay in the body (make numbers checks them) but the page no longer shows them.
    stubFetch(body);

    const html = renderToStaticMarkup(await CostPage(props({ days_page: "3" })));

    const dayRows = rowsMatching(html, /\bdt\d+\b/);
    expect(dayRows).toHaveLength(5);
    // Newest first: page 3 holds the oldest days, so dt0 (2026-08-01) is on it and dt44 (2026-08-17) is not.
    expect(dayRows.join(" ")).toMatch(/\bdt0\b/);
    expect(dayRows.join(" ")).not.toContain("dt44");
    expect(rowsMatching(html, /\bst\d+\b/)).toEqual([]);
    expect(pagerTexts(html)).toEqual(["Page 3 of 3"]);
  });

  it("quality pages taxonomy and blocked releases", async () => {
    // Catches a taxonomy or blocked list that shows every row, reads the wrong URL name, or pages one panel with the other's number.
    const body = fixture("quality");
    body.taxonomy = Array.from({ length: 21 }, (_, i) => ({ tag: `tag_${i}`, count: 1 }));
    body.blocked = Array.from({ length: 25 }, (_, i) => ({
      git_sha: `${String(i + 1).padStart(2, "0")}${"c".repeat(38)}`,
      deployed_at: "2026-09-11T14:00:00Z",
      pr_url: `https://github.com/example/helpdesk/pull/${i + 1}`,
      eval_run_id: null,
      regressed: [],
    }));
    stubFetch(body);

    const html = renderToStaticMarkup(await QualityPage(props({ taxonomy_page: "2", blocked_page: "2" })));

    expect(rowsMatching(html, /\bTag \d+\b/)).toHaveLength(1);
    expect(text(html)).toContain("Tag 20");
    expect(text(html)).not.toContain("Tag 19");
    const pulls = Array.from(html.matchAll(/helpdesk\/pull\/(\d+)/g), (m) => Number(m[1]));
    expect(pulls).toEqual([21, 22, 23, 24, 25]);
    expect(pagerTexts(html)).toEqual(["Page 2 of 2", "Page 2 of 2"]);
  });

  it("requests asks console-api for its page", async () => {
    // Catches a requests page that never sends the page, builds it from the wrong URL name, or takes the pager text from anything but the body.
    const body = fixture("requests");
    body.requests_page = 2;
    body.requests_pages = 7;
    const fetch = stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({ requests_page: "2" })));

    expect(requestedUrl(fetch)).toBe(`${API}/console/requests?page=2`);
    expect(pagerTexts(html)).toEqual(["Page 2 of 7"]);
  });

  it("a bad requests page asks for page 1", async () => {
    // Catches a page that puts raw URL text in the query string, or fails on a value that is not a whole number.
    const fetch = stubFetch(fixture("requests"));

    await RequestsPage(props({ requests_page: "abc" }));

    expect(requestedUrl(fetch)).toBe(`${API}/console/requests?page=1`);
  });

  it("fixed panels get no pager", async () => {
    // Catches a pager drawn under a fixed-size panel: 25 redaction kinds stay unpaged (and read Kind 0 to Kind 24, not kind_0), and the one paged panel gives the only pager.
    const body = fixture("requests");
    body.redactions = Array.from({ length: 25 }, (_, i) => ({ kind: `kind_${i}`, count: 1 }));
    body.requests_page = 1;
    body.requests_pages = 3;
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({})));

    expect(rowsMatching(html, /\bKind \d+\b/)).toHaveLength(25);
    expect(rowsMatching(html, /\bkind_\d+\b/)).toEqual([]);
    expect(pagerTexts(html)).toEqual(["Page 1 of 3"]);
  });

  it("requests guard refuses a body missing requests_pages", async () => {
    // Catches a guard that does not require the page count, so a body without it draws a pager of "undefined" instead of the unavailable notice.
    const body = fixture("requests");
    delete body.requests_pages;
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({})));

    expect(html).toContain("Console data unavailable");
  });
});
