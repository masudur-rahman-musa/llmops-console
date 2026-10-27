import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import HomePage from "@/app/(console)/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";
import { WINDOW_DAYS, WINDOW_LINES } from "@/lib/labels";

type PageName = "home" | "requests" | "cost" | "quality";
type Fixture = Record<string, unknown>;
type Params = Record<string, string | string[] | undefined>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const DASHBOARD = fileURLToPath(new URL("../../backend/app/api/routes/dashboard.py", import.meta.url));

const PAGES: Record<PageName, (props?: { searchParams: Promise<Params> }) => Promise<ReactElement>> = {
  home: HomePage,
  requests: RequestsPage,
  cost: CostPage,
  quality: QualityPage,
};

function fixture(name: PageName): Fixture {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Fixture;
}

async function render(name: PageName, body: unknown = fixture(name), params: Params = {}): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return renderToStaticMarkup(await PAGES[name]({ searchParams: Promise.resolve(params) }));
}

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,:;)%])/g, "$1")
    .trim();
}

function rowTexts(html: string): string[] {
  return Array.from(html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) => text(match[1]));
}

function count(html: string, pattern: RegExp): number {
  return (html.match(pattern) ?? []).length;
}

const OVER_BAR = /<div\b[^>]*class="(?=[^"]*\bbar\b)[^"]*\bover\b[^"]*"/g;

function budgetBody(budgets: Array<Record<string, unknown>>): Fixture {
  return { ...fixture("cost"), budgets };
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

describe("console clarity", () => {
  it("requests and quality name their window under the question", async () => {
    // Catches a window line missing, placed after the source notice or a table instead of right under the question, or worded differently.
    const expected: Array<[PageName, string]> = [
      ["requests", "Last 30 days. Newest requests lists every request, newest first."],
      ["quality", "Last 30 days."],
    ];
    for (const [name, sentence] of expected) {
      const html = await render(name);
      const match = html.match(/<p class="question">[^<]*<\/p><p class="window">([^<]*)<\/p>/);
      expect(match, name).not.toBeNull();
      expect(text(match?.[1] ?? ""), name).toBe(sentence);
    }
  });

  it("the window matches console-api's default", () => {
    // Catches a web window constant that drifts from the backend's DEFAULT_DAYS, so the line names a window the data does not cover.
    const source = readFileSync(DASHBOARD, "utf8");
    const match = source.match(/^DEFAULT_DAYS\b[^=\n]*=\s*(\d+)\s*$/m);
    expect(match).not.toBeNull();
    expect(WINDOW_DAYS).toBe(Number(match?.[1]));
  });

  it("overview and cost name their window under the question", async () => {
    // Catches a tab with no window line, one worded differently (Overview covers today and the last 24 hours, Cost the 30 days and the calendar month), and a line from another tab.
    const expected: Array<[PageName, string]> = [
      ["home", "Today so far (UTC). p95 latency and error rate cover the last 24 hours."],
      ["cost", "Last 30 days. Budgets and this month's spend cover the calendar month so far (UTC)."],
    ];
    for (const [name, sentence] of expected) {
      const match = (await render(name)).match(/<p class="question">[^<]*<\/p><p class="window">([^<]*)<\/p>/);
      expect(match, name).not.toBeNull();
      expect(text(match?.[1] ?? ""), name).toBe(sentence);
      expect(WINDOW_LINES[name as "home" | "cost"], name).toBe(sentence);
    }
  });

  it("quality chart shows its scores and no commit hash", async () => {
    // Catches a quality page that still prints a short sha above each marker, or leaves the chart with no score axis.
    const html = await render("quality");
    const chart = (html.match(/<svg\b[\s\S]*?<\/svg>/g) ?? []).filter((svg) => count(svg, /<circle\b/g) > 0)[0];
    const labels = Array.from(chart.matchAll(/<text\b[^>]*>([\s\S]*?)<\/text>/g), (match) =>
      match[1].replace(/<!--[\s\S]*?-->/g, "").trim(),
    );
    expect(labels).not.toContain("5e2d8c1");
    expect(labels).not.toContain("3f9c2a7");
    // The score axis is fixed at 0% to 100%, whatever the fixture scores are.
    expect(labels).toContain("0%");
    expect(labels).toContain("100%");
  });

  it("daily cost lists newest first", async () => {
    // Catches rows left in console-api's order, a plain reverse, or a day sort with no tenant tie break.
    const row = (day: string, tenant: string) => ({
      day,
      tenant,
      provider: "mistral",
      requests: 1,
      cached_requests: 0,
      cost_usd: 1,
    });
    const body = fixture("cost");
    body.days = [row("2026-09-13", "acme"), row("2026-09-15", "beta"), row("2026-09-15", "alpha"), row("2026-09-14", "acme")];
    body.savings = [];
    const rows = rowTexts(await render("cost", body)).filter((line) => /^2026-09-1\d /.test(line));
    expect(rows.map((line) => line.split(" ").slice(0, 2).join(" "))).toEqual([
      "2026-09-15 alpha",
      "2026-09-15 beta",
      "2026-09-14 acme",
      "2026-09-13 acme",
    ]);
  });

  it("daily cost page 1 opens on today", async () => {
    // Catches a newest-first sort applied only inside each page, so page 1 still opens on the oldest day.
    const body = fixture("cost");
    body.days = Array.from({ length: 45 }, (_, i) => ({
      day: `2026-08-${String((i % 28) + 1).padStart(2, "0")}`,
      tenant: `dt${i}`,
      provider: "mistral",
      requests: 1,
      cached_requests: 0,
      cost_usd: 1,
    }));
    body.savings = [];
    const rows = rowTexts(await render("cost", body)).filter((line) => /\bdt\d+\b/.test(line));
    expect(rows).toHaveLength(20);
    expect(rows[0].startsWith("2026-08-28 ")).toBe(true);
  });

  it("an over budget tenant is marked over", async () => {
    // Catches an over budget tenant drawn like the rest: no "over" word, no over class, or the class put on a tenant at 62%.
    const body = budgetBody([
      { tenant: "nordlicht", cap_usd: 15.0, spent_usd: 15.07, ratio: 1.0047 },
      { tenant: "acme-eu", cap_usd: 50.0, spent_usd: 31.0, ratio: 0.62 },
    ]);
    const html = await render("cost", body);
    expect(text(html)).toContain("nordlicht: $15.07 of $15.00, over");
    expect(text(html)).not.toContain("acme-eu: $31.00 of $50.00, over");
    expect(count(html, OVER_BAR)).toBe(1);
    expect(count(html, /width:\s*100%/g)).toBe(1);
  });

  it("a budget exactly at its cap is not over", async () => {
    // Catches an over test written as ratio >= 1, which marks a tenant that has spent exactly its cap.
    const html = await render("cost", budgetBody([{ tenant: "nordlicht", cap_usd: 15.0, spent_usd: 15.0, ratio: 1.0 }]));
    expect(text(html)).toContain("nordlicht: $15.00 of $15.00");
    expect(text(html)).not.toMatch(/\bover\b/);
    expect(count(html, OVER_BAR)).toBe(0);
    expect(count(html, /width:\s*100%/g)).toBe(1);
  });

  it("a budget with no ratio is not over", async () => {
    // Catches a null ratio coerced to a number or to infinity, which would mark a tenant with no known ratio as over.
    const html = await render("cost", budgetBody([{ tenant: "initech-eu", cap_usd: 20.0, spent_usd: 5.0, ratio: null }]));
    expect(text(html)).toContain("initech-eu: $5.00 of $20.00");
    expect(text(html)).toContain("n/a");
    expect(text(html)).not.toMatch(/\bover\b/);
    expect(count(html, OVER_BAR)).toBe(0);
  });
});
