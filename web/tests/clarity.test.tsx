import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import HomePage from "@/app/(console)/page";
import CostPage from "@/app/(console)/cost/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";

type PageName = "home" | "requests" | "cost" | "quality";
type Fixture = Record<string, unknown>;

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  home: HomePage,
  requests: RequestsPage,
  cost: CostPage,
  quality: QualityPage,
};
const NAMES = Object.keys(PAGES) as PageName[];

const QUESTIONS: Record<PageName, string> = {
  home: "Is the gateway healthy today, and what has it cost?",
  cost: "What does the AI cost, per day, per tenant and per model, and what did caching save?",
  quality: "Does the help desk still answer well enough to ship?",
  requests:
    "What did the gateway hide, how many tokens went in and out, how often did the cache answer, and where is each request's trace?",
};

const CHECK_SCORE_SENTENCE =
  "Check score: the share of help desk conversations where every turn passed all three checks (the right tools in order, no change before the user says yes, no made-up name).";

function fixture(name: PageName): Fixture {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Fixture;
}

async function render(name: PageName, body: unknown = fixture(name)): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return renderToStaticMarkup(await PAGES[name]());
}

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,:;)%])/g, "$1")
    .replace(/\(\s+/g, "(")
    .trim();
}

function has(html: string, expected: string): boolean {
  return text(html).includes(text(expected));
}

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

function tables(html: string): string[] {
  return html.match(/<table\b[\s\S]*?<\/table>/g) ?? [];
}

function headers(table: string): string[] {
  return Array.from(table.matchAll(/<th\b[^>]*>([\s\S]*?)<\/th>/g), (match) => text(match[1]));
}

function rows(table: string): string[] {
  return Array.from(table.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) => text(match[1]));
}

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

// The first table whose header row is exactly these words.
function tableWith(html: string, columns: string[]): string {
  const found = tables(html).filter((table) => JSON.stringify(headers(table)) === JSON.stringify(columns));
  expect(found, columns.join(", ")).toHaveLength(1);
  return found[0];
}

// The line of text printed straight above the table's wrapper.
function descriptionAbove(html: string, table: string): string {
  const at = html.indexOf(table);
  const before = html.slice(0, at).replace(/<div class="table-wrap">\s*$/, "");
  return text((before.match(/<p\b[^>]*>([^<]*)<\/p>\s*$/) ?? [])[1] ?? "");
}

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.stubEnv("CONSOLE_TRADEOFF_PATH", TRADEOFF);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("console pages are clear to a new reader", () => {
  it("each page opens with its question", async () => {
    // Catches a page that prints no question, a reworded one, or one placed away from the title instead of right after the h1.
    for (const name of NAMES) {
      const html = await render(name);
      const match = html.match(/<h1\b[^>]*>[\s\S]*?<\/h1>\s*<(\w+)\b[^>]*class="[^"]*\bquestion\b[^"]*"[^>]*>([\s\S]*?)<\/\1>/);
      expect(match, name).not.toBeNull();
      expect(text(match?.[2] ?? ""), name).toBe(QUESTIONS[name]);
    }
  });

  it("the check score sentence shows wherever the score does", async () => {
    // Catches a page that leaves the check score unexplained, words it differently, or prints the sentence twice; Overview shows only a number card, so it carries none.
    const expected = { home: 0, quality: 1, cost: 1 } as const;
    for (const name of ["home", "quality", "cost"] as const) {
      const html = await render(name);
      expect(occurrences(text(html), text(CHECK_SCORE_SENTENCE)), name).toBe(expected[name]);
    }
  });

  it("cost shows the check score note only beside a measured table", async () => {
    // Catches a note printed under "Not measured yet." where no check score shows (a note about nothing), and one dropped or doubled beside a table that does show it.
    vi.stubEnv("CONSOLE_TRADEOFF_PATH", fileURLToPath(new URL("./fixtures/no-such-tradeoff.json", import.meta.url)));
    const missing = await render("cost");
    expect(has(missing, "Not measured yet.")).toBe(true);
    expect(occurrences(text(missing), text(CHECK_SCORE_SENTENCE))).toBe(0);

    vi.stubEnv("CONSOLE_TRADEOFF_PATH", TRADEOFF);
    const measured = await render("cost");
    expect(has(measured, "Not measured yet.")).toBe(false);
    expect(occurrences(text(measured), text(CHECK_SCORE_SENTENCE))).toBe(1);
  });

  it("no table header is a raw key, and every table has a thead", async () => {
    // Catches a column named by its code key (usd_per_1000_turns) or a table drawn with no header row.
    let seen = 0;
    for (const name of NAMES) {
      for (const table of tables(await render(name))) {
        seen += 1;
        expect(table, name).toContain("<thead>");
        expect(headers(table).length, name).toBeGreaterThan(0);
        for (const header of headers(table)) {
          expect(header, `${name} ${header}`).not.toMatch(/^[a-z0-9]+(_[a-z0-9]+)+$/);
        }
      }
    }
    expect(seen).toBeGreaterThanOrEqual(8);
  });

  it("the tradeoff table names its columns in words", async () => {
    // Catches tradeoff headers left as check_score, judge_score and usd_per_1000_turns.
    const table = tableWith(await render("cost"), ["Option", "Kind", "Check score", "Judge score", "Cost per 1,000 turns"]);
    expect(rows(table)).toHaveLength(4);
  });

  it("quality shows gate scores as percentages beside the baseline", async () => {
    // Catches a score card that drops a score, swaps value and baseline, flips a verdict, loses the judge note, prints the scores as bare numbers with one decimal, still calls the baseline a threshold, or keeps the old gate table.
    const html = await render("quality");
    expect(has(html, "Check score 52.10% Baseline 48.48% passed")).toBe(true);
    expect(has(html, "Judge score 25.00% Baseline 27.40% failed")).toBe(true);
    expect(has(html, "kappa 0.7")).toBe(true);
    const flat = text(html);
    expect(flat.indexOf("Check score 52.10%")).toBeLessThan(flat.indexOf("Judge score 25.00%"));
    expect(tables(html).filter((table) => headers(table).includes("Baseline"))).toEqual([]);
    expect(flat).not.toMatch(/threshold/i);
  });

  it("a missing baseline says baseline not set", async () => {
    // Catches a null baseline shown as 0%, as n/a, or with a pass or fail word the page cannot know, and one that still says Threshold not set.
    const body = fixture("quality");
    const gate = body.gate as Array<Record<string, unknown>>;
    Object.assign(gate[0], { threshold: null, passed: null });
    const html = await render("quality", body);
    expect(has(html, "Check score 52.10% Baseline not set")).toBe(true);
    const card = (html.match(/<li class="card"[^>]*>[\s\S]*?<\/li>/g) ?? []).map(text).find((line) => line.startsWith("Check score"));
    expect(card).toBeDefined();
    expect(card).not.toMatch(/passed|failed/);
    // The judge card still has its baseline, so the notice belongs to the check card alone.
    expect(occurrences(text(html), "Baseline not set")).toBe(1);
    expect(text(html)).not.toMatch(/threshold/i);
  });

  it("categories and taxonomy say what they hold", async () => {
    // Catches tables with no column words and no line above them saying what the rows are.
    const html = await render("quality");
    const categories = tableWith(html, ["Category", "Cases passed"]);
    const taxonomy = tableWith(html, ["Failure tag", "Cases"]);
    expect(descriptionAbove(html, categories).length).toBeGreaterThan(10);
    expect(descriptionAbove(html, taxonomy).length).toBeGreaterThan(10);
  });

  it("no page names residency", async () => {
    // Catches a page, nav entry or source file that still talks about residency, EU data or regions.
    const banned = /residency|Residency|\bEU\b|region/;
    for (const name of NAMES) {
      // Fixture data may say EU in a made-up incident title; only page-owned words are under test.
      const body = JSON.parse(JSON.stringify(fixture(name)).replace(/\bEU\b/g, "main")) as unknown;
      expect(text(await render(name, body)), name).not.toMatch(banned);
    }
    const offenders = filesUnder(SRC).filter((file) => banned.test(readFileSync(file, "utf8")));
    expect(offenders).toEqual([]);
  });
});
