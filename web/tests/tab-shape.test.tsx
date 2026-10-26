import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import HomePage from "@/app/(console)/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";
import { CHECK_SCORE_SENTENCE } from "@/lib/labels";

type PageName = "home" | "requests" | "cost" | "quality";
type Fixture = Record<string, unknown>;
type Card = { label: string; value: string; text: string };

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));
const HINT = "The help desk stops a person approving their own request.";
const UNDERSCORE_WORD = /\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b/;

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  home: HomePage,
  requests: RequestsPage,
  cost: CostPage,
  quality: QualityPage,
};

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

/** The number cards of a page, in order: label, value and the whole card's words. */
function cards(html: string): Card[] {
  const list = html.match(/<ul class="cards">[\s\S]*?<\/ul>/)?.[0] ?? "";
  return Array.from(list.matchAll(/<li class="card"[^>]*>([\s\S]*?)<\/li>/g), (match) => ({
    label: text(match[1].match(/class="card-label"[^>]*>([\s\S]*?)<\/span>/)?.[1] ?? ""),
    value: text(match[1].match(/class="card-value"[^>]*>([\s\S]*?)<\/strong>/)?.[1] ?? ""),
    text: text(match[1]),
  }));
}

function card(html: string, label: string): Card {
  const found = cards(html).find((item) => item.label === label);
  expect(found, label).toBeDefined();
  return found as Card;
}

function tableCells(html: string): string[] {
  return Array.from(html.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g), (match) => text(match[1]));
}

function rowTexts(html: string): string[] {
  return Array.from(html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) => text(match[1]));
}

function anchors(html: string): Array<{ href: string | null; label: string }> {
  return Array.from(html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g), (match) => ({
    href: match[1].match(/\bhref="([^"]*)"/)?.[1] ?? null,
    label: text(match[2]),
  }));
}

/** Every bar of every bar chart: the rects inside, as series colour numbers and heights. */
function bars(html: string): Array<Array<{ series: number; height: number }>> {
  return Array.from(html.matchAll(/<g\b[^>]*class="bar-stack"[^>]*>([\s\S]*?)<\/g>/g), (match) =>
    Array.from(match[1].matchAll(/<rect\b([^>]*)\/?>/g), (rect) => ({
      series: Number(rect[1].match(/\bseries-(\d)\b/)?.[1]),
      height: Number(rect[1].match(/\bheight="([^"]*)"/)?.[1]),
    })),
  );
}

function legend(html: string): string[] {
  return Array.from(html.matchAll(/<li\b[^>]*class="legend-item"[^>]*>([\s\S]*?)<\/li>/g), (match) => text(match[1]));
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

describe("every tab has the same shape", () => {
  it("every tab opens with question, window, cards, then charts, then tables", async () => {
    // Catches a tab that puts its cards after a chart or a table, a chart below the tables, a window line missing or after the cards, or a second row of cards.
    for (const name of ["home", "requests", "cost", "quality"] as const) {
      const html = await render(name);
      const at = (needle: string): number => html.indexOf(needle);
      const order = [at("<h1"), at('class="question"'), at('class="window"'), at('<ul class="cards"')];
      for (const position of order) expect(position, name).toBeGreaterThan(-1);
      expect(order, name).toEqual([...order].sort((a, b) => a - b));
      expect(html.split('<ul class="cards"').length - 1, name).toBe(1);

      const firstTable = at("<table");
      expect(firstTable, name).toBeGreaterThan(order[3]);
      const lastChart = html.lastIndexOf("<figure");
      if (name !== "home") {
        expect(lastChart, name).toBeGreaterThan(order[3]);
        expect(lastChart, name).toBeLessThan(firstTable);
      }
    }
  });

  it("overview cards", async () => {
    // Catches a missing or renamed card, a card value formatted another way, a Last eval card that does not link to Quality, a check score sentence left on Overview, and a failed run shown as passed.
    const html = await render("home");

    expect(cards(html).map((item) => [item.label, item.value])).toEqual([
      ["Requests today", "842"],
      ["Cost today", "$1,234.50"],
      ["p95 latency", "412 ms"],
      ["Error rate", "0.50%"],
      ["Last eval", "93.12% passed"],
    ]);
    const link = anchors(html).filter((anchor) => anchor.href === "/quality" && anchor.label.includes("Last eval"));
    expect(link).toHaveLength(1);
    expect(link[0].label).toContain("93.12% passed");
    expect(text(html)).not.toContain(CHECK_SCORE_SENTENCE);
    expect(text(html)).not.toContain("Check score:");

    const failed = fixture("home");
    failed.last_eval = { ...(failed.last_eval as Fixture), passed: false };
    expect(card(await render("home", failed), "Last eval").value).toBe("93.12% failed");
  });

  it("requests cards", async () => {
    // Catches a Requests card that counts something other than the per day totals (650 is 300 + 250 + 100), values hidden taken from one kind only (41 + 7), the old "guardrail" wording left anywhere, the hint missing, and a missing count shown as 0 instead of n/a.
    const html = await render("requests");

    expect(cards(html).map((item) => [item.label, item.value])).toEqual([
      ["Requests", "650"],
      ["Cache hit rate", "31.25%"],
      ["Values hidden", "48"],
      ["Self-approvals blocked", "5"],
    ]);
    expect(card(html, "Self-approvals blocked").text).toContain(HINT);
    expect(text(html)).not.toMatch(/guardrail/i);

    const unknown = fixture("requests");
    unknown.guardrail_blocks = null;
    unknown.cache_hit_rate = null;
    const nulls = await render("requests", unknown);
    expect(card(nulls, "Self-approvals blocked").value).toBe("n/a");
    expect(card(nulls, "Cache hit rate").value).toBe("n/a");
  });

  it("requests chart stacks answered and failed per day", async () => {
    // Catches a chart with a bar missing or doubled per day, errors drawn as a part when Prometheus gave none (null) or none happened (0), parts in the wrong colour order, and a legend that does not name both series.
    const html = await render("requests");

    const drawn = bars(html);
    expect(drawn.map((bar) => bar.map((part) => part.series))).toEqual([[1, 2], [1], [1]]);
    const names = legend(html);
    expect(names).toHaveLength(2);
    expect(names[0]).toMatch(/answered/i);
    expect(names[1]).toMatch(/failed/i);
    const [first] = drawn;
    expect(first[0].height).toBeGreaterThan(first[1].height);
  });

  it("cost cards and chart", async () => {
    // Catches a month spend that is not the body's total, a Cache saved card read from the rows, a missing per alias card, a part drawn per provider row instead of per tenant, a zero part drawn, and the old Cache savings table or "What one request costs" lines left on the page.
    const row = (day: string, tenant: string, provider: string, cost: number) => ({
      day,
      tenant,
      provider,
      requests: 1,
      cached_requests: 0,
      cost_usd: cost,
    });
    const body = fixture("cost");
    body.days = [
      row("2026-09-13", "acme-eu", "mistral", 10),
      row("2026-09-14", "acme-eu", "mistral", 4),
      row("2026-09-14", "acme-eu", "local", 1),
      row("2026-09-14", "globex-eu", "local", 3),
      row("2026-09-15", "acme-eu", "mistral", 18.25),
      row("2026-09-15", "globex-eu", "local", 0),
    ];
    const html = await render("cost", body);

    expect(cards(html).map((item) => [item.label, item.value])).toEqual([
      ["This month's spend", "$61.00"],
      ["Cache saved", "$4.55"],
      ["support-fast", "$0.0042"],
      ["support-smart", "$0.0187"],
    ]);

    const drawn = bars(html);
    expect(drawn.map((bar) => bar.length).sort()).toEqual([1, 1, 2]);
    expect([...legend(html)].sort()).toEqual(["acme-eu", "globex-eu"]);
    // Day totals 10, 5 + 3 and 18.25 keep their proportions: the providers of one tenant are summed.
    const totals = drawn.map((bar) => bar.reduce((sum, part) => sum + part.height, 0)).sort((a, b) => a - b);
    const scale = totals[2] / 18.25;
    expect(Math.abs(totals[0] - 8 * scale)).toBeLessThanOrEqual(1);
    expect(Math.abs(totals[1] - 10 * scale)).toBeLessThanOrEqual(1);

    const flat = text(html);
    expect(flat).not.toContain("Cache savings");
    expect(flat).not.toContain("Estimated cache savings");
    expect(flat).not.toContain("What one request costs");
    expect(flat).not.toContain("support-fast: $0.0042");
    expect(flat).not.toContain("2026-09-15 acme-eu $3.10");
  });

  it("cost spend shows n/a when the month spend is unknown", async () => {
    // Catches a null month spend drawn as $0.00, which reads as a month with no cost.
    const body = fixture("cost");
    body.month_spend_usd = null;
    expect(card(await render("cost", body), "This month's spend").value).toBe("n/a");
  });

  it("quality cards", async () => {
    // Catches a score beside the wrong baseline, a verdict word taken from the wrong score, a missing baseline shown as a number or with a verdict, the kappa note moved above the cards or dropped, and the old gate table left behind.
    const html = await render("quality");

    expect(cards(html).map((item) => [item.label, item.value])).toEqual([
      ["Check score", "52.10%"],
      ["Judge score", "25.00%"],
    ]);
    expect(card(html, "Check score").text).toContain("Baseline 48.48% passed");
    expect(card(html, "Judge score").text).toContain("27.40% failed");
    const flat = text(html);
    expect(flat.indexOf("kappa 0.7")).toBeGreaterThan(flat.indexOf("Judge score 25.00%"));
    expect(rowTexts(html).filter((line) => line.startsWith("Check score 52.10%"))).toEqual([]);
    expect(flat).not.toMatch(/threshold/i);

    const noBaseline = fixture("quality");
    Object.assign((noBaseline.gate as Array<Fixture>)[0], { threshold: null, passed: null });
    const missing = await render("quality", noBaseline);
    expect(card(missing, "Check score").text).toContain("Baseline not set");
    expect(card(missing, "Check score").text).not.toMatch(/passed|failed/);
    expect(card(missing, "Judge score").text).not.toContain("Baseline not set");
  });

  it("plain labels", async () => {
    // Catches a raw kind or tag left in a table (EMAIL, wrong_tool), a label cased wrong, and any underscore word left in a table cell or card on Requests and Quality.
    const requests = fixture("requests");
    requests.redactions = [
      { kind: "EMAIL", count: 41 },
      { kind: "CARD", count: 7 },
      { kind: "PHONE", count: 3 },
      { kind: "iban", count: 2 },
    ];
    requests.refusals = [
      { reason: "out_of_scope", count: 12 },
      { reason: "needs_approval", count: 3 },
    ];
    const requestsHtml = await render("requests", requests);
    const requestRows = rowTexts(requestsHtml);
    expect(requestRows).toEqual(
      expect.arrayContaining(["Emails 41", "Card numbers 7", "Phone numbers 3", "Bank accounts 2", "Out of scope 12", "Needs approval 3"]),
    );
    for (const raw of ["EMAIL", "CARD", "PHONE", "iban", "out_of_scope", "needs_approval"]) {
      expect(tableCells(requestsHtml), raw).not.toContain(raw);
    }

    const quality = fixture("quality");
    quality.categories = [{ category: "out_of_scope", passed: 3, total: 4 }];
    const qualityHtml = await render("quality", quality);
    expect(rowTexts(qualityHtml)).toEqual(expect.arrayContaining(["Out of scope 3 of 4", "Wrong tool 4", "Missed refusal 2"]));

    // Cost is left out: its tradeoff table shows the option kind (self_hosted) as the file gives it, and the plan names no label for it.
    for (const [name, html] of [["requests", requestsHtml], ["quality", qualityHtml]] as const) {
      for (const cell of [...tableCells(html), ...cards(html).map((item) => item.text)]) {
        expect(cell, name).not.toMatch(UNDERSCORE_WORD);
      }
    }
  });
});
