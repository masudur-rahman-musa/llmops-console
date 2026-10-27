import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock, type MockInstance } from "vitest";

import HomePage, { dynamic as homeDynamic } from "@/app/(console)/page";
import CostPage, { dynamic as costDynamic } from "@/app/(console)/cost/page";
import QualityPage, { dynamic as qualityDynamic } from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";

type PageName = "home" | "cost" | "quality";
type Fixture = Record<string, unknown>;

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  home: HomePage,
  cost: CostPage,
  quality: QualityPage,
};
const NAMES = Object.keys(PAGES) as PageName[];

function fixture(name: PageName): Fixture {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Fixture;
}

function list(body: Fixture, field: string): Fixture[] {
  return body[field] as Fixture[];
}

let fetchMock: Mock;
let errorSpy: MockInstance;

function serve(respond: () => Response): void {
  fetchMock = vi.fn(async () => respond());
  vi.stubGlobal("fetch", fetchMock);
}

function serveBody(body: unknown): void {
  serve(() => new Response(JSON.stringify(body), { status: 200 }));
}

async function render(name: PageName, body: unknown = fixture(name)): Promise<string> {
  serveBody(body);
  return renderToStaticMarkup(await PAGES[name]());
}

// Visible text: comments dropped, tags as spaces, entities decoded, whitespace collapsed, and no space before
// closing punctuation, so `costing <b>$1.00</b>.` and `costing $1.00.` read the same.
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

type Anchor = { tag: string; href: string | null; label: string };

function anchors(html: string): Anchor[] {
  return Array.from(html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g), (match) => ({
    tag: match[1],
    href: match[1].match(/\bhref="([^"]*)"/)?.[1]?.replace(/&amp;/g, "&") ?? null,
    label: text(match[2]),
  }));
}

function svgs(html: string): string[] {
  return html.match(/<svg\b[\s\S]*?<\/svg>/g) ?? [];
}

function count(html: string, pattern: RegExp): number {
  return (html.match(pattern) ?? []).length;
}

// The words of each number card on a page, label then value (then any hint), in order.
function cardTexts(html: string): string[] {
  const list = html.match(/<ul class="cards">[\s\S]*?<\/ul>/)?.[0] ?? "";
  return Array.from(list.matchAll(/<li class="card"[^>]*>([\s\S]*?)<\/li>/g), (match) => text(match[1]));
}

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

function expectInOrder(html: string, needles: string[]): void {
  const flat = text(html);
  const positions = needles.map((needle) => flat.indexOf(text(needle)));
  for (const [index, position] of positions.entries()) {
    expect(position, needles[index]).toBeGreaterThan(-1);
  }
  expect([...positions].sort((a, b) => a - b)).toEqual(positions);
}

const REASONS: Array<{ reason: string; arrange: () => void }> = [
  {
    reason: "unconfigured",
    arrange: () => {
      vi.stubEnv("CONSOLE_API_URL", undefined);
      serveBody({});
    },
  },
  { reason: "unauthorized", arrange: () => serve(() => new Response("", { status: 401 })) },
  { reason: "unreachable", arrange: () => serve(() => new Response("", { status: 500 })) },
  { reason: "invalid", arrange: () => serve(() => new Response("not json", { status: 200 })) },
];

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.stubEnv("CONSOLE_TRADEOFF_PATH", TRADEOFF);
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("console pages", () => {
  it("every page module exports force-dynamic", () => {
    // Catches a page left statically rendered, which would cache one request's data for every visitor.
    const exported = [homeDynamic, costDynamic, qualityDynamic];
    expect(exported).toEqual(Array(3).fill("force-dynamic"));
  });

  it("each page renders its ok fixture", async () => {
    // Catches a page that fetches the wrong path, drops or misformats a rendered field, or renders a notice on a good body.
    for (const name of NAMES) {
      const html = await render(name);
      expect(fetchMock, name).toHaveBeenCalledTimes(1);
      expect(String(fetchMock.mock.calls[0][0]), name).toBe(`${API}/console/${name}`);
      expect(html, name).not.toContain("Console data unavailable");
    }

    const home = await render("home");
    expect(cardTexts(home)).toEqual([
      "Requests today 842",
      "Cost today $1,234.50",
      "p95 latency 412 ms",
      "Error rate 0.50%",
      "Last eval 93.12% passed",
    ]);
    const failedEval = fixture("home");
    failedEval.last_eval = { ...(failedEval.last_eval as Fixture), passed: false };
    expect(cardTexts(await render("home", failedEval))).toContain("Last eval 93.12% failed");
    const nullHome = fixture("home");
    nullHome.requests_today = null;
    expect(cardTexts(await render("home", nullHome))).toContain("Requests today n/a");

    const cost = await render("cost");
    expect(has(cost, "2026-09-15 acme-eu mistral 412 37 $18.25")).toBe(true);
    expect(has(cost, "2026-09-14 globex-eu local 268 54 $0.00")).toBe(true);
    expect(cardTexts(cost)).toEqual([
      "This month's spend $61.00",
      "Answer cache saved $4.55",
      "Prompt cache saved $0.000120",
      "support-fast $0.0042",
      "support-smart $0.0187",
    ]);
    expect(has(cost, "2026-09-15 acme-eu $3.10")).toBe(false);
    expect(has(cost, "Cache savings")).toBe(false);
    expect(has(cost, "Estimated cache savings")).toBe(false);
    expect(has(cost, "$31.00 of $50.00")).toBe(true);
    expect(has(cost, "$25.00 of $100.00")).toBe(true);
    expect(has(cost, "$5.00 of $20.00")).toBe(true);
    expect(cost).toMatch(/style="[^"]*width:\s*62%/);
    expect(cost).toMatch(/style="[^"]*width:\s*25%/);
    expect(has(cost, "What one request costs")).toBe(false);
    expect(cost).not.toContain("Daily rollup not yet populated.");
    // The null ratio is the only difference between these two bodies, so only its n/a can separate them.
    const knownRatio = fixture("cost");
    list(knownRatio, "budgets")[2].ratio = 0.25;
    expect(occurrences(text(cost), "n/a")).toBeGreaterThan(occurrences(text(await render("cost", knownRatio)), "n/a"));

    const tables = cost.match(/<table\b[\s\S]*?<\/table>/g) ?? [];
    const tradeoff = tables.filter((table) => table.includes("Cost per 1,000 turns"));
    expect(tradeoff).toHaveLength(1);
    const header = (tradeoff[0].match(/<tr\b[\s\S]*?<\/tr>/) ?? [""])[0];
    const columns = Array.from(header.matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/g), (match) => text(match[1]));
    expect(columns).toEqual([
      "Option",
      "Kind",
      "Check score",
      "Judge score",
      "Cost per 1,000 turns",
    ]);
    expectInOrder(cost, [
      "Hosted small model",
      "Hosted large model",
      "Local 8B on one GPU",
      "Hosted small model matches the large model within tolerance at a fraction of the cost.",
    ]);
    expect(cost).not.toContain("Not measured yet.");

    const quality = await render("quality");
    expect(has(quality, "Check score")).toBe(true);
    const qualityCharts = svgs(quality).filter((svg) => count(svg, /<circle\b/g) > 0);
    expect(qualityCharts).toHaveLength(1);
    expect(count(qualityCharts[0], /<circle\b/g)).toBe(3);
    expect(count(qualityCharts[0], /<line[^>]*\bchart-marker\b/g)).toBe(2);
    expect(has(quality, "Refusal 18 of 20")).toBe(true);
    expect(has(quality, "Citation 9 of 12")).toBe(true);
    expect(has(quality, "Wrong tool 4")).toBe(true);
    expect(has(quality, "Missed refusal 2")).toBe(true);
    expect(has(quality, "3f9c2a7b1e4d")).toBe(true);
    expect(has(quality, "a1b2c3d4e5f6")).toBe(true);
    expect(has(quality, "Check score: 0.88 (was 0.93)")).toBe(true);
    expect(has(quality, "Judge score: 0.9 (was 0.95)")).toBe(true);
    expect(has(quality, "javascript:alert(1)")).toBe(true);
    const prLinks = anchors(quality).filter((anchor) => anchor.label === "PR");
    expect(prLinks.map((anchor) => anchor.href)).toEqual(["https://github.com/example/helpdesk/pull/41"]);
    expect(quality).not.toContain("No regression recorded.");
  });

  it("each page renders the unavailable notice for each reason", async () => {
    // Catches a page that throws or shows Next's error page on a failed read, repeats the notice, or names the wrong reason.
    for (const name of NAMES) {
      for (const { reason, arrange } of REASONS) {
        vi.stubEnv("CONSOLE_API_URL", API);
        arrange();
        const html = renderToStaticMarkup(await PAGES[name]());
        const label = `${name} ${reason}`;
        expect(occurrences(text(html), "Console data unavailable"), label).toBe(1);
        expect(has(html, `Console data unavailable (${reason}).`), label).toBe(true);
        expect(html, label).not.toMatch(/Postgres (unavailable|not configured)|Prometheus (unavailable|not configured)/);
        if (reason === "unconfigured") expect(fetchMock, label).not.toHaveBeenCalled();
      }
    }
  });

  it("a source error and unconfigured render their notices", async () => {
    // Catches a page that skips SourceNotice, maps a state to the other source's notice, or shows a notice for ok.
    const notices = ["Postgres unavailable.", "Postgres not configured.", "Prometheus unavailable.", "Prometheus not configured."];
    const rows: Array<[Fixture, string[]]> = [
      [{ postgres: "error", prometheus: "unconfigured" }, ["Postgres unavailable.", "Prometheus not configured."]],
      [{ postgres: "unconfigured", prometheus: "error" }, ["Postgres not configured.", "Prometheus unavailable."]],
      [{ postgres: "ok", prometheus: "ok" }, []],
    ];
    for (const name of NAMES) {
      for (const [sources, expected] of rows) {
        const body = fixture(name);
        body.sources = sources;
        const html = await render(name, body);
        for (const notice of notices) {
          expect(has(html, notice), `${name} ${JSON.stringify(sources)} ${notice}`).toBe(expected.includes(notice));
        }
      }
    }
  });

  it("a null prometheus source shows no notice", async () => {
    // Catches a notice line for a source the page does not have (null read as unconfigured or error, or as "Prometheus not configured."), and a null that also hides the Postgres notice.
    const clean = fixture("cost");
    clean.sources = { postgres: "ok", prometheus: null };
    expect(has(await render("cost", clean), "Prometheus")).toBe(false);

    const down = fixture("cost");
    down.sources = { postgres: "error", prometheus: null };
    const html = await render("cost", down);
    expect(has(html, "Postgres unavailable.")).toBe(true);
    expect(has(html, "Prometheus")).toBe(false);
  });

  it("home shows the stored score once and the gate verdict", async () => {
    // Catches a score scaled by 100 a second time (57.14 shown as 5714.00%), a verdict word taken from anything but `passed`, and a null verdict drawn as failed or passed.
    const home = fixture("home");
    home.last_eval = { ...(home.last_eval as Fixture), check_score: 57.14, passed: true };
    const html = await render("home", home);
    expect(cardTexts(html)).toContain("Last eval 57.14% passed");
    expect(html).not.toContain("5714");

    const noBaseline = fixture("home");
    noBaseline.last_eval = { ...(noBaseline.last_eval as Fixture), check_score: 57.14, passed: null };
    const nullHtml = await render("home", noBaseline);
    const lastEval = cardTexts(nullHtml).find((card) => card.startsWith("Last eval")) ?? "";
    expect(lastEval).toContain("57.14%");
    expect(lastEval).not.toMatch(/passed|failed/);
  });

  it("cost shows savings under a cent", async () => {
    // Catches savings rounded to cents: a total of $0.0000257 must not read $0.00.
    const body = fixture("cost");
    body.savings_usd_total = 0.0000257;
    expect(cardTexts(await render("cost", body))).toContain("Answer cache saved $0.000026");
  });

  it("home renders no eval yet when last_eval is null", async () => {
    // Catches a null last_eval rendered as a failed or n/a eval, or the notice shown when an eval exists.
    const body = fixture("home");
    body.last_eval = null;
    const html = await render("home", body);
    expect(has(html, "No eval yet.")).toBe(true);
    expect(html).not.toContain("Last eval:");
    expect(await render("home")).not.toContain("No eval yet.");
  });

  it("pages show an empty state for a run without check_score", async () => {
    // Catches a null check_score drawn as 0%, as NaN, or as a plotted point: home says n/a and
    // the quality chart has no data.
    const home = fixture("home");
    home.last_eval = { ...(home.last_eval as Fixture), check_score: null };
    const lastEval = cardTexts(await render("home", home)).find((card) => card.startsWith("Last eval")) ?? "";
    expect(lastEval).toContain("n/a");
    expect(lastEval).not.toContain("0.00%");

    const quality = fixture("quality");
    for (const run of list(quality, "runs")) run.metrics = { judge_score: 0.9 };
    const qualityHtml = await render("quality", quality);
    expect(has(qualityHtml, "No data.")).toBe(true);
    expect(svgs(qualityHtml).filter((svg) => count(svg, /<circle\b/g) > 0)).toHaveLength(0);
  });

  it("cost page shows top users by cost", async () => {
    // Catches a cost page that does not draw `top_users` (no table, wrong columns, rows in another order or without the dollar format), one that draws the table before "Budgets this month", and one that shows an empty table instead of the plain empty-state line.
    const html = await render("cost");
    const tables = html.match(/<table\b[\s\S]*?<\/table>/g) ?? [];
    const users = tables.filter((table) => table.includes("<th") && text(table).startsWith("Tenant User Requests Cost"));
    expect(users).toHaveLength(1);
    const header = (users[0].match(/<tr\b[\s\S]*?<\/tr>/) ?? [""])[0];
    const columns = Array.from(header.matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/g), (match) => text(match[1]));
    expect(columns).toEqual(["Tenant", "User", "Requests", "Cost"]);
    expectInOrder(html, [
      "Budgets this month",
      "Top users by cost",
      "acme-eu u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c 120 $12.50",
      "acme-eu u_00000000000000000000000000000002 45 $3.75",
      "globex-eu u_00000000000000000000000000000003 30 $0.50",
    ]);
    expect(has(html, "No user ids recorded in this window.")).toBe(false);

    const empty = fixture("cost");
    empty.top_users = [];
    const emptyHtml = await render("cost", empty);
    expect(has(emptyHtml, "Top users by cost")).toBe(true);
    expect(has(emptyHtml, "No user ids recorded in this window.")).toBe(true);
    expect(has(emptyHtml, "u_8f3a1c0e5b7d4a2f9e6c3b1a0d8f7e6c")).toBe(false);
  });

  it("costs under a cent keep their decimals on the cost tab", async () => {
    // Catches a cost tab that still rounds a top user, a daily row, the total, a chart tip, this month's spend or a budget's spend under a cent to $0.00.
    const body = fixture("cost");
    const row = (tenant: string, cost: number) => ({
      day: "2026-09-15",
      tenant,
      provider: "mistral",
      requests: 1,
      cached_requests: 0,
      cost_usd: cost,
    });
    body.days = [row("acme-eu", 0.000412), row("globex-eu", 0.000412)];
    body.top_users = [{ tenant: "acme-eu", user_id: "u_cuan", requests: 7, cost_usd: 0.001934 }];
    body.month_spend_usd = 0.0042;
    body.budgets = [{ tenant: "acme-eu", cap_usd: 50, spent_usd: 0.0031, ratio: 0.000062 }];
    const html = await render("cost", body);
    expect(has(html, "acme-eu u_cuan 7 $0.001934")).toBe(true);
    expect(has(html, "2026-09-15 acme-eu mistral 1 0 $0.000412")).toBe(true);
    expect(has(html, "Total 2 0 $0.000824")).toBe(true);
    expect(cardTexts(html)).toContain("This month's spend $0.004200");
    expect(has(html, "$0.003100 of $50.00")).toBe(true);

    const tip = fixture("cost");
    tip.days = [row("cuan", 0.001934)];
    expect(has(svgs(await render("cost", tip)).join(" "), "cuan: $0.001934")).toBe(true);
  });

  it("costs of a cent or more and zero keep two decimals", async () => {
    // Catches an over-eager fix that prints six decimals everywhere: a cent, twelve fifty and zero must read $0.01, $12.50 and $0.00.
    const body = fixture("cost");
    const user = (id: string, cost: number) => ({ tenant: "acme-eu", user_id: id, requests: 3, cost_usd: cost });
    body.top_users = [user("u_cent", 0.01), user("u_big", 12.5)];
    body.days = [
      { day: "2026-09-15", tenant: "acme-eu", provider: "local", requests: 2, cached_requests: 0, cost_usd: 0 },
    ];
    const html = await render("cost", body);
    expect(has(html, "acme-eu u_cent 3 $0.01")).toBe(true);
    expect(has(html, "acme-eu u_big 3 $12.50")).toBe(true);
    expect(has(html, "2026-09-15 acme-eu local 2 0 $0.00")).toBe(true);
    expect(text(html)).not.toMatch(/\$0\.010000|\$12\.500000|\$0\.000000/);
  });

  it("cost today under a cent keeps its decimals", async () => {
    // Catches the Overview card still rounding a day's cost under a cent to $0.00.
    const body = fixture("home");
    body.cost_usd_today = 0.0019;
    expect(cardTexts(await render("home", body))).toContain("Cost today $0.001900");
  });

  it("budgets show worst first", async () => {
    // Catches budgets left in console-api's order, sorted lowest first, a null share placed first, or a tie not broken by tenant name.
    const body = fixture("cost");
    const budget = (tenant: string, ratio: number | null) => ({ tenant, cap_usd: 10, spent_usd: 1, ratio });
    body.budgets = [
      budget("low-tn", 0.25),
      budget("none-tn", null),
      budget("over-tn", 1.5),
      budget("tn-b", 0.62),
      budget("tn-a", 0.62),
    ];
    expectInOrder(await render("cost", body), ["over-tn:", "tn-a:", "tn-b:", "low-tn:", "none-tn:"]);
  });

  it("cost renders rollup not populated", async () => {
    // Catches a false rollup_populated rendered as an empty table with no explanation.
    const body = fixture("cost");
    body.rollup_populated = false;
    body.days = [];
    expect(has(await render("cost", body), "Daily rollup not yet populated.")).toBe(true);
    expect(await render("cost")).not.toContain("Daily rollup not yet populated.");
  });

  it("cost renders not measured yet and the rest of the page", async () => {
    // Catches a missing or broken tradeoff file that blanks or crashes the whole cost page.
    for (const path of [undefined, "", join(ROOT, "tests", "fixtures", "absent-tradeoff.json")]) {
      vi.stubEnv("CONSOLE_TRADEOFF_PATH", path);
      const html = await render("cost");
      const label = String(path);
      expect(has(html, "Not measured yet."), label).toBe(true);
      expect(html, label).not.toContain("Cost per 1,000 turns");
      expect(has(html, "2026-09-15 acme-eu mistral 412 37 $18.25"), label).toBe(true);
      expect(cardTexts(html), label).toContain("Answer cache saved $4.55");
      expect(cardTexts(html), label).toContain("support-fast $0.0042");
    }
  });

  it("a budget ratio above 1 caps the bar at 100 percent", async () => {
    // Catches a bar width taken from the raw ratio, which draws past its track at 150%.
    const ok = await render("cost");
    const body = fixture("cost");
    Object.assign(list(body, "budgets")[0], { spent_usd: 75.0, ratio: 1.5 });
    const over = await render("cost", body);
    expect(has(over, "$75.00 of $50.00")).toBe(true);
    expect(over).not.toMatch(/width:\s*150%/);
    expect(count(over, /width:\s*100%/g) - count(ok, /width:\s*100%/g)).toBe(1);
  });

  it("quality markers sit at their own runs", async () => {
    // Catches a marker drawn at its deploy time (stretching the axis past the runs) instead of at the run that let the release through.
    const html = await render("quality");
    const chart = svgs(html).filter((svg) => count(svg, /<circle\b/g) > 0)[0];
    const circles = Array.from(chart.matchAll(/<circle\b[^>]*\bcx="([^"]*)"/g), (m) => Number(m[1]));
    const lines = Array.from(chart.matchAll(/<line\b[^>]*\bx1="([^"]*)"[^>]*\bchart-marker\b/g), (m) => Number(m[1]));
    // Runs by time: 09-12, 09-14, 09-15. Markers name the 09-14 and 09-15 runs.
    expect(circles).toHaveLength(3);
    expect(circles[0]).toBe(48);
    expect(circles[2]).toBe(592);
    expect(lines).toEqual([circles[1], circles[2]]);
  });

  it("quality drops a marker whose run is not in the window", async () => {
    // Catches a marker still drawn at its deploy time when it names no run or a run the window does not hold.
    const body = fixture("quality");
    const markers = list(body, "release_markers");
    markers[0].eval_run_id = null;
    markers[1].eval_run_id = "run-not-in-window";
    const html = await render("quality", body);
    const chart = svgs(html).filter((svg) => count(svg, /<circle\b/g) > 0)[0];
    expect(count(chart, /<line[^>]*\bchart-marker\b/g)).toBe(0);
    expect(count(chart, /<circle\b/g)).toBe(3);
  });

  it("empty panels say none in this window", async () => {
    // Catches an empty refusals or taxonomy table drawn as a bare header, and an empty Blocked releases drawn as a bare heading.
    const section = (html: string, heading: string): string => {
      const start = html.indexOf(`<h2>${heading}</h2>`);
      expect(start, heading).toBeGreaterThan(-1);
      const next = html.indexOf("<h2", start + 1);
      return text(html.slice(start, next === -1 ? undefined : next));
    };
    const reqBody = JSON.parse(readFileSync(new URL("./fixtures/requests.json", import.meta.url), "utf8")) as Fixture;
    reqBody.refusals = [];
    serveBody(reqBody);
    const requestsHtml = renderToStaticMarkup(await RequestsPage());
    expect(section(requestsHtml, "Help desk refusals by reason")).toContain("None in this window.");

    const body = fixture("quality");
    body.taxonomy = [];
    body.blocked = [];
    const html = await render("quality", body);
    expect(section(html, "Taxonomy")).toContain("None in this window.");
    expect(section(html, "Blocked releases")).toContain("None in this window.");
    expect(section(html, "Categories")).not.toContain("None in this window.");
  });

  it("blocked with empty regressed renders no regression recorded", async () => {
    // Catches an empty regressed list rendered as nothing, or the notice shown beside real regressions.
    const body = fixture("quality");
    list(body, "blocked")[0].regressed = [];
    const html = await render("quality", body);
    expect(occurrences(text(html), "No regression recorded.")).toBe(1);
    expect(html).not.toContain("Check score: 0.88");
    expect(has(html, "Judge score: 0.9 (was 0.95)")).toBe(true);
  });

  it("external links carry rel noopener noreferrer", async () => {
    // Catches a PR or Grafana link rendered without rel, or any javascript: URL placed in an href.
    let external = 0;
    for (const name of NAMES) {
      const html = await render(name);
      expect(html, name).not.toMatch(/href="\s*javascript:/i);
      for (const anchor of anchors(html)) {
        if (anchor.href !== null && /^(https?:)?\/\//i.test(anchor.href)) {
          external += 1;
          expect(anchor.tag, `${name} ${anchor.href}`).toMatch(/\brel="noopener noreferrer"/);
        }
      }
    }
    // The quality fixture carries one safe link and the home fixture the Grafana link.
    expect(external).toBeGreaterThanOrEqual(2);
  });

  it("rendered html never contains the admin key", async () => {
    // Catches a page or client that echoes the key into markup or a log line on any path, ok or failed.
    for (const name of NAMES) {
      const htmls = [await render(name)];
      for (const { arrange } of REASONS) {
        vi.stubEnv("CONSOLE_API_URL", API);
        arrange();
        htmls.push(renderToStaticMarkup(await PAGES[name]()));
      }
      for (const html of htmls) {
        expect(html, name).not.toContain(KEY);
      }
    }
    const logged = errorSpy.mock.calls
      .map((call) => call.map((arg) => (typeof arg === "string" ? arg : `${String(arg)} ${JSON.stringify(arg)}`)).join(" "))
      .join("\n");
    expect(errorSpy).toHaveBeenCalled();
    expect(logged).not.toContain(KEY);
  });

  it("no use client directive under the console pages or components", () => {
    // Catches a page or component shipped as client JavaScript; this file sits outside src, so it cannot match itself.
    // Two client parts are allowed: the nav's current-page mark, which needs usePathname,
    // and the console error page, because Next.js only lets a client part catch a page error.
    const allowed = [
      join(ROOT, "src", "components", "layout", "nav-links.tsx"),
      join(ROOT, "src", "app", "(console)", "error.tsx"),
    ];
    const files = [...filesUnder(join(ROOT, "src", "app", "(console)")), ...filesUnder(join(ROOT, "src", "components"))];
    expect(files.length).toBeGreaterThan(0);
    const offenders = files.filter((file) => !allowed.includes(file) && /["']use client["']/.test(readFileSync(file, "utf8")));
    expect(offenders).toEqual([]);
  });
});
