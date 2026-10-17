import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock, type MockInstance } from "vitest";

import HomePage, { dynamic as homeDynamic } from "@/app/(console)/page";
import CostPage, { dynamic as costDynamic } from "@/app/(console)/cost/page";
import IncidentsPage, { dynamic as incidentsDynamic } from "@/app/(console)/incidents/page";
import QualityPage, { dynamic as qualityDynamic } from "@/app/(console)/quality/page";
import ReleasesPage, { dynamic as releasesDynamic } from "@/app/(console)/releases/page";
import ReliabilityPage, { dynamic as reliabilityDynamic } from "@/app/(console)/reliability/page";
import ResidencyPage, { dynamic as residencyDynamic } from "@/app/(console)/residency/page";

type PageName = "home" | "cost" | "quality" | "residency" | "reliability" | "incidents" | "releases";
type Fixture = Record<string, unknown>;

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  home: HomePage,
  cost: CostPage,
  quality: QualityPage,
  residency: ResidencyPage,
  reliability: ReliabilityPage,
  incidents: IncidentsPage,
  releases: ReleasesPage,
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

function words(html: string, word: string): number {
  return (text(html).match(new RegExp(`(^|[^A-Za-z])${word}(?![A-Za-z])`, "g")) ?? []).length;
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
    const exported = [homeDynamic, costDynamic, qualityDynamic, residencyDynamic, reliabilityDynamic, incidentsDynamic, releasesDynamic];
    expect(exported).toEqual(Array(7).fill("force-dynamic"));
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
    expect(has(home, "Today the gateway handled 842 requests costing $1,234.50.")).toBe(true);
    expect(has(home, "p95 latency: 412 ms")).toBe(true);
    expect(has(home, "Error rate: 0.50%")).toBe(true);
    expect(has(home, "Open incidents: 2")).toBe(true);
    expect(has(home, "Last eval: decision accuracy 93.12%, passed")).toBe(true);
    expect(has(home, "All EU data stayed in the EU: ✓")).toBe(true);
    const failedEval = fixture("home");
    failedEval.last_eval = { ...(failedEval.last_eval as Fixture), passed: false };
    expect(has(await render("home", failedEval), "Last eval: decision accuracy 93.12%, failed")).toBe(true);
    const nullHome = fixture("home");
    nullHome.requests_today = null;
    expect(has(await render("home", nullHome), "Today the gateway handled n/a requests costing $1,234.50.")).toBe(true);

    const cost = await render("cost");
    expect(has(cost, "2026-09-15 acme-eu mistral 412 37 $18.25")).toBe(true);
    expect(has(cost, "2026-09-14 globex-eu local 268 54 $0.00")).toBe(true);
    expect(has(cost, "2026-09-15 acme-eu $3.10")).toBe(true);
    expect(has(cost, "2026-09-14 globex-eu $1.45")).toBe(true);
    expect(has(cost, "Estimated cache savings: $4.55")).toBe(true);
    expect(has(cost, "$31.00 of $50.00")).toBe(true);
    expect(has(cost, "$25.00 of $100.00")).toBe(true);
    expect(has(cost, "$5.00 of $20.00")).toBe(true);
    expect(cost).toMatch(/style="[^"]*width:\s*62%/);
    expect(cost).toMatch(/style="[^"]*width:\s*25%/);
    expect(has(cost, "What one request costs")).toBe(true);
    expect(has(cost, "support-fast: $0.0042")).toBe(true);
    expect(has(cost, "support-smart: $0.0187")).toBe(true);
    expect(cost).not.toContain("Daily rollup not yet populated.");
    // The null ratio is the only difference between these two bodies, so only its n/a can separate them.
    const knownRatio = fixture("cost");
    list(knownRatio, "budgets")[2].ratio = 0.25;
    expect(occurrences(text(cost), "n/a")).toBeGreaterThan(occurrences(text(await render("cost", knownRatio)), "n/a"));

    const tables = cost.match(/<table\b[\s\S]*?<\/table>/g) ?? [];
    const tradeoff = tables.filter((table) => table.includes("usd_per_1000_turns"));
    expect(tradeoff).toHaveLength(1);
    const header = (tradeoff[0].match(/<tr\b[\s\S]*?<\/tr>/) ?? [""])[0];
    const columns = Array.from(header.matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/g), (match) => text(match[1]));
    expect(columns).toEqual([
      "Option",
      "Kind",
      "decision_accuracy",
      "refusal_recall",
      "citation_accuracy",
      "p50_latency_ms",
      "p95_latency_ms",
      "usd_per_1000_turns",
    ]);
    expectInOrder(cost, [
      "Hosted small model",
      "Hosted large model",
      "Local 8B on one GPU",
      "Hosted small model matches the large model within tolerance at a fraction of the cost.",
    ]);
    expect(cost).not.toContain("Not measured yet.");

    const quality = await render("quality");
    expect(has(quality, "Decision accuracy")).toBe(true);
    const qualityCharts = svgs(quality).filter((svg) => count(svg, /<circle\b/g) > 0);
    expect(qualityCharts).toHaveLength(1);
    expect(count(qualityCharts[0], /<circle\b/g)).toBe(3);
    expect(count(qualityCharts[0], /<line[\s/>]/g)).toBe(2);
    expect(has(quality, "refusal 18 of 20")).toBe(true);
    expect(has(quality, "citation 9 of 12")).toBe(true);
    expect(has(quality, "wrong_tool 4")).toBe(true);
    expect(has(quality, "missed_refusal 2")).toBe(true);
    expect(has(quality, "3f9c2a7b1e4d")).toBe(true);
    expect(has(quality, "a1b2c3d4e5f6")).toBe(true);
    expect(has(quality, "decision_accuracy: 0.88 (was 0.93)")).toBe(true);
    expect(has(quality, "refusal_recall: 0.9 (was 0.95)")).toBe(true);
    expect(has(quality, "javascript:alert(1)")).toBe(true);
    const prLinks = anchors(quality).filter((anchor) => anchor.label === "PR");
    expect(prLinks.map((anchor) => anchor.href)).toEqual(["https://github.com/example/helpdesk/pull/41"]);
    expect(quality).not.toContain("No regression recorded.");

    const residency = await render("residency");
    expect(has(residency, "acme-eu eu-west mistral 640")).toBe(true);
    expect(has(residency, "globex-eu eu-central local 215")).toBe(true);
    expect(has(residency, "Violations: 0")).toBe(true);
    expect(has(residency, "acme-eu email 12")).toBe(true);
    expect(has(residency, "globex-eu iban 3")).toBe(true);

    const reliability = await render("reliability");
    expect(has(reliability, "p95 latency")).toBe(true);
    expect(has(reliability, "Error rate")).toBe(true);
    const circleCounts = svgs(reliability)
      .map((svg) => count(svg, /<circle\b/g))
      .filter((circles) => circles > 0)
      .sort();
    expect(circleCounts).toEqual([2, 3]);
    expect(has(reliability, "mistral closed")).toBe(true);
    expect(has(reliability, "local half_open")).toBe(true);
    expect(has(reliability, "acme-eu support-fast local 5")).toBe(true);

    const incidents = await render("incidents");
    expectInOrder(incidents, ["Cache hit rate dropped", "EU provider latency spike", "Budget counter stalled"]);
    for (const expected of [
      "sev3",
      "sev2",
      "sev1",
      "Redis evicted keys under memory pressure.",
      "p95 above two seconds on the primary EU route.",
      "Spend was not debited for forty minutes.",
      "2026-09-14 08:30 UTC",
      "2026-09-14 09:10 UTC",
      "2026-09-16 07:45 UTC",
      "2026-09-10 22:15 UTC",
      "2026-09-11 01:05 UTC",
      "n/a",
    ]) {
      expect(has(incidents, expected), expected).toBe(true);
    }

    const releases = await render("releases");
    expectInOrder(releases, ["9b1c3d5e7f9a", "3f9c2a7b1e4d", "c4d5e6f70819"]);
    for (const expected of ["2026-09-15 10:00 UTC", "2026-09-14 16:20 UTC", "2026-09-13 09:45 UTC", "deployed", "blocked", "javascript:alert(1)"]) {
      expect(has(releases, expected), expected).toBe(true);
    }
    expect(words(releases, "passed")).toBe(1);
    expect(words(releases, "failed")).toBe(1);
    expect(occurrences(text(releases), "no eval")).toBe(1);
    const releaseLinks = anchors(releases).filter((anchor) => anchor.label === "PR");
    expect(releaseLinks.map((anchor) => anchor.href)).toEqual(["https://github.com/example/helpdesk/pull/44"]);
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

  it("home residency line for 0, 3 and null", async () => {
    // Catches a residency line that treats null as zero, shows the tick for a violation, or picks the wrong wording per count.
    const lines = ["All EU data stayed in the EU: ✓", "3 residency violations today.", "Residency status unknown."];
    const rows: Array<[number | null, string]> = [
      [0, lines[0]],
      [3, lines[1]],
      [null, lines[2]],
    ];
    for (const [violations, expected] of rows) {
      const body = fixture("home");
      body.residency_violations_today = violations;
      const html = await render("home", body);
      for (const line of lines) {
        expect(has(html, line), `${String(violations)} ${line}`).toBe(line === expected);
      }
      expect(html.includes("✓"), String(violations)).toBe(violations === 0);
    }
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
      expect(html, label).not.toContain("usd_per_1000_turns");
      expect(has(html, "2026-09-15 acme-eu mistral 412 37 $18.25"), label).toBe(true);
      expect(has(html, "Estimated cache savings: $4.55"), label).toBe(true);
      expect(has(html, "support-fast: $0.0042"), label).toBe(true);
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

  it("blocked with empty regressed renders no regression recorded", async () => {
    // Catches an empty regressed list rendered as nothing, or the notice shown beside real regressions.
    const body = fixture("quality");
    list(body, "blocked")[0].regressed = [];
    const html = await render("quality", body);
    expect(occurrences(text(html), "No regression recorded.")).toBe(1);
    expect(html).not.toContain("decision_accuracy: 0.88");
    expect(has(html, "refusal_recall: 0.9 (was 0.95)")).toBe(true);
  });

  it("reliability renders the slo table above the charts", async () => {
    // Catches the SLO table left off the page, or placed below the latency and error-rate charts it summarises.
    const html = await render("reliability");
    expectInOrder(html, ["availability", "residency", "p95 latency", "Error rate"]);
  });

  it("incidents render the open badge and a safe postmortem link only", async () => {
    // Catches a badge on resolved incidents, a Postmortem link for a javascript: URL, or a link for a null URL.
    const html = await render("incidents");
    expect(words(html, "Open")).toBe(1);
    const postmortems = anchors(html).filter((anchor) => anchor.label === "Postmortem");
    expect(postmortems.map((anchor) => anchor.href)).toEqual(["https://docs.example.com/postmortems/31"]);
    expect(has(html, "javascript:alert(1)")).toBe(true);
    expect(html).not.toMatch(/href="\s*javascript:/i);

    const allOpen = fixture("incidents");
    for (const incident of list(allOpen, "incidents")) incident.open = true;
    expect(words(await render("incidents", allOpen), "Open")).toBe(3);

    const noneOpen = fixture("incidents");
    for (const incident of list(noneOpen, "incidents")) {
      incident.open = false;
      incident.postmortem_url = null;
    }
    const closed = await render("incidents", noneOpen);
    expect(words(closed, "Open")).toBe(0);
    expect(anchors(closed)).toEqual([]);
    expect(closed).not.toContain("javascript:");
  });

  it("releases render no eval when eval_passed is null", async () => {
    // Catches a null eval_passed rendered as failed or passed.
    const body = fixture("releases");
    for (const release of list(body, "releases")) release.eval_passed = null;
    const html = await render("releases", body);
    expect(occurrences(text(html), "no eval")).toBe(3);
    expect(words(html, "passed")).toBe(0);
    expect(words(html, "failed")).toBe(0);
  });

  it("external links carry rel noopener noreferrer", async () => {
    // Catches a PR or Postmortem link rendered without rel, or any javascript: URL placed in an href.
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
    // The quality, incidents and releases fixtures carry one safe link each.
    expect(external).toBeGreaterThanOrEqual(3);
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
