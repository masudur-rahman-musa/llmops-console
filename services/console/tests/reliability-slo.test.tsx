import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ReliabilityPage from "@/app/(console)/reliability/page";
import { GUARDS } from "@/lib/guards";

type Fixture = Record<string, unknown>;
type SloRow = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));

function fixture(): Fixture {
  return JSON.parse(readFileSync(new URL("./fixtures/reliability.json", import.meta.url), "utf8")) as Fixture;
}

function slos(body: Fixture): SloRow[] {
  return body.slos as SloRow[];
}

function passes(value: unknown): boolean {
  const guard = GUARDS.reliability as (value: unknown) => boolean;
  return guard(value);
}

// Visible text: tags as spaces, entities decoded, whitespace collapsed; the same reading pages.test.tsx uses.
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
    .trim();
}

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

async function render(body: unknown): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return text(renderToStaticMarkup(await ReliabilityPage()));
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

describe("reliability slo guard", () => {
  it("accepts the fixture and rejects a body without slos", () => {
    // Catches a guard that never reads slos, so a gateway predating the budgets renders an empty table as if every SLO were fine.
    expect(passes(fixture())).toBe(true);
    const missing = fixture();
    delete missing.slos;
    expect(passes(missing)).toBe(false);
  });

  it("rejects a row missing budget_remaining", () => {
    // Catches a guard treating an absent budget_remaining like null, which renders "no traffic" for a malformed row.
    const body = fixture();
    delete slos(body)[0].budget_remaining;
    expect(passes(body)).toBe(false);
  });
});

describe("reliability slo table", () => {
  it("renders the three row names", async () => {
    // Catches a table that renders a fixed subset of rows, or none, instead of every row the body carries. Occurrences are
    // compared against the same body with no rows, because the chart title "p95 latency" already contains "latency".
    const html = await render(fixture());
    expect(html).not.toContain("Console data unavailable");
    const noRows = fixture();
    noRows.slos = [];
    const bare = await render(noRows);
    for (const name of ["availability", "latency", "residency"]) {
      expect(occurrences(html, name), name).toBeGreaterThan(occurrences(bare, name));
    }
  });

  it("shows no traffic for a null budget", async () => {
    // Catches a null budget rendered as 0, 0%, a blank cell or the text null, which reads as a spent budget.
    const body = fixture();
    expect(slos(body)[2].budget_remaining).toBeNull();
    expect(await render(body)).toContain("no traffic");
  });

  it("shows overspent for a negative budget", async () => {
    // Catches a negative budget clamped to 0 on the page or shown as a bare negative number with no warning.
    const body = fixture();
    slos(body)[0].budget_remaining = -1.0;
    expect(await render(body)).toContain("overspent");
  });
});
