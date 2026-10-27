import { readFileSync } from "node:fs";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import RequestsPage from "@/app/(console)/requests/page";
import { GUARDS } from "@/lib/console-api";

type PageName = "requests" | "cost";
type Fixture = Record<string, unknown>;
type Card = { label: string; value: string; text: string };

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  requests: RequestsPage,
  cost: CostPage,
};
const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";

const ANSWER_HINT = "Answered by the gateway's own cache, with no model call.";
const PROMPT_HINT =
  "Input tokens the provider read from its cache, billed cheaper.";
const COST_NOTE =
  "Answer cache saved is estimated: what the answers served from the gateway's cache would have cost. Prompt cache saved is what the input read from the provider's cache would have cost at the full input rate, less what it was billed and less the extra paid to write it. Each alias card is the average cost of one request.";

function fixture(name: PageName): Fixture {
  return JSON.parse(
    readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8"),
  ) as Fixture;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

async function render(
  name: PageName,
  body: unknown = fixture(name),
): Promise<string> {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
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

function cards(html: string): Card[] {
  const list = html.match(/<ul class="cards">[\s\S]*?<\/ul>/)?.[0] ?? "";
  return Array.from(
    list.matchAll(/<li class="card"[^>]*>([\s\S]*?)<\/li>/g),
    (match) => ({
      label: text(
        match[1].match(/class="card-label"[^>]*>([\s\S]*?)<\/span>/)?.[1] ?? "",
      ),
      value: text(
        match[1].match(/class="card-value"[^>]*>([\s\S]*?)<\/strong>/)?.[1] ??
          "",
      ),
      text: text(match[1]),
    }),
  );
}

function card(html: string, label: string): Card {
  const found = cards(html).find((item) => item.label === label);
  expect(found, label).toBeDefined();
  return found as Card;
}

function passes(name: PageName, value: unknown): boolean {
  return (GUARDS[name] as (value: unknown) => boolean)(value);
}

describe("prompt cache on Requests", () => {
  it("requests shows both caches by name", async () => {
    // Catches a page that keeps the old "Cache hit rate" name, puts the new card anywhere but third, shows the share unrounded or as a ratio, or leaves either hint off its card.
    const html = await render("requests");

    expect(cards(html).map((item) => item.label)).toEqual([
      "Requests",
      "Answer cache hit rate",
      "Prompt cache share",
      "Values hidden",
      "Self-approvals blocked",
    ]);
    expect(card(html, "Prompt cache share").value).toBe("42.13%");
    expect(card(html, "Answer cache hit rate").text).toContain(ANSWER_HINT);
    expect(card(html, "Prompt cache share").text).toContain(PROMPT_HINT);
  });

  it("a null share reads n/a", async () => {
    // Catches a null share drawn as 0% or "NaN%": no traffic is not a zero share.
    const body = fixture("requests");
    body.prompt_cache_share = null;

    expect(
      cards(await render("requests", body)).map(
        (item) => `${item.label} ${item.value}`,
      ),
    ).toContain("Prompt cache share n/a");
  });
});

describe("prompt cache on Cost", () => {
  it("cost shows both savings by name", async () => {
    // Catches a page that keeps the old "Cache saved" name, orders the new card anywhere but after the answer cache one, or formats a sub-cent saving as $0.00 or a large one without cents.
    const html = await render("cost");

    expect(cards(html).map((item) => item.label)).toEqual([
      "This month's spend",
      "Answer cache saved",
      "Prompt cache saved",
      "support-fast",
      "support-smart",
    ]);
    expect(card(html, "Prompt cache saved").text).toBe(
      "Prompt cache saved $0.000120",
    );
    const big = fixture("cost");
    big.prompt_cache_saved_usd = 12.5;
    expect(card(await render("cost", big), "Prompt cache saved").value).toBe(
      "$12.50",
    );
  });

  it("a loss and a null saving show as they are", async () => {
    // Catches a saving clamped to $0 when cache writes cost more than reads saved, an abs() that hides the sign, and a null drawn as $0.
    const loss = fixture("cost");
    loss.prompt_cache_saved_usd = -0.00012;
    expect(card(await render("cost", loss), "Prompt cache saved").value).toBe(
      "$-0.000120",
    );

    const unknown = fixture("cost");
    unknown.prompt_cache_saved_usd = null;
    expect(
      card(await render("cost", unknown), "Prompt cache saved").value,
    ).toBe("n/a");
  });

  it("the cost note says how both savings are worked out", async () => {
    // Catches a note that still explains only the answer cache, or words the prompt saving without the write cost.
    expect(text(await render("cost"))).toContain(COST_NOTE);
  });
});

describe("the new fields in the guards", () => {
  it("guards refuse a bad new field", () => {
    // Catches a guard that does not check the new field: a body missing it, or carrying text, would reach the page and render as NaN.
    const cases: Array<[PageName, string]> = [
      ["requests", "prompt_cache_share"],
      ["cost", "prompt_cache_saved_usd"],
    ];
    for (const [name, field] of cases) {
      expect(passes(name, fixture(name)), `${name} fixture`).toBe(true);
      const missing = fixture(name);
      delete missing[field];
      expect(passes(name, missing), `${field} missing`).toBe(false);
      const text = fixture(name);
      text[field] = "0.4";
      expect(passes(name, text), `${field} text`).toBe(false);
      const none = fixture(name);
      none[field] = null;
      expect(passes(name, none), `${field} null`).toBe(true);
    }
  });
});
