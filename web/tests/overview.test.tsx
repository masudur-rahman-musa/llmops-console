import { readFileSync } from "node:fs";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OverviewPage from "@/app/(console)/page";
import { GUARDS } from "@/lib/console-api";
import { NAV, safeNext } from "@/lib/nav";

type Fixture = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const GRAFANA_LABEL = "Health details in Grafana";
const GRAFANA_URL = "https://grafana.example.com/d/ops";

function fixture(): Fixture {
  return JSON.parse(readFileSync(new URL("./fixtures/home.json", import.meta.url), "utf8")) as Fixture;
}

async function render(body: unknown = fixture()): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return renderToStaticMarkup(await (OverviewPage as () => Promise<ReactElement>)());
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

type Anchor = { tag: string; href: string | null; label: string };

function anchors(html: string): Anchor[] {
  return Array.from(html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g), (match) => ({
    tag: match[1],
    href: match[1].match(/\bhref="([^"]*)"/)?.[1]?.replace(/&amp;/g, "&") ?? null,
    label: text(match[2]),
  }));
}

function external(html: string): Anchor[] {
  return anchors(html).filter((anchor) => anchor.href !== null && /^(https?:)?\/\//i.test(anchor.href));
}

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("overview", () => {
  it("nav carries four links in order", () => {
    // Catches a nav that keeps a removed tab, keeps the old "Home" label, or puts Requests after Quality.
    expect(NAV).toEqual([
      { href: "/", label: "Overview" },
      { href: "/requests", label: "Requests" },
      { href: "/cost", label: "Cost" },
      { href: "/quality", label: "Quality" },
    ]);
  });

  it("overview shows each alias with its providers in fallback order", async () => {
    // Catches a routing list that sorts the providers, drops an alias or its purpose, or prints the
    // candidates of one alias under another; each alias is read between its own name and the next one's.
    const html = text(await render());
    const names = ["agent-default", "agent-strong", "judge", "embed"];
    // Walk forward from the previous alias, so a word like "embed" inside "local_embed" is not mistaken for a name.
    const starts: number[] = [];
    let cursor = 0;
    for (const name of names) {
      const at = html.indexOf(name, cursor);
      expect(at, name).toBeGreaterThan(-1);
      starts.push(at);
      cursor = at + name.length;
    }
    const sections = starts.map((start, index) => html.slice(start, starts[index + 1] ?? html.length));
    const [agentDefault, agentStrong, judge, embed] = sections;

    expect(agentDefault).toContain("Everyday help desk chat replies on the cheap model.");
    expect(agentDefault.indexOf("anthropic")).toBeGreaterThan(-1);
    expect(agentDefault.indexOf("anthropic")).toBeLessThan(agentDefault.indexOf("openai"));
    expect(agentDefault.indexOf("claude-haiku-5-5")).toBeLessThan(agentDefault.indexOf("gpt-5-mini"));

    expect(agentStrong).toContain("Harder chat turns that need the strong model.");
    expect(agentStrong.indexOf("claude-sonnet-5-5")).toBeGreaterThan(-1);
    expect(agentStrong.indexOf("claude-sonnet-5-5")).toBeLessThan(agentStrong.indexOf("gpt-5"));

    expect(judge).toContain("Grades eval answers against the reference labels.");
    expect(embed).toContain("Turns text into vectors for search and the semantic cache.");
    expect(embed).toContain("local_embed");
    expect(embed).toContain("bge-small-en-v1.5");

    expect(html).toContain("2026-10-26 09:00 UTC");
  });

  it("overview links Grafana when grafana_url is set", async () => {
    // Catches a page that drops the link, labels it differently, points it at another URL, or
    // renders more than one outside link (the Grafana link is the only one on this page).
    const links = external(await render());

    expect(links).toHaveLength(1);
    expect(links[0].href).toBe(GRAFANA_URL);
    expect(links[0].label).toBe(GRAFANA_LABEL);
    expect(links[0].tag).toMatch(/\brel="noopener noreferrer"/);
  });

  it("overview shows no Grafana link when grafana_url is null", async () => {
    // Catches a page that links a null URL (an empty or "null" href) or says nothing when the link
    // is unset, so an operator cannot tell a missing setting from a broken page.
    const body = fixture();
    body.grafana_url = null;
    const html = await render(body);

    expect(anchors(html).filter((anchor) => anchor.label === GRAFANA_LABEL)).toEqual([]);
    expect(external(html)).toEqual([]);
    expect(text(html)).toContain("Grafana link not set");
    expect(text(await render())).not.toContain("Grafana link not set");
  });

  it("overview shows no open incidents card", async () => {
    // Catches a page that still draws the Open incidents card, with a value or as n/a, now that the
    // body no longer carries `open_incidents`.
    const html = text(await render());

    expect(html).not.toContain("Open incidents");
    expect(html).toContain("p95 latency");
  });
});

describe("removed tabs", () => {
  it("safeNext refuses a removed tab", () => {
    // Catches an allowlist that still holds a removed path, which would redirect a sign-in to a 404.
    // The kept paths are asserted first so a safeNext that refuses everything cannot pass.
    for (const kept of ["/", "/requests", "/cost", "/quality"]) {
      expect(safeNext(kept), kept).toBe(kept);
    }
    for (const removed of ["/incidents", "/reliability", "/releases", "/models"]) {
      expect(safeNext(removed), removed).toBe("/");
    }
  });
});

describe("home guard", () => {
  it("home guard refuses a body missing routing", () => {
    // Catches a guard that does not require `routing`, so the Overview would crash on a body from an
    // older console-api. The full fixture is accepted first, so the missing key is the only difference.
    const guard = GUARDS.home as (value: unknown) => boolean;
    expect(guard(fixture())).toBe(true);

    const body = fixture();
    delete body.routing;
    expect(guard(body)).toBe(false);
  });
});
