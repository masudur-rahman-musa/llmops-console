import { readFileSync } from "node:fs";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ModelsPage from "@/app/(console)/models/page";
import { GUARDS } from "@/lib/console-api";

type Fixture = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const QUESTION =
  "Which models does each alias use, in what order, and what is loaded right now?";

function fixture(): Fixture {
  return JSON.parse(
    readFileSync(new URL("./fixtures/models.json", import.meta.url), "utf8"),
  ) as Fixture;
}

async function render(body: unknown = fixture()): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return renderToStaticMarkup(await ModelsPage());
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

function rows(html: string): string[] {
  return Array.from(html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) =>
    text(match[1]),
  );
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

describe("models page", () => {
  it("models page shows question, aliases and fallback order", async () => {
    // Catches a page without its question, one that drops an alias or purpose, or one that lists providers sorted or out of fallback order.
    const html = await render();
    expect(text(html)).toContain(QUESTION);
    const body = fixture();
    const aliases = body.aliases as Array<{ name: string; purpose: string }>;
    const lines = rows(html);
    for (const alias of aliases) {
      const line = lines.find((row) => row.includes(alias.name));
      expect(line, alias.name).toBeDefined();
      expect(line, alias.name).toContain(alias.purpose);
    }
    const flat = text(html);
    const names = aliases.map((alias) => flat.indexOf(alias.name));
    expect([...names].sort((a, b) => a - b)).toEqual(names);
    expect(lines.find((row) => row.includes("agent-default"))).toContain(
      "1. anthropic claude-haiku-5-5, 2. openai gpt-5-mini",
    );
    expect(lines.find((row) => row.includes("agent-strong"))).toContain(
      "1. anthropic claude-sonnet-5-5, 2. openai gpt-5",
    );
    expect(lines.find((row) => row.startsWith("embed "))).toContain(
      "1. local_embed bge-small-en-v1.5",
    );
  });

  it("models page shows loaded files and time", async () => {
    // Catches a page that drops the provider files, hides their kind or enabled state, or prints loaded_at raw or not at all.
    const html = await render();
    const lines = rows(html);
    const anthropic = lines.find((row) => row.includes("anthropic.yaml"));
    expect(anthropic).toContain("anthropic");
    expect(anthropic).toMatch(/\benabled\b/);
    const local = lines.find((row) => row.includes("local_embed.yaml"));
    expect(local).toContain("openai_compatible");
    const openai = lines.find((row) => row.includes("openai.yaml"));
    expect(openai).toMatch(/\bdisabled\b/);
    expect(text(html)).toContain("2026-10-26 09:00 UTC");
  });

  it("models page shows no source notice", async () => {
    // Catches a page that still prints a Postgres or Prometheus notice for a body that has no sources.
    const html = await render();
    expect(html).not.toContain('role="status"');
    expect(html).not.toContain("Console data unavailable");
  });

  it("alias with no purpose reads no purpose given", async () => {
    // Catches a null purpose printed as "null", left blank, or crashing the page.
    const body = fixture();
    (body.aliases as Array<Record<string, unknown>>)[1].purpose = null;
    const html = await render(body);
    const line = rows(html).find((row) => row.includes("agent-strong"));
    expect(line).toContain("no purpose given");
    expect(line).not.toContain("null");
    expect(text(html).split("no purpose given").length - 1).toBe(1);
  });

  it("models guard refuses a body missing loaded_at", async () => {
    // Catches a guard that does not require loaded_at, so a body without it renders a page with a blank time.
    const body = fixture();
    delete body.loaded_at;
    expect(GUARDS.models(body)).toBe(false);
    expect(GUARDS.models(fixture())).toBe(true);
    const html = await render(body);
    expect(html).toContain("Console data unavailable");
  });
});
