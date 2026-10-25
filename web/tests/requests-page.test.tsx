import { readFileSync } from "node:fs";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RequestsPage from "@/app/(console)/requests/page";
import { GUARDS } from "@/lib/console-api";

type Fixture = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const QUESTION =
  "What did the gateway hide, how many tokens went in and out, how often did the cache answer, and where is each request's trace?";
const TRACE_URL =
  "https://cloud.langfuse.com/project/demo/traces/0af7651916cd43dd8448eb211c80319c";

function fixture(): Fixture {
  return JSON.parse(
    readFileSync(new URL("./fixtures/requests.json", import.meta.url), "utf8"),
  ) as Fixture;
}

async function render(body: unknown = fixture()): Promise<string> {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })),
  );
  return renderToStaticMarkup(await RequestsPage());
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

function rowsOf(html: string): string[] {
  return Array.from(html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (match) => match[1]);
}

// The table row holding the given text, as raw markup so links can be counted.
function rowWith(html: string, needle: string): string {
  const found = rowsOf(html).find((row) => text(row).includes(needle));
  expect(found, needle).toBeDefined();
  return found ?? "";
}

function anchors(html: string): Array<{ href: string; label: string }> {
  return Array.from(html.matchAll(/<a\b[^>]*\bhref="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g), (match) => ({
    href: match[1],
    label: text(match[2]),
  }));
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

describe("requests page", () => {
  it("requests page shows question and counts", async () => {
    // Catches a page without its question, or one that drops a redaction kind, a refusal reason, the guardrail blocks, a model's tokens in or out, or shows the hit rate as a raw ratio.
    const html = await render();
    const flat = text(html);
    expect(flat).toContain(QUESTION);

    const redactions = rowWith(html, "email");
    expect(text(redactions)).toContain("41");
    expect(text(rowWith(html, "iban"))).toContain("7");

    expect(text(rowWith(html, "out_of_scope"))).toContain("12");
    expect(text(rowWith(html, "needs_approval"))).toContain("3");

    expect(flat).toMatch(/guardrail blocks[^0-9]*5\b/i);

    // Input and output stay in separate cells of the model's row, so a swap is caught.
    const haiku = text(rowsOf(html).find((row) => /<td\b/.test(row) && text(row).startsWith("claude-haiku-5-5")) ?? "");
    expect(haiku).toBe("claude-haiku-5-5 120,450 30,210");
    const mini = text(rowsOf(html).find((row) => /<td\b/.test(row) && text(row).startsWith("gpt-5-mini")) ?? "");
    expect(mini).toBe("gpt-5-mini 8,800 2,100");

    expect(flat).toContain("31.25%");
    expect(flat).not.toContain("0.3125");
  });

  it("requests page lists requests with prompt version and trace link", async () => {
    // Catches a request list that drops the prompt version, prints a null one as "null" or blank, or shows no "Open trace" link to the trace url.
    const html = await render();

    const first = rowWith(html, "nordlicht");
    const links = anchors(first).filter((anchor) => anchor.label === "Open trace");
    expect(links).toEqual([{ href: TRACE_URL, label: "Open trace" }]);
    expect(first).toMatch(/rel="noopener noreferrer"/);
    expect(text(first)).toContain("9f2c4be07a1d");
    expect(text(first)).not.toContain("none sent");

    const second = rowWith(html, "none sent");
    expect(text(second)).not.toContain("null");
    expect(text(second)).toContain("cuan");
    expect(text(html).split("none sent").length - 1).toBe(1);
  });

  it("request with no trace url shows no link", async () => {
    // Catches a row that links to a made-up trace address, or shows an empty "Open trace" anchor, when the console-api sent a null trace_url.
    const html = await render();
    const second = rowWith(html, "none sent");
    expect(anchors(second)).toEqual([]);
    expect(text(second)).not.toContain("Open trace");

    const body = fixture();
    for (const row of body.requests as Array<Record<string, unknown>>) row.trace_url = null;
    const bare = await render(body);
    expect(anchors(bare).filter((anchor) => anchor.label === "Open trace")).toEqual([]);
    expect(text(bare)).not.toContain("Open trace");
  });

  it("requests guard refuses a body missing requests", async () => {
    // Catches a guard that does not require the request list, so a body without it renders a half page instead of the Unavailable notice.
    const body = fixture();
    delete body.requests;
    expect(GUARDS.requests(body)).toBe(false);
    expect(GUARDS.requests(fixture())).toBe(true);
    const html = await render(body);
    expect(html).toContain("Console data unavailable");
  });
});
