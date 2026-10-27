import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import HomePage from "@/app/(console)/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";
import { GUARDS, type PageName } from "@/lib/console-api";

import { MESSY } from "./fixtures/messy";
import { findJunk } from "./junk";

type Params = Record<string, string | string[] | undefined>;

const TABS: Array<[PageName, string, (props?: { searchParams: Promise<Params> }) => Promise<ReactElement>]> = [
  ["home", "Overview", HomePage],
  ["requests", "Requests", RequestsPage],
  ["cost", "Cost", CostPage],
  ["quality", "Quality", QualityPage],
];

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", "http://console-api.test");
  vi.stubEnv("CONSOLE_ADMIN_KEY", "adminkey-sentinel-7c41e2d9f0");
  vi.stubEnv("CONSOLE_TRADEOFF_PATH", "");
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("messy page bodies", () => {
  it("each messy body passes its page guard", () => {
    // Catches a guard that refuses a real-looking body (a fraction in a Prometheus count, a null score), which would blank the tab, and keeps the render test below from passing on an Unavailable notice.
    for (const [name] of TABS) {
      expect(GUARDS[name](MESSY[name]), name).toBe(true);
    }
  });

  it.each(TABS)("every tab renders the messy fixture with no junk: %s", async (name, title, Page) => {
    // Catches a page that prints a raw fraction, a bare commit hash, NaN, null or a zero-only row from real-looking values, as the Failed line and the sha labels did.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(MESSY[name]), { status: 200 })),
    );
    let html = "";
    let thrown: unknown = null;
    try {
      html = renderToStaticMarkup(await Page({ searchParams: Promise.resolve({}) }));
    } catch (error) {
      thrown = error;
    }
    // The shared render check may throw first; either way the failure names the bad text.
    expect(thrown === null ? null : String(thrown), name).toBeNull();
    expect(html, name).toContain(`<h1>${title}</h1>`);
    expect(html, name).not.toContain("Console data unavailable");
    expect(findJunk(html), name).toEqual([]);
  });
});
