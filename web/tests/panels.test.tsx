import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import HomePage from "@/app/(console)/page";
import QualityPage from "@/app/(console)/quality/page";
import RequestsPage from "@/app/(console)/requests/page";

type PageName = "home" | "requests" | "cost" | "quality";
type Fixture = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const TRADEOFF = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));
const CONSOLE_DIR = fileURLToPath(new URL("../src/app/(console)", import.meta.url));
const CSS = readFileSync(new URL("../src/app/globals.css", import.meta.url), "utf8");

const PAGES: Record<PageName, () => Promise<ReactElement>> = {
  home: HomePage,
  requests: RequestsPage,
  cost: CostPage,
  quality: QualityPage,
};

const HEADINGS: Record<PageName, string[]> = {
  home: ["Model routing"],
  requests: ["Redactions by kind", "Help desk refusals by reason", "Tokens per model", "Newest requests"],
  cost: ["Budgets this month", "Top users by cost", "Daily cost", "Model tradeoff"],
  quality: ["Needs a look", "Categories", "Taxonomy", "Blocked releases"],
};
const NAMES = Object.keys(PAGES) as PageName[];

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

type Panel = { label: string; start: number; end: number; html: string };

// Sections never nest on these pages, so the first closing tag ends the panel.
function panels(html: string): Panel[] {
  return Array.from(html.matchAll(/<section\b([^>]*\bclass="panel"[^>]*)>([\s\S]*?)<\/section>/g), (match) => ({
    label: match[1].match(/\baria-label="([^"]*)"/)?.[1] ?? "",
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
    html: match[2],
  }));
}

function panelNamed(html: string, label: string): Panel {
  const found = panels(html).find((panel) => panel.label === label);
  expect(found, `panel ${label}`).toBeDefined();
  return found as Panel;
}

function insideAPanel(html: string, index: number): boolean {
  return panels(html).some((panel) => index >= panel.start && index < panel.end);
}

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

type Rule = { selectors: string[]; decls: Map<string, string> };

function rules(css: string): Rule[] {
  const bare = css.replace(/\/\*[\s\S]*?\*\//g, "");
  return Array.from(bare.matchAll(/([^{}]+)\{([^{}]*)\}/g), (match) => ({
    selectors: match[1].split(",").map((s) => s.trim().replace(/\s+/g, " ")),
    decls: new Map(
      match[2]
        .split(";")
        .map((d) => d.trim())
        .filter((d) => d.includes(":"))
        .map((d): [string, string] => {
          const at = d.indexOf(":");
          return [d.slice(0, at).trim(), d.slice(at + 1).trim().replace(/\s+/g, " ")];
        }),
    ),
  }));
}

// Every declaration that applies to exactly this selector, later rules winning.
function effective(selector: string): Map<string, string> {
  const merged = new Map<string, string>();
  for (const rule of rules(CSS)) {
    if (rule.selectors.includes(selector)) for (const [k, v] of rule.decls) merged.set(k, v);
  }
  return merged;
}

function marginSide(margin: string | undefined, side: "top" | "bottom"): string | undefined {
  if (margin === undefined) return undefined;
  const parts = margin.split(" ");
  if (side === "top") return parts[0];
  return parts[parts.length === 1 ? 0 : parts.length === 2 ? 0 : 2];
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

describe("panels", () => {
  it("panel draws a labelled section with a plain heading", async () => {
    // Catches a Panel that drops the aria-label, wraps the title in a div or styled heading, uses another class, or puts the child before the heading.
    const { Panel } = await import("@/components/ui/panel");
    const html = renderToStaticMarkup(
      <Panel title="Taxonomy">
        <p>child</p>
      </Panel>,
    );
    expect(html).toBe('<section class="panel" aria-label="Taxonomy"><h2>Taxonomy</h2><p>child</p></section>');
  });

  it.each(NAMES)("every section heading on %s sits in its own panel", async (name) => {
    // Catches a page that keeps a bare h2 outside a panel, a panel whose label differs from its heading, a heading that is not first in its panel, or a missing or reordered section.
    const html = await render(name);
    const found = panels(html);
    expect(found.map((panel) => panel.label)).toEqual(HEADINGS[name]);
    for (const panel of found) {
      expect(panel.html.startsWith(`<h2>${panel.label}</h2>`), panel.label).toBe(true);
    }
    const allHeadings = Array.from(html.matchAll(/<h2\b[^>]*>([\s\S]*?)<\/h2>/g), (m) => m[1]);
    expect(allHeadings).toEqual(HEADINGS[name]);
  });

  it("there are thirteen panels across the four pages", async () => {
    // Catches a page set that covers only some pages, leaving the total short of the thirteen named sections.
    let total = 0;
    for (const name of NAMES) total += panels(await render(name)).length;
    expect(total).toBe(13);
  });

  it("a section's notes, filters and pager stay inside its panel", async () => {
    // Catches a Panel closed right after its heading so the note, link, filter form or pager spills out below, and a filter form placed in the neighbouring panel.
    const home = await render("home");
    const routing = text(panelNamed(home, "Model routing").html);
    expect(routing).toContain("Policy last loaded");
    expect(routing).toContain("Health details in Grafana");

    const body = fixture("requests");
    body.requests_pages = 3;
    const requests = await render("requests", body);
    const newest = panelNamed(requests, "Newest requests").html;
    expect(newest.match(/<form\b/g)).toHaveLength(1);
    expect(newest).toContain('name="tenant"');
    expect(newest).toContain('name="model"');
    expect(text(newest)).toContain("Slow means above");
    expect(newest).toMatch(/<nav class="pager"/);
    expect(text(newest)).toContain("Page 1 of 3");
    for (const other of panels(requests).filter((panel) => panel.label !== "Newest requests")) {
      expect(other.html, other.label).not.toMatch(/<form\b|class="pager"/);
    }

    const cost = await render("cost");
    const users = panelNamed(cost, "Top users by cost").html;
    const daily = panelNamed(cost, "Daily cost").html;
    expect(users.match(/<form\b/g)).toHaveLength(1);
    expect(users).toContain('name="users_tenant"');
    expect(users).not.toContain('name="tenant"');
    expect(daily.match(/<form\b/g)).toHaveLength(1);
    expect(daily).toContain('name="tenant"');
    expect(daily).not.toContain('name="users_tenant"');
  });

  it.each(NAMES)("number cards and charts on %s stay outside panels", async (name) => {
    // Catches a Panel wrapped around the whole page body, which would swallow the number cards and the self-bordered charts.
    const html = await render(name);
    expect(panels(html).length).toBeGreaterThan(0);

    const cards = html.indexOf('<ul class="cards"');
    expect(cards).toBeGreaterThan(-1);
    expect(insideAPanel(html, cards)).toBe(false);

    const figures = Array.from(html.matchAll(/<figure\b/g), (m) => m.index ?? 0);
    if (name !== "home") expect(figures.length).toBeGreaterThan(0);
    for (const at of figures) expect(insideAPanel(html, at)).toBe(false);
  });

  it("an empty section still shows its panel", async () => {
    // Catches an empty list that drops the whole panel (or its heading) instead of keeping the box with its none-in-this-window line.
    const quality = fixture("quality");
    quality.taxonomy = [];
    const qualityHtml = await render("quality", quality);
    expect(text(panelNamed(qualityHtml, "Taxonomy").html)).toContain("None in this window.");

    const cost = fixture("cost");
    cost.top_users = [];
    const costHtml = await render("cost", cost);
    expect(text(panelNamed(costHtml, "Top users by cost").html)).toContain(
      "No user ids recorded in this window.",
    );
  });

  it("no console page writes its own section heading", () => {
    // Catches a page that goes back to a literal h2, skipping Panel, so its section has no box.
    const offenders = filesUnder(CONSOLE_DIR).filter((file) => readFileSync(file, "utf8").includes("<h2"));
    expect(offenders).toEqual([]);
  });
});

describe("panel and select styles", () => {
  it("the panel rule draws a box with a gap", () => {
    // Catches a panel with no surface colour, no border, no rounded corners, no inner padding, no gap below, or a heading that keeps the global top margin and pushes the box content down.
    const panel = effective(".panel");
    expect(panel.get("background")).toBe("var(--surface)");
    expect(panel.get("border")).toBe("var(--line)");
    expect(panel.get("border-radius")).toBe("var(--radius)");
    expect(panel.get("padding") ?? "").not.toBe("");
    expect(panel.get("padding")).not.toBe("0");
    const bottom = panel.get("margin-bottom") ?? marginSide(panel.get("margin"), "bottom");
    expect(bottom, "bottom margin").toBeDefined();
    expect(bottom).not.toBe("0");

    const heading = effective(".panel h2");
    const top = heading.get("margin-top") ?? marginSide(heading.get("margin"), "top");
    expect(top).toBe("0");
  });

  it("select matches the date input", () => {
    // Catches a select left as the browser default, one with its own padding, border, corner or background, no custom arrow, or an arrow in a fixed colour that ignores dark mode.
    const select = effective("select");
    const input = effective("input");
    expect(select.get("font")).toBe("inherit");
    expect(select.get("appearance")).toBe("none");
    expect(select.get("padding")).toBeDefined();
    expect(select.get("padding")).toBe(input.get("padding"));
    expect(select.get("border")).toBe(input.get("border"));
    expect(select.get("border-radius")).toBe(input.get("border-radius"));
    expect(select.get("background-color")).toBe(input.get("background") ?? input.get("background-color"));
    expect(select.get("background-image") ?? "").toContain("var(--muted)");
  });

  it("no rule takes the focus ring off a select", () => {
    // Catches a select rule (such as select:focus) that sets outline to none or 0, hiding the keyboard ring the other fields show.
    const hiding = rules(CSS).filter(
      (rule) =>
        rule.selectors.some((s) => /(^|[\s>+~])select\b/.test(s)) &&
        /^(none|0)\b/.test(rule.decls.get("outline") ?? "") ,
    );
    expect(hiding).toEqual([]);
    const ring = effective(":focus-visible").get("outline") ?? "";
    expect(ring).toMatch(/^2px solid var\(--accent\)/);
  });
});
