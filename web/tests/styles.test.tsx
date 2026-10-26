import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { usePathname } from "next/navigation";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { NavLinks } from "@/components/layout/nav-links";
import { NAV } from "@/lib/nav";

vi.mock("next/navigation", () => ({ usePathname: vi.fn() }));

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SRC = join(ROOT, "src");
const APP = join(SRC, "app");
const ROOT_LAYOUT = join(APP, "layout.tsx");
const GLOBALS = join(APP, "globals.css");

const DARK_COLORS = ["--bg", "--fg", "--muted", "--border", "--accent", "--danger"];
const COLOR_WORDS = ["red", "green", "blue", "black", "white", "gray", "grey", "orange", "yellow", "purple", "pink", "silver"];

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  });
}

function read(path: string): string {
  return readFileSync(path, "utf8");
}

function rel(path: string): string {
  return relative(ROOT, path).split(sep).join("/");
}

function stripCssComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

// The text between the brace that opens at or after `from` and its matching close brace.
function braceBody(text: string, from: number): string {
  const open = text.indexOf("{", from);
  if (open === -1) return "";
  let depth = 0;
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === "{") depth += 1;
    if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(open + 1, i);
    }
  }
  return "";
}

function declared(block: string): Set<string> {
  return new Set([...block.matchAll(/(--[\w-]+)\s*:/g)].map((m) => m[1]));
}

const files = walk(SRC);
const pages = files.filter((path) => path.endsWith(`${sep}page.tsx`) && path.startsWith(APP + sep));
const cssFiles = files.filter((path) => path.endsWith(".css"));
const moduleCss = files.filter((path) => path.endsWith(".module.css"));
const codeFiles = files.filter((path) => /\.tsx?$/.test(path));

describe("console styles", () => {
  it("every page loads the shared stylesheet", () => {
    // Catches a root layout that never imports globals.css, or a second layout rendering <html> that would take a page out of the root layout's stylesheet.
    expect(read(ROOT_LAYOUT)).toContain('import "./globals.css"');
    expect(pages.map(rel).sort()).toHaveLength(5);
    for (const page of pages) {
      const chain: string[] = [];
      let dir = dirname(page);
      for (;;) {
        const layout = join(dir, "layout.tsx");
        if (existsSync(layout)) chain.push(layout);
        if (dir === APP) break;
        dir = dirname(dir);
      }
      const htmlLayouts = chain.filter((layout) => read(layout).includes("<html"));
      expect(htmlLayouts.map(rel), rel(page)).toEqual([rel(ROOT_LAYOUT)]);
    }
  });

  it("no other global css", () => {
    // Catches a component or page importing a plain .css file of its own, which would load global rules outside globals.css.
    const offenders = codeFiles
      .filter((path) => path !== ROOT_LAYOUT)
      .flatMap((path) =>
        [...read(path).matchAll(/import\s+(?:[^"';]*?\s+from\s+)?["']([^"']+\.css)["']/g)]
          .map((m) => m[1])
          .filter((spec) => !spec.endsWith(".module.css"))
          .map((spec) => `${rel(path)} -> ${spec}`),
      );
    expect(offenders).toEqual([]);
  });

  it("every css variable is defined", () => {
    // Catches a stylesheet using a var(--name) that :root in globals.css never declares, which the browser would silently drop.
    expect(existsSync(GLOBALS)).toBe(true);
    const globals = stripCssComments(read(GLOBALS));
    const rootAt = globals.search(/:root\s*\{/);
    expect(rootAt).toBeGreaterThan(-1);
    const defined = declared(braceBody(globals, rootAt));

    const used = new Set(
      cssFiles.flatMap((path) => [...stripCssComments(read(path)).matchAll(/var\(\s*(--[\w-]+)/g)].map((m) => m[1])),
    );
    expect(used.size).toBeGreaterThan(0);
    expect([...used].filter((name) => !defined.has(name)).sort()).toEqual([]);
  });

  it("dark mode redefines the colors", () => {
    // Catches a dark-mode block that is missing or leaves one of the six colors at its light value.
    expect(existsSync(GLOBALS)).toBe(true);
    const globals = stripCssComments(read(GLOBALS));
    const darkAt = globals.search(/@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)/);
    expect(darkAt).toBeGreaterThan(-1);
    const dark = declared(braceBody(globals, darkAt));
    expect(DARK_COLORS.filter((name) => !dark.has(name))).toEqual([]);
  });

  it("modules hold no raw color", () => {
    // Catches a CSS module writing a hex, rgb(), hsl() or named color instead of a variable, which would ignore dark mode.
    expect(moduleCss.length).toBeGreaterThan(0);
    const named = new RegExp(`(?<![\\w-])(?:${COLOR_WORDS.join("|")})(?![\\w-])`, "i");
    const patterns = [/#[0-9a-fA-F]{3,8}\b/, /rgb\(/i, /hsl\(/i, named];
    const offenders = moduleCss.flatMap((path) => {
      const css = stripCssComments(read(path));
      return patterns.filter((re) => re.test(css)).map((re) => `${rel(path)} matches ${re.source}`);
    });
    expect(offenders).toEqual([]);
  });

  it("only the budget bar is inline-styled", () => {
    // Catches an inline style left anywhere besides the budget bar, or a bar whose inline style still carries height, border or color.
    const bodies = codeFiles
      .filter((path) => path.endsWith(".tsx"))
      .flatMap((path) => {
        const text = read(path);
        return [...text.matchAll(/style=\{\{/g)].map((m) => braceBody(text, (m.index ?? 0) + "style={".length));
      });
    expect(bodies).toHaveLength(1);
    const body = bodies[0].trim();
    expect(body).toMatch(/^width\s*:/);
    const keys = [...body.matchAll(/(?:^|,)\s*([A-Za-z_$][\w$]*)\s*:/g)].map((m) => m[1]);
    expect(keys).toEqual(["width"]);
  });
});

describe("chart tips", () => {
  // Every plain rule in the sheet, comments stripped, as selector list and body.
  function rules(): Array<{ selectors: string[]; body: string }> {
    const css = stripCssComments(read(GLOBALS));
    return Array.from(css.matchAll(/([^{}]+)\{([^{}]*)\}/g), (match) => ({
      selectors: match[1].split(",").map((selector) => selector.trim()),
      body: match[2],
    }));
  }

  const HIDES = /opacity\s*:\s*0\s*(;|$)|display\s*:\s*none|visibility\s*:\s*hidden/;
  const SHOWS = /opacity\s*:\s*1\s*(;|$)|display\s*:\s*(block|inline)\b|visibility\s*:\s*visible/;

  it("chart tips show on hover and on focus", () => {
    // Catches a tip that is never hidden (the label sits on top of the bars all the time), one shown on hover only (keyboard users get no value), and one shown on focus only (mouse users get none).
    const all = rules();
    const hidden = all.filter((rule) => rule.selectors.includes(".chart-tip") && HIDES.test(rule.body));
    expect(hidden.length).toBeGreaterThan(0);

    const shownBy = (state: RegExp) =>
      all.filter((rule) => rule.selectors.some((selector) => state.test(selector)) && SHOWS.test(rule.body));
    expect(shownBy(/:hover[^,]*\.chart-tip\b/).length, ":hover").toBeGreaterThan(0);
    expect(shownBy(/:focus(?![\w-])[^,]*\.chart-tip\b/).length, ":focus").toBeGreaterThan(0);
  });
});

describe("NavLinks", () => {
  type Link = { href: string; label: string; current: boolean };

  function render(pathname: string | null): Link[] {
    // usePathname is typed `string`, but Next returns null outside the app router.
    vi.mocked(usePathname).mockReturnValue(pathname as string);
    const html = renderToStaticMarkup(<NavLinks />);
    return [...html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g)].map((m) => ({
      href: /\bhref="([^"]*)"/.exec(m[1])?.[1] ?? "",
      label: m[2].replace(/<[^>]*>/g, "").trim(),
      current: /\baria-current="page"/.test(m[1]),
    }));
  }

  it.each([
    // Catches a prefix match (startsWith) that marks Home on every page, or a mark on the wrong link.
    { pathname: "/cost", marked: ["/cost"] },
    // Catches an implementation that special-cases "/" away and leaves Home unmarked on the home page.
    { pathname: "/", marked: ["/"] },
    // Catches a fallback that marks Home (or any link) when the path matches no nav item.
    { pathname: "/nope", marked: [] },
    // Catches a null pathname being treated as "/" or crashing instead of marking nothing.
    { pathname: null, marked: [] },
  ])("pathname $pathname marks $marked", ({ pathname, marked }) => {
    const links = render(pathname);
    expect(links.filter((link) => link.current).map((link) => link.href)).toEqual(marked);
  });

  it("renders the seven NAV links in order", () => {
    // Catches a nav that drops, adds, reorders, relabels or re-addresses a link compared with NAV.
    const links = render("/cost");
    expect(links.map(({ href, label }) => ({ href, label }))).toEqual(NAV.map(({ href, label }) => ({ href, label })));
  });
});
