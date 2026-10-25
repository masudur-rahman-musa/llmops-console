import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Table } from "@/components/ui/table";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SRC = join(ROOT, "src");
const TABLE = join(SRC, "components", "ui", "table.tsx");
const WRAPPER = join(SRC, "components", "layout", "console-page.tsx");
const README = join(ROOT, "README.md");

const PAGE_NAMES = ["", "cost", "quality", "reliability", "incidents", "releases"] as const;
const PAGES = PAGE_NAMES.map((name) => join(SRC, "app", "(console)", name, "page.tsx"));

function read(path: string): string {
  return readFileSync(path, "utf8");
}

// Every `.tsx` and `.ts` file under src, read as text, so a CONSOLE_* name can be searched anywhere.
function srcText(): string {
  const walk = (dir: string): string[] =>
    readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const path = join(dir, entry.name);
      return entry.isDirectory() ? walk(path) : [path];
    });
  return walk(SRC)
    .filter((path) => /\.tsx?$/.test(path))
    .map((path) => read(path))
    .join("\n");
}

function cells(row: string): string[] {
  return Array.from(row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g), (match) => match[1]);
}

function headers(html: string): string[] {
  return Array.from(html.matchAll(/<th\b[^>]*>([\s\S]*?)<\/th>/g), (match) => match[1]);
}

describe("Table part", () => {
  it("table with columns renders the table-wrap thead tbody shape", () => {
    // Catches a Table that drops the wrap, skips the header row, loses a column, or reorders cells within a row.
    const html = renderToStaticMarkup(
      <Table
        columns={["Day", "Tenant", "Cost"]}
        rows={[
          { key: "a", cells: ["2026-09-15", "acme", "$1.00"] },
          { key: "b", cells: ["2026-09-14", "globex", "$2.00"] },
        ]}
      />,
    );
    expect((html.match(/class="table-wrap"/g) ?? []).length).toBe(1);
    expect((html.match(/<thead\b/g) ?? []).length).toBe(1);
    expect(headers(html)).toEqual(["Day", "Tenant", "Cost"]);

    const body = html.match(/<tbody\b[^>]*>([\s\S]*?)<\/tbody>/)?.[1] ?? "";
    const bodyRows = body.match(/<tr\b[\s\S]*?<\/tr>/g) ?? [];
    expect(bodyRows.map(cells)).toEqual([
      ["2026-09-15", "acme", "$1.00"],
      ["2026-09-14", "globex", "$2.00"],
    ]);
  });

  it("table without columns renders no thead", () => {
    // Catches a Table that always draws a header row, so a body-only table would show an empty or stray <thead>.
    const html = renderToStaticMarkup(
      <Table
        rows={[
          { key: "a", cells: ["2026-09-15", "$1.00"] },
          { key: "b", cells: ["2026-09-14", "$2.00"] },
        ]}
      />,
    );
    expect(html).toContain("<tbody");
    expect(html).not.toContain("<thead");
  });
});

describe("console pages use the shared parts", () => {
  it("every console page routes its tables through the table part", () => {
    // Catches a page still hand-writing its own <table> markup instead of calling the shared Table part.
    for (const page of PAGES) {
      const source = read(page);
      expect(source, page).not.toContain("table-wrap");
      expect(source, page).not.toContain("<table");
    }
    const table = read(TABLE);
    expect(table).toContain("table-wrap");
    expect(table).toContain("<table");
  });

  it("every console page opens through the wrapper", () => {
    // Catches a page still importing Unavailable or calling getConsolePage itself instead of going through consolePage.
    for (const page of PAGES) {
      const source = read(page);
      expect(source, page).not.toContain("Unavailable");
      expect(source, page).not.toContain("getConsolePage");
    }
    const wrapper = read(WRAPPER);
    expect(wrapper).toContain("getConsolePage");
    expect(wrapper).toContain("Unavailable");
  });
});

describe("README", () => {
  it("the readme names only real commands and settings", () => {
    // Catches a README that prints a pnpm script or CONSOLE_* setting that no longer exists in package.json or src.
    expect(existsSync(README)).toBe(true);
    const readme = read(README);

    const scripts = Object.keys(
      (JSON.parse(read(join(ROOT, "package.json"))) as { scripts: Record<string, string> }).scripts,
    );
    const named = Array.from(readme.matchAll(/\bpnpm\s+(?:-C\s+\S+\s+)?([a-z][a-z0-9-]*)\b/g), (match) => match[1]);
    expect(named.length).toBeGreaterThan(0);
    for (const script of named) {
      expect(scripts, script).toContain(script);
    }

    const source = srcText();
    const envNames = Array.from(new Set(Array.from(readme.matchAll(/CONSOLE_[A-Z0-9_]+/g), (match) => match[0])));
    expect(envNames.length).toBeGreaterThan(0);
    for (const name of envNames) {
      expect(source.includes(name), name).toBe(true);
    }
  });
});
