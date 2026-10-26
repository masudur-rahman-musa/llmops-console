import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Pager } from "@/components/ui/pager";
import { PAGE_SIZE, pageCount, parsePage, slicePage } from "@/lib/paging";

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function anchors(html: string): Array<{ href: string; label: string }> {
  return Array.from(html.matchAll(/<a\b[^>]*\bhref="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g), (match) => ({
    href: match[1].replace(/&amp;/g, "&"),
    label: text(match[2]),
  }));
}

function query(href: string): URLSearchParams {
  expect(href.startsWith("?"), href).toBe(true);
  return new URLSearchParams(href.slice(1));
}

describe("paging helpers", () => {
  it("parses a page value", () => {
    // Catches a parser that lets a bad value through (0, negative, text, a fraction), reads the whole array instead of its first element, or ignores the page count.
    expect(PAGE_SIZE).toBe(20);
    expect(parsePage("3")).toBe(3);
    for (const bad of [undefined, "", "0", "-1", "abc", "2.5"]) {
      expect(parsePage(bad), String(bad)).toBe(1);
    }
    expect(parsePage(["2", "5"])).toBe(2);
    expect(parsePage("99", 3)).toBe(3);
    expect(parsePage("2", 3)).toBe(2);
  });

  it("counts pages", () => {
    // Catches a page count of 0 for an empty list, or an off-by-one at the 20 and 21 row edges.
    expect(pageCount(0)).toBe(1);
    expect(pageCount(1)).toBe(1);
    expect(pageCount(20)).toBe(1);
    expect(pageCount(21)).toBe(2);
    expect(pageCount(45)).toBe(3);
  });

  it("slices one page", () => {
    // Catches a slice that starts at the wrong row, uses another page size, or drops the short last page.
    const rows = Array.from({ length: 45 }, (_, index) => index + 1);
    expect(slicePage(rows, 1)).toEqual(rows.slice(0, 20));
    expect(slicePage(rows, 3)).toEqual([41, 42, 43, 44, 45]);
  });
});

describe("pager", () => {
  it("draws page 2 of 3", () => {
    // Catches a pager with the wrong text, links that point at the wrong page, or links that drop the other panel's page from the URL.
    const html = renderToStaticMarkup(
      <Pager name="days_page" page={2} pages={3} params={{ days_page: "2", savings_page: "2" }} />,
    );
    expect(text(html)).toContain("Page 2 of 3");
    const links = anchors(html);
    const prev = links.find((link) => link.label.includes("Prev"));
    const next = links.find((link) => link.label.includes("Next"));
    expect(prev).toBeDefined();
    expect(next).toBeDefined();
    const prevQuery = query(prev?.href ?? "");
    const nextQuery = query(next?.href ?? "");
    expect(prevQuery.getAll("days_page")).toEqual(["1"]);
    expect(nextQuery.getAll("days_page")).toEqual(["3"]);
    expect(prevQuery.getAll("savings_page")).toEqual(["2"]);
    expect(nextQuery.getAll("savings_page")).toEqual(["2"]);
  });

  it("has no Prev link on the first page and no Next link on the last", () => {
    // Catches an edge button that still links (to page 0 or past the end) instead of showing the word as plain text.
    const first = renderToStaticMarkup(<Pager name="days_page" page={1} pages={3} params={{}} />);
    expect(text(first)).toContain("Prev");
    expect(text(first)).toContain("Page 1 of 3");
    expect(anchors(first).map((link) => link.label)).toEqual(["Next"]);

    const last = renderToStaticMarkup(<Pager name="days_page" page={3} pages={3} params={{}} />);
    expect(text(last)).toContain("Next");
    expect(anchors(last).map((link) => link.label)).toEqual(["Prev"]);
  });

  it("is not drawn for one page", () => {
    // Catches a pager drawn under a table that fits on one page (20 rows) or is empty (0 rows).
    for (const total of [20, 0]) {
      const html = renderToStaticMarkup(
        <Pager name="days_page" page={1} pages={pageCount(total)} params={{}} />,
      );
      expect(html, String(total)).toBe("");
    }
  });
});
