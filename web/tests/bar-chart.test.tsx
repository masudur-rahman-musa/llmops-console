import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { BarChart } from "@/components/ui/bar-chart";

type Day = { label: string; parts: Array<{ series: string; value: number }> };

const GLOBALS = fileURLToPath(new URL("../src/app/globals.css", import.meta.url));
const BAR = /<g\b[^>]*class="bar-stack"[^>]*>([\s\S]*?)<\/g>/g;

function render(days: Day[], series: string[], title = "Requests per day"): string {
  return renderToStaticMarkup(<BarChart title={title} days={days} series={series} />);
}

/** The inside of every bar, in order. */
function bars(html: string): string[] {
  return Array.from(html.matchAll(BAR), (match) => match[1]);
}

/** Every rect of one bar: its height in px and its series colour number. */
function rects(bar: string): Array<{ height: number; series: number }> {
  return Array.from(bar.matchAll(/<rect\b([^>]*)\/?>/g), (match) => ({
    height: Number(match[1].match(/\bheight="([^"]*)"/)?.[1]),
    series: Number(match[1].match(/\bseries-(\d)\b/)?.[1]),
  }));
}

function total(bar: string): number {
  return rects(bar).reduce((sum, rect) => sum + rect.height, 0);
}

function legend(html: string): string[] {
  return Array.from(html.matchAll(/<li\b[^>]*class="legend-item"[^>]*>([\s\S]*?)<\/li>/g), (match) =>
    match[1].replace(/<!--[\s\S]*?-->/g, "").replace(/<[^>]*>/g, "").trim(),
  );
}

describe("BarChart", () => {
  it("parts stack to the bar total", () => {
    // Catches parts scaled each on their own (a bar of 30 + 20 drawn taller or shorter than a bar of 50), a part drawn at the wrong height, a zero part drawn as a sliver, and a chart that leaves the title out.
    const html = render(
      [
        { label: "Oct 24", parts: [{ series: "Answered", value: 30 }, { series: "Failed", value: 20 }] },
        { label: "Oct 25", parts: [{ series: "Answered", value: 50 }, { series: "Failed", value: 0 }] },
        { label: "Oct 26", parts: [{ series: "Answered", value: 25 }, { series: "Failed", value: 0 }] },
      ],
      ["Answered", "Failed"],
    );
    const [split, whole, half] = bars(html);

    expect(html).toContain("Requests per day");
    expect(rects(split)).toHaveLength(2);
    expect(rects(whole)).toHaveLength(1);
    // Within one px, so a part height rounded to whole px still passes.
    expect(Math.abs(total(split) - total(whole))).toBeLessThanOrEqual(1);
    expect(Math.abs(total(half) - total(whole) / 2)).toBeLessThanOrEqual(1);
    const [answered, failed] = rects(split);
    expect(answered.height / failed.height).toBeCloseTo(30 / 20, 0);
    expect(answered.height).toBeGreaterThan(0);
  });

  it("one legend entry per series, colours cycle past four", () => {
    // Catches a legend that lists only the series with data or lists one twice, a fifth series drawn in a colour that does not exist (series-5), and colours that do not follow the series order.
    const names = ["a", "b", "c", "d", "e"];
    const html = render(
      [{ label: "Oct 24", parts: names.map((series, index) => ({ series, value: index + 1 })) }],
      names,
    );

    expect(legend(html)).toEqual(names);
    expect(rects(bars(html)[0]).map((rect) => rect.series)).toEqual([1, 2, 3, 4, 1]);
  });

  it("a series with no data still gets its legend entry and keeps its colour", () => {
    // Catches colours assigned by position among the drawn parts: with the first series at zero, the second must still be series 2.
    const html = render(
      [{ label: "Oct 24", parts: [{ series: "Answered", value: 0 }, { series: "Failed", value: 4 }] }],
      ["Answered", "Failed"],
    );

    expect(legend(html)).toEqual(["Answered", "Failed"]);
    expect(rects(bars(html)[0]).map((rect) => rect.series)).toEqual([2]);
  });

  it("no bars renders no data", () => {
    // Catches an empty chart drawn as a bare svg and an all-zero chart drawn as flat bars: both say "No data." and draw nothing.
    for (const days of [
      [],
      [{ label: "Oct 24", parts: [{ series: "Answered", value: 0 }] }],
      [{ label: "Oct 24", parts: [] }],
    ] as Day[][]) {
      const html = render(days, ["Answered"]);
      expect(html).toContain("No data.");
      expect(html).not.toContain("<svg");
    }
    expect(render([{ label: "Oct 24", parts: [{ series: "Answered", value: 3 }] }], ["Answered"])).not.toContain("No data.");
  });

  it("draws no inline style", () => {
    // Catches a bar coloured or sized through a style attribute, which the page's no-inline-style rule forbids.
    const html = render(
      [{ label: "Oct 24", parts: [{ series: "Answered", value: 3 }, { series: "Failed", value: 1 }] }],
      ["Answered", "Failed"],
    );
    expect(html).not.toMatch(/\sstyle=/);
  });

  it("the four series colours are defined for light and dark", () => {
    // Catches a series token missing from the dark block (unreadable bars at night) or a series class with no rule behind it.
    const css = readFileSync(GLOBALS, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    const dark = css.slice(css.search(/@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)/));
    const light = css.slice(0, css.search(/@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)/));
    for (const n of [1, 2, 3, 4]) {
      expect(light, `--series-${n} light`).toMatch(new RegExp(`--series-${n}\\s*:`));
      expect(dark, `--series-${n} dark`).toMatch(new RegExp(`--series-${n}\\s*:`));
      expect(css, `.series-${n}`).toMatch(new RegExp(`\\.series-${n}\\b[^{]*\\{[^}]*var\\(--series-${n}\\)`));
    }
  });
});
