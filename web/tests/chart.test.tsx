import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LineChart } from "@/components/ui/line-chart";
import { points as formatPoints } from "@/lib/format";

function count(html: string, pattern: RegExp): number {
  return (html.match(pattern) ?? []).length;
}

const SVG = /<svg\b/g;
const CIRCLE = /<circle\b/g;
// `<line` followed by whitespace, `/` or `>`, so `<linearGradient` does not count.
const LINE = /<line[\s/>]/g;
const POLYLINE = /<polyline\b/g;

const DAY = 86400;
const DATE = /^[A-Z][a-z]{2} \d{1,2}$/;

/** Epoch seconds of 2026-09-<day> at <hour>:00 UTC. */
function utc(day: number, hour: number): number {
  return Date.UTC(2026, 8, day, hour) / 1000;
}

/** Every `<text>` element: its x attribute and its visible content. */
function textElements(html: string): Array<{ x: string | null; body: string }> {
  return Array.from(html.matchAll(/<text\b([^>]*)>([\s\S]*?)<\/text>/g), (match) => ({
    x: match[1].match(/\bx="([^"]*)"/)?.[1] ?? null,
    body: match[2].replace(/<!--[\s\S]*?-->/g, "").replace(/<[^>]*>/g, "").trim(),
  }));
}

describe("LineChart", () => {
  it("zero points renders no data", () => {
    // Catches an empty chart drawn as an svg with no series instead of the No data. notice.
    const html = renderToStaticMarkup(<LineChart title="Decision accuracy" points={[]} />);
    expect(html).toContain("No data.");
    expect(count(html, SVG)).toBe(0);
  });

  it("one point renders one circle", () => {
    // Catches a single point treated as no data, or plotted with no circle or with more than one.
    const html = renderToStaticMarkup(<LineChart title="Decision accuracy" points={[[1789545600, 0.93]]} />);
    expect(html).not.toContain("No data.");
    expect(count(html, SVG)).toBe(1);
    expect(count(html, CIRCLE)).toBe(1);

    const two = renderToStaticMarkup(
      <LineChart title="Decision accuracy" points={[[1789545600, 0.93], [1789549200, 0.91]]} />,
    );
    expect(count(two, CIRCLE)).toBe(2);
    expect(count(two, POLYLINE)).toBe(1);
  });

  it("non finite values are dropped", () => {
    // Catches NaN or Infinity plotted as coordinates, or a chart of only non-finite points drawn instead of No data.
    const mixed = renderToStaticMarkup(
      <LineChart
        title="p95 latency"
        points={[
          [1789545600, 0.41],
          [Number.NaN, 0.44],
          [1789552800, Number.POSITIVE_INFINITY],
          [Number.NEGATIVE_INFINITY, 0.3],
          [1789556400, Number.NaN],
          [1789560000, 0.39],
        ]}
      />,
    );
    expect(count(mixed, CIRCLE)).toBe(2);
    expect(mixed).not.toMatch(/NaN|Infinity/);

    const none = renderToStaticMarkup(
      <LineChart title="p95 latency" points={[[Number.NaN, 1], [1789545600, Number.POSITIVE_INFINITY]]} />,
    );
    expect(none).toContain("No data.");
    expect(count(none, SVG)).toBe(0);
  });

  it("markers render one line per marker", () => {
    // Catches markers dropped, drawn once for all, or axes and gridlines drawn as extra line elements.
    const points: Array<[number, number]> = [[1789545600, 0.93], [1789560000, 0.91], [1789574400, 0.88]];
    const three = renderToStaticMarkup(
      <LineChart title="Decision accuracy" points={points} markers={[
          { at: 1789550000, label: "aaaaaaa" },
          { at: 1789565000, label: "bbbbbbb" },
          { at: 1789570000, label: "ccccccc" },
        ]} />,
    );
    expect(count(three, LINE)).toBe(3);

    const none = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} />);
    expect(count(none, LINE)).toBe(0);
    const empty = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} markers={[]} />);
    expect(count(empty, LINE)).toBe(0);
  });

  it("the score axis names the lowest middle and highest score", () => {
    // Catches an axis that shows no scores, shows only one end, ignores yFormat, or prints a lone score more than once.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[[DAY * 1, 88], [DAY * 2, 91], [DAY * 3, 93]]}
        yFormat={formatPoints}
      />,
    );
    const labels = textElements(html).map((item) => item.body);
    expect(labels).toContain("88.00%");
    expect(labels).toContain("90.50%");
    expect(labels).toContain("93.00%");

    const single = renderToStaticMarkup(
      <LineChart title="Check score" points={[[DAY * 1, 91]]} yFormat={formatPoints} />,
    );
    expect(textElements(single).filter((item) => item.body === "91.00%")).toHaveLength(1);
  });

  it("each run gets its date", () => {
    // Catches a chart with no run dates, dates in local time or the wrong format, or a date dropped when runs sit close together.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[
          [utc(12, 6), 0.91],
          [utc(14, 6), 0.93],
          [utc(15, 6), 0.88],
        ]}
        yFormat={formatPoints}
      />,
    );
    const labels = textElements(html).map((item) => item.body);
    expect(labels).toContain("Sep 12");
    expect(labels).toContain("Sep 14");
    expect(labels).toContain("Sep 15");
  });

  it("dates closer than 48 px are skipped", () => {
    // Catches date labels drawn for every run so they overlap, or a thinning that drops the first run's date.
    const runs = Array.from({ length: 30 }, (_, i): [number, number] => [utc(1, 6) + i * DAY, 0.9 + (i % 3) / 100]);
    const html = renderToStaticMarkup(<LineChart title="Check score" points={runs} yFormat={formatPoints} />);
    const dates = textElements(html)
      .filter((item) => DATE.test(item.body))
      .map((item) => ({ body: item.body, x: Number(item.x) }))
      .sort((a, b) => a.x - b.x);
    expect(dates.length).toBeGreaterThan(1);
    expect(dates.length).toBeLessThan(30);
    expect(dates[0].body).toBe("Sep 1");
    for (let i = 1; i < dates.length; i += 1) {
      expect(dates[i].x - dates[i - 1].x).toBeGreaterThanOrEqual(48);
    }
  });

  it("each marker carries its label", () => {
    // Catches a marker drawn without its label, a label shared by all markers, or an axis drawn with extra line elements.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[[utc(12, 6), 0.91], [utc(14, 6), 0.93], [utc(15, 6), 0.88]]}
        yFormat={formatPoints}
        markers={[
          { at: utc(14, 6), label: "5e2d8c1" },
          { at: utc(15, 6), label: "3f9c2a7" },
        ]}
      />,
    );
    expect(count(html, LINE)).toBe(2);
    const labels = textElements(html).map((item) => item.body);
    expect(labels).toContain("5e2d8c1");
    expect(labels).toContain("3f9c2a7");
  });

  it("a marker with a non finite position draws neither line nor label", () => {
    // Catches a marker at NaN still drawing its label (or a line at NaN) after the finite check moved to the line only.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[[utc(12, 6), 0.91], [utc(14, 6), 0.93]]}
        yFormat={formatPoints}
        markers={[{ at: Number.NaN, label: "bad" }]}
      />,
    );
    expect(count(html, LINE)).toBe(0);
    expect(html).not.toContain("bad");
    expect(html).not.toMatch(/NaN/);
  });
});
