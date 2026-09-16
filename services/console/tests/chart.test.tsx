import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LineChart } from "@/components/line-chart";

function count(html: string, pattern: RegExp): number {
  return (html.match(pattern) ?? []).length;
}

const SVG = /<svg\b/g;
const CIRCLE = /<circle\b/g;
// `<line` followed by whitespace, `/` or `>`, so `<linearGradient` does not count.
const LINE = /<line[\s/>]/g;
const POLYLINE = /<polyline\b/g;

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
      <LineChart title="Decision accuracy" points={points} markers={[1789550000, 1789565000, 1789570000]} />,
    );
    expect(count(three, LINE)).toBe(3);

    const none = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} />);
    expect(count(none, LINE)).toBe(0);
    const empty = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} markers={[]} />);
    expect(count(empty, LINE)).toBe(0);
  });
});
