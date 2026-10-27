import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import QualityPage from "@/app/(console)/quality/page";
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

const GLOBALS = fileURLToPath(new URL("../src/app/globals.css", import.meta.url));
const DAY = 86400;
const DATE = /^[A-Z][a-z]{2} \d{1,2}$/;
const TIME = /^\d{2}:\d{2}$/;

/** Epoch seconds of 2026-09-<day> at <hour>:00 UTC. */
function utc(day: number, hour: number): number {
  return Date.UTC(2026, 8, day, hour) / 1000;
}

/** Every `<text>` element: its x and y attributes, its anchor and its visible content. */
function textElements(html: string): Array<{ x: string | null; y: string | null; anchor: string | null; body: string }> {
  return Array.from(html.matchAll(/<text\b([^>]*)>([\s\S]*?)<\/text>/g), (match) => ({
    x: match[1].match(/\bx="([^"]*)"/)?.[1] ?? null,
    y: match[1].match(/\by="([^"]*)"/)?.[1] ?? null,
    anchor: match[1].match(/\btext-anchor="([^"]*)"/)?.[1] ?? null,
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
      <LineChart title="Decision accuracy" points={points} markers={[{ at: 1789550000 }, { at: 1789565000 }, { at: 1789570000 }]} />,
    );
    expect(count(three, LINE)).toBe(3);

    const none = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} />);
    expect(count(none, LINE)).toBe(0);
    const empty = renderToStaticMarkup(<LineChart title="Decision accuracy" points={points} markers={[]} />);
    expect(count(empty, LINE)).toBe(0);
  });

  it("markers draw lines and no text", () => {
    // Catches a marker that still prints something beside its dashed line (a sha, a time, an index), which a reader cannot use.
    const points: Array<[number, number]> = [[utc(12, 6), 91], [utc(14, 6), 93], [utc(15, 6), 88]];
    const bare = renderToStaticMarkup(<LineChart title="Check score" points={points} yFormat={formatPoints} />);
    const marked = renderToStaticMarkup(
      <LineChart title="Check score" points={points} yFormat={formatPoints} markers={[{ at: utc(14, 6) }, { at: utc(15, 6) }]} />,
    );
    const markerLines = (marked.match(/<line[\s/>][^>]*>/g) ?? []).filter((tag) => /\bclass="[^"]*\bchart-marker\b/.test(tag));
    expect(markerLines).toHaveLength(2);
    expect(textElements(marked)).toEqual(textElements(bare));
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

  it("a marker with a non finite position draws no line", () => {
    // Catches a marker at NaN still drawn as a line at NaN instead of being dropped.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[[utc(12, 6), 0.91], [utc(14, 6), 0.93]]}
        yFormat={formatPoints}
        markers={[{ at: Number.NaN }]}
      />,
    );
    expect(count(html, LINE)).toBe(0);
    expect(html).not.toMatch(/NaN/);
  });

  it("each shown date carries its UTC time", () => {
    // Catches a chart with no run times, times in the server's zone (17:00 must stay 17:00), a time under no date or beside the wrong one, and a thinned-out date whose time is still drawn.
    const runs: Array<[number, number]> = [
      [utc(12, 6), 0.91],
      [utc(14, 17), 0.93],
      [utc(15, 6), 0.88],
    ];
    const html = renderToStaticMarkup(<LineChart title="Check score" points={runs} yFormat={formatPoints} />);
    const items = textElements(html);
    const dates = items.filter((item) => DATE.test(item.body));
    const times = items.filter((item) => TIME.test(item.body));
    expect(dates.map((item) => item.body)).toEqual(["Sep 12", "Sep 14", "Sep 15"]);
    expect(times.map((item) => item.body)).toEqual(["06:00", "17:00", "06:00"]);
    for (const [index, date] of dates.entries()) {
      expect(times[index].x, date.body).toBe(date.x);
      expect(Number(times[index].y), date.body).toBeGreaterThan(Number(date.y));
    }

    const crowded = Array.from({ length: 30 }, (_, i): [number, number] => [utc(1, 6) + i * DAY, 0.9 + (i % 3) / 100]);
    const thinned = textElements(renderToStaticMarkup(<LineChart title="Check score" points={crowded} yFormat={formatPoints} />));
    const shownDates = thinned.filter((item) => DATE.test(item.body));
    expect(shownDates.length).toBeLessThan(30);
    expect(thinned.filter((item) => TIME.test(item.body))).toHaveLength(shownDates.length);
  });

  it("the chart fills its width", () => {
    // Catches a fixed width or height on the svg (it would stop at 600 px instead of filling the page), a lost viewBox (the drawing would not scale), and a chart class that never sets width 100%.
    const html = renderToStaticMarkup(
      <LineChart title="Check score" points={[[utc(12, 6), 0.91], [utc(14, 6), 0.93]]} yFormat={formatPoints} />,
    );
    const tag = html.match(/<svg\b[^>]*>/)?.[0] ?? "";
    expect(tag).toMatch(/\bviewBox="0 0 \d+ \d+"/);
    expect(tag).toMatch(/\bclass="chart"/);
    expect(tag).not.toMatch(/\swidth=/);
    expect(tag).not.toMatch(/\sheight=/);

    const css = readFileSync(GLOBALS, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
    const rule = css.match(/(?:^|\})\s*\.chart\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(rule).toMatch(/width:\s*100%/);
  });

  describe("fixed score range", () => {
    const STEPS = [0, 25, 50, 75, 100];
    const RANGE: [number, number] = [0, 100];

    /** The y of a step label's mark: the label sits 4 px below the height it names. */
    function stepHeight(html: string, label: string): number {
      const found = textElements(html).filter((item) => item.body === label);
      expect(found, label).toHaveLength(1);
      return Number(found[0].y) - 4;
    }

    function circleHeights(html: string): number[] {
      return Array.from(html.matchAll(/<circle\b[^>]*\bcy="([^"]*)"/g), (match) => Number(match[1]));
    }

    it("a fixed range keeps its steps on a flat line", () => {
      // Catches a chart that ignores range and steps and so shows only the one score the runs share, repeats a step, writes the steps in yFormat (0.00%) instead of whole percents, or draws a flat line anywhere but at its place on the 0 to 100 scale.
      const html = renderToStaticMarkup(
        <LineChart
          title="Check score"
          points={[[utc(12, 6), 57.14], [utc(13, 6), 57.14], [utc(14, 6), 57.14]]}
          yFormat={formatPoints}
          range={RANGE}
          steps={STEPS}
        />,
      );
      const heights = ["0%", "25%", "50%", "75%", "100%"].map((label) => stepHeight(html, label));
      for (let i = 1; i < heights.length; i += 1) {
        expect(heights[i], "bottom to top").toBeLessThan(heights[i - 1]);
      }
      const flat = circleHeights(html);
      expect(flat).toHaveLength(3);
      for (const cy of flat) {
        expect(cy).toBeLessThan(heights[2]);
        expect(cy).toBeGreaterThan(heights[3]);
      }
      expect(textElements(html).map((item) => item.body)).not.toContain("57.14%");
    });

    it("a score outside the fixed range stays on the plot", () => {
      // Catches a score of 120 or -5 drawn past the top or bottom of the plot (off the axis) instead of at the nearest edge.
      const html = renderToStaticMarkup(
        <LineChart
          title="Check score"
          points={[[utc(12, 6), 120], [utc(14, 6), -5]]}
          yFormat={formatPoints}
          range={RANGE}
          steps={STEPS}
        />,
      );
      const top = stepHeight(html, "100%");
      const bottom = stepHeight(html, "0%");
      expect(circleHeights(html)).toEqual([top, bottom]);
      expect(html).not.toMatch(/NaN|Infinity/);
    });

    it("the baseline draws one line with its label", () => {
      // Catches a baseline that is not drawn, drawn as a plain marker (no chart-baseline class), at the wrong height, drawn once per marker, or labelled without its value or anchored middle.
      const html = renderToStaticMarkup(
        <LineChart
          title="Check score"
          points={[[utc(12, 6), 57.14], [utc(14, 6), 57.14], [utc(15, 6), 57.14]]}
          yFormat={formatPoints}
          range={RANGE}
          steps={STEPS}
          baseline={{ value: 48.48, label: "Baseline 48.48%" }}
          markers={[{ at: utc(14, 6) }, { at: utc(15, 6) }]}
        />,
      );
      const lines = html.match(/<line[\s/>][^>]*>/g) ?? [];
      expect(lines).toHaveLength(3);
      const baseline = lines.filter((tag) => /\bclass="[^"]*\bchart-baseline\b/.test(tag));
      expect(baseline).toHaveLength(1);
      const bottom = stepHeight(html, "0%");
      const top = stepHeight(html, "100%");
      const expected = bottom + (top - bottom) * 0.4848;
      const y1 = Number(baseline[0].match(/\by1="([^"]*)"/)?.[1]);
      const y2 = Number(baseline[0].match(/\by2="([^"]*)"/)?.[1]);
      expect(y1).toBeCloseTo(expected, 1);
      expect(y2).toBeCloseTo(expected, 1);
      expect(html).toMatch(/<p class="chart-key">.*Check score 57\.14% \(newest run\).*Baseline 48\.48%.*Release<\/p>/);
    });

    it("the baseline label never covers a point", () => {
      // Catches a baseline label still drawn just above its line, where it sits on a point at the baseline score, or one moved into the plot where a point at 100% lies under it.
      const html = renderToStaticMarkup(
        <LineChart
          title="Check score"
          points={[[utc(12, 6), 100], [utc(15, 6), 48.48]]}
          yFormat={formatPoints}
          range={RANGE}
          steps={STEPS}
          baseline={{ value: 48.48, label: "Baseline 48.48%" }}
        />,
      );
      const top = stepHeight(html, "100%");
      const radius = Number(html.match(/<circle\b[^>]*\br="([^"]*)"/)?.[1]);
      expect(radius).toBe(3);
      expect(Math.min(...circleHeights(html))).toBe(top);
      expect(textElements(html).filter((item) => item.body === "Baseline 48.48%")).toEqual([]);
      expect(html.split("</svg>")[1]).toContain("Baseline 48.48%");
    });

    it("a baseline that is not a number draws nothing", () => {
      // Catches a NaN baseline drawn as a line at NaN (or its label printed anyway) instead of being dropped like a marker at NaN.
      const html = renderToStaticMarkup(
        <LineChart
          title="Check score"
          points={[[utc(12, 6), 57.14], [utc(14, 6), 57.14]]}
          yFormat={formatPoints}
          range={RANGE}
          steps={STEPS}
          baseline={{ value: Number.NaN, label: "Baseline NaN%" }}
        />,
      );
      expect(count(html, LINE)).toBe(0);
      expect(html).not.toContain("Baseline");
      expect(html).not.toMatch(/NaN/);
    });
  });

  it("dates and times at the ends stay inside the chart", () => {
    // Catches a first or last run date and time centred on its point, which the svg edge cuts in half ("Oct 2…"), or a middle one pushed to an edge anchor.
    const html = renderToStaticMarkup(
      <LineChart
        title="Check score"
        points={[[utc(12, 6), 0.91], [utc(14, 7), 0.93], [utc(15, 8), 0.88]]}
        yFormat={formatPoints}
      />,
    );
    const items = textElements(html);
    const anchorOf = (body: string): string | null => items.find((item) => item.body === body)?.anchor ?? null;
    expect(anchorOf("Sep 12")).toBe("start");
    expect(anchorOf("06:00")).toBe("start");
    expect(anchorOf("Sep 14")).toBe("middle");
    expect(anchorOf("07:00")).toBe("middle");
    expect(anchorOf("Sep 15")).toBe("end");
    expect(anchorOf("08:00")).toBe("end");
  });
});

describe("Quality page chart", () => {
  beforeEach(() => {
    vi.stubEnv("CONSOLE_API_URL", "http://console-api.test");
    vi.stubEnv("CONSOLE_ADMIN_KEY", "adminkey-sentinel-7c41e2d9f0");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("the quality chart prints no commit hash", async () => {
    // Catches a Quality page that still hands each release's short sha to the chart as a label, or one that drops the release lines along with the labels.
    const body: unknown = JSON.parse(readFileSync(new URL("./fixtures/quality.json", import.meta.url), "utf8"));
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
    const html = renderToStaticMarkup(await QualityPage({ searchParams: Promise.resolve({}) }));
    const chart = (html.match(/<svg\b[\s\S]*?<\/svg>/g) ?? []).filter((svg) => count(svg, CIRCLE) > 0);
    expect(chart).toHaveLength(1);
    expect(chart[0]).not.toContain("5e2d8c1");
    expect(chart[0]).not.toContain("3f9c2a7");
    const markerLines = (chart[0].match(/<line[\s/>][^>]*>/g) ?? []).filter((tag) => /\bclass="[^"]*\bchart-marker\b/.test(tag));
    expect(markerLines).toHaveLength(2);
  });
});
