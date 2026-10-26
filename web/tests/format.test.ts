import { describe, expect, it } from "vitest";

import { NA, ms, percent, points, safeHref, shortSha, smallUsd, usd, utcTime } from "@/lib/format";

describe("format", () => {
  it("usd formats with two decimals and grouping", () => {
    // Catches a formatter without en-US grouping, with other than two default decimals, ignoring the decimals argument, or treating zero as missing.
    expect(usd(1234.5)).toBe("$1,234.50");
    expect(usd(1234.567)).toBe("$1,234.57");
    expect(usd(1234567.8)).toBe("$1,234,567.80");
    expect(usd(0)).toBe("$0.00");
    expect(usd(0.0042, 4)).toBe("$0.0042");
  });

  it("percent multiplies by 100 with two decimals", () => {
    // Catches a rate printed unscaled, rounded to other than two decimals, or zero treated as missing.
    expect(percent(0.005)).toBe("0.50%");
    expect(percent(0.9312)).toBe("93.12%");
    expect(percent(1)).toBe("100.00%");
    expect(percent(0)).toBe("0.00%");
  });

  it("points shows percent points as they are", () => {
    // Catches a points value multiplied by 100 a second time (57.14 shown as 5714.00%), rounded to other than two decimals, zero treated as missing, or null shown as a number.
    expect(points(57.14)).toBe("57.14%");
    expect(points(100)).toBe("100.00%");
    expect(points(0)).toBe("0.00%");
    expect(points(null)).toBe("n/a");
    expect(points(undefined)).toBe("n/a");
  });

  it("small money shows six decimals under a cent", () => {
    // Catches an amount between zero and one cent rounded to $0.00, six decimals used for amounts a cent or more, zero shown with six decimals, or null shown as a number.
    expect(smallUsd(0.0000257)).toBe("$0.000026");
    expect(smallUsd(0.01)).toBe("$0.01");
    expect(smallUsd(0)).toBe("$0.00");
    expect(smallUsd(4.55)).toBe("$4.55");
    expect(smallUsd(1234.5)).toBe("$1,234.50");
    expect(smallUsd(null)).toBe("n/a");
  });

  it("ms rounds seconds to whole milliseconds", () => {
    // Catches seconds printed unscaled, truncated instead of rounded, or zero treated as missing.
    expect(ms(0.4124)).toBe("412 ms");
    expect(ms(0.4126)).toBe("413 ms");
    expect(ms(2)).toBe("2000 ms");
    expect(ms(0)).toBe("0 ms");
  });

  it("utc time formats as yyyy-mm-dd hh:mm utc", () => {
    // Catches a time rendered in the server's local zone, with seconds, or in a locale format.
    expect(utcTime("2026-09-16T07:05:00Z")).toBe("2026-09-16 07:05 UTC");
    expect(utcTime("2026-09-16T23:59:59Z")).toBe("2026-09-16 23:59 UTC");
    expect(utcTime("2026-09-16T01:30:00+02:00")).toBe("2026-09-15 23:30 UTC");
  });

  it("null renders n/a", () => {
    // Catches a formatter that prints null as $0.00, NaN%, 0 ms, a 1970 date, or throws.
    expect(NA).toBe("n/a");
    expect(usd(null)).toBe("n/a");
    expect(usd(undefined)).toBe("n/a");
    expect(percent(null)).toBe("n/a");
    expect(percent(undefined)).toBe("n/a");
    expect(ms(null)).toBe("n/a");
    expect(ms(undefined)).toBe("n/a");
    expect(utcTime(null)).toBe("n/a");
    expect(utcTime(undefined)).toBe("n/a");
    expect(utcTime("not a time")).toBe("n/a");
  });

  it("safeHref accepts http and https", () => {
    // Catches a check so strict it refuses plain http or https links, or one that rewrites the value it accepts.
    for (const value of [
      "https://github.com/example/helpdesk/pull/41",
      "http://docs.example.com/postmortems/31",
      "https://docs.example.com/a?b=1#c",
    ]) {
      expect(safeHref(value)).toBe(value);
    }
  });

  it("safeHref rejection table", () => {
    // Catches a prefix or case-sensitive check, or a relative or protocol-relative URL let through as a link.
    const rows = ["javascript:alert(1)", "JAVASCRIPT:alert(1)", "data:text/html,x", "//evil.example", "", " https://example.com", "/relative"];
    for (const value of rows) {
      expect(safeHref(value), JSON.stringify(value)).toBeNull();
    }
    expect(safeHref(null)).toBeNull();
    expect(safeHref(undefined)).toBeNull();
  });

  it("shortSha keeps 12 and cuts 13 to 12", () => {
    // Catches a short SHA of other than 12 characters, or an off-by-one at the boundary.
    expect(shortSha("0123456789ab")).toBe("0123456789ab");
    expect(shortSha("0123456789abc")).toBe("0123456789ab");
    expect(shortSha("3f9c2a7b1e4d5c6f7a8b9c0d1e2f3a4b5c6d7e8f")).toBe("3f9c2a7b1e4d");
  });
});
