/** Pure formatters every page shares. Pinned by tests/format.test.ts. */

export const NA = "n/a";

const SHORT_SHA_LENGTH = 12;

function missing(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || !Number.isFinite(value);
}

export function usd(value: number | null | undefined, decimals = 2): string {
  if (missing(value)) {
    return NA;
  }
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

const SMALL_USD_DECIMALS = 6;

/**
 * Money that can be a fraction of a cent: an amount above zero and under one cent shows
 * six decimals, so one cached request's saving does not read $0.00.
 */
export function smallUsd(value: number | null | undefined): string {
  if (missing(value)) {
    return NA;
  }
  return value !== 0 && Math.abs(value) < 0.01 ? usd(value, SMALL_USD_DECIMALS) : usd(value);
}

/** A rate from 0 to 1, shown as a percentage with two decimals. */
export function percent(value: number | null | undefined): string {
  return missing(value) ? NA : `${(value * 100).toFixed(2)}%`;
}

/** Percent points as stored (57.14 is 57.14%), never scaled again. */
export function points(value: number | null | undefined): string {
  return missing(value) ? NA : `${value.toFixed(2)}%`;
}

/** A score to at most 2 decimals, with no trailing zeros: 0.9 stays `0.9`, 48.484848 is `48.48`. */
export function rounded(value: number): string {
  return String(Number(value.toFixed(2)));
}

/** Seconds, shown as whole milliseconds. */
export function ms(seconds: number | null | undefined): string {
  return missing(seconds) ? NA : `${Math.round(seconds * 1000)} ms`;
}

/** `YYYY-MM-DD HH:MM UTC`, converted from whatever offset the value carries. */
export function utcTime(iso: string | null | undefined): string {
  if (!iso) {
    return NA;
  }
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) {
    return NA;
  }
  return `${at.toISOString().slice(0, 10)} ${at.toISOString().slice(11, 16)} UTC`;
}

/**
 * The value when it is an absolute http or https URL, otherwise null. Surrounding
 * whitespace is refused because `new URL` would silently strip it. Pinned by the
 * rejection table in tests/format.test.ts.
 */
export function safeHref(value: string | null | undefined): string | null {
  if (!value || value.trim() !== value) {
    return null;
  }
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? value : null;
  } catch {
    return null;
  }
}

export function shortSha(sha: string): string {
  return sha.slice(0, SHORT_SHA_LENGTH);
}

/** A `YYYY-MM-DD` day as `Oct 24`, read as a UTC date so no zone moves it. */
export function shortDay(day: string): string {
  const at = new Date(`${day}T00:00:00Z`);
  if (Number.isNaN(at.getTime())) {
    return day;
  }
  return at.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}
