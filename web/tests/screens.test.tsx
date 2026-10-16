import { readFileSync } from "node:fs";

import { isValidElement, type ReactElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiIsConfigured } from "@/lib/console-api";
import LoadingScreen from "@/app/(console)/loading";
import NotFoundScreen from "@/app/(console)/not-found";
import ErrorScreen from "@/app/(console)/error";
import { GET as readyzRoute } from "@/app/readyz/route";

function source(relative: string): string {
  return readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8");
}

type Anchor = { href: string | null };

function anchors(html: string): Anchor[] {
  return Array.from(html.matchAll(/<a\b([^>]*)>/g), (match) => ({
    href: match[1].match(/\bhref="([^"]*)"/)?.[1]?.replace(/&amp;/g, "&") ?? null,
  }));
}

// Walk a React element tree and return every node's onClick handler.
function clickHandlers(node: ReactNode): unknown[] {
  if (!isValidElement(node)) {
    return [];
  }
  const props = node.props as { onClick?: unknown; children?: ReactNode };
  const here = props.onClick === undefined ? [] : [props.onClick];
  const children = props.children;
  const kids = (Array.isArray(children) ? children : [children]).flatMap(clickHandlers);
  return [...here, ...kids];
}

const HASH = "a".repeat(64);
const SECRET = "0123456789abcdef0123456789abcdef0123456789abcdef"; // 48 hex chars
const SHORT_SECRET = "0123456789abcdef0123456789abcde"; // 31 chars
const API_URL = "http://gateway.apps.svc.cluster.local:8080";
const KEY = "admin-key";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("console screens", () => {
  it("loading screen renders a status message", () => {
    // Catches a loading screen with no status role for a screen reader, or no visible "loading" text at all.
    const html = renderToStaticMarkup(<LoadingScreen />);
    expect(html).toMatch(/role="status"/);
    expect(html.toLowerCase()).toContain("loading");
  });

  it("not-found screen links back to the home page", () => {
    // Catches a not-found screen that drops the home link, or points it somewhere other than the console home "/".
    const html = renderToStaticMarkup(<NotFoundScreen />);
    expect(html.replace(/<[^>]*>/g, " ").trim().length).toBeGreaterThan(0);
    expect(anchors(html).map((anchor) => anchor.href)).toContain("/");
  });

  it("error screen is a client part with a retry", () => {
    // Catches an error.tsx shipped as a server part (no "use client"), or a Try again button wired to anything but the reset prop.
    const first = source("app/(console)/error.tsx")
      .split("\n")
      .map((line) => line.trim())
      .find((line) => line.length > 0);
    expect(first).toMatch(/^(["'])use client\1/);

    const reset = vi.fn();
    const element = ErrorScreen({ error: new Error("boom"), reset }) as ReactElement;
    const html = renderToStaticMarkup(element);
    expect(html).toMatch(/role="alert"/);
    expect(html).toContain("<button");

    const handlers = clickHandlers(element);
    expect(handlers).toContain(reset);
    expect(reset).not.toHaveBeenCalled();
    for (const handler of handlers) {
      if (handler === reset) (handler as () => void)();
    }
    expect(reset).toHaveBeenCalledTimes(1);
  });

  it("readyz is still 200 ready and 503 unconfigured", async () => {
    // Catches a reworked readyz that stops answering 200 ready when all four are set, or 503 unconfigured for a 31-character secret.
    vi.stubEnv("CONSOLE_PASSWORD_SHA256", HASH);
    vi.stubEnv("CONSOLE_SESSION_SECRET", SECRET);
    vi.stubEnv("CONSOLE_API_URL", API_URL);
    vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
    const ready = readyzRoute();
    expect(ready.status).toBe(200);
    expect(await ready.json()).toEqual({ status: "ready" });

    vi.stubEnv("CONSOLE_SESSION_SECRET", SHORT_SECRET);
    expect(SHORT_SECRET).toHaveLength(31);
    const unconfigured = readyzRoute();
    expect(unconfigured.status).toBe(503);
    expect(await unconfigured.json()).toEqual({ status: "unconfigured" });
  });

  it("readyz keeps no length number of its own", () => {
    // Catches a readyz still holding its own 32 or MIN_SESSION_SECRET_LENGTH instead of reusing the session and console-api checks.
    const text = readFileSync(new URL("../src/app/readyz/route.ts", import.meta.url), "utf8");
    expect(text).toContain("secretIsUsable");
    expect(text).toContain("apiIsConfigured");
    expect(text).not.toContain("MIN_SESSION_SECRET_LENGTH");
    expect(text).not.toMatch(/\b32\b/);
  });

  it("apiIsConfigured is true only for an http url and a set key", () => {
    // Catches an apiIsConfigured that accepts a non-http scheme, an unparseable url, or a missing key.
    expect(apiIsConfigured({ CONSOLE_API_URL: "http://x", CONSOLE_ADMIN_KEY: KEY })).toBe(true);
    expect(apiIsConfigured({ CONSOLE_API_URL: "https://x", CONSOLE_ADMIN_KEY: KEY })).toBe(true);
    expect(apiIsConfigured({ CONSOLE_API_URL: "ftp://x", CONSOLE_ADMIN_KEY: KEY })).toBe(false);
    expect(apiIsConfigured({ CONSOLE_API_URL: "not a url", CONSOLE_ADMIN_KEY: KEY })).toBe(false);
    expect(apiIsConfigured({ CONSOLE_API_URL: "http://x", CONSOLE_ADMIN_KEY: undefined })).toBe(false);
    expect(apiIsConfigured({ CONSOLE_ADMIN_KEY: KEY })).toBe(false);
  });
});
