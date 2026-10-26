import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import ConsoleLayout from "@/app/(console)/layout";
import RootLayout from "@/app/layout";
import LoginPage from "@/app/login/page";

const BANNER = "Demo data, invented companies";

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

// An element whose whole text is the banner, so a variant that merely contains it does not count.
function bannerIndex(html: string): number {
  return html.indexOf(`>${BANNER}<`);
}

describe("console shell", () => {
  it("the root layout renders the banner text exactly", () => {
    // Catches a banner with different wording, casing or punctuation, a missing banner, or one rendered after the page content.
    const html = renderToStaticMarkup(
      <RootLayout>
        <p id="child">page content</p>
      </RootLayout>,
    );
    expect(html).toMatch(/<html\b[^>]*\blang="en"/);
    expect(html).toMatch(/<body\b/);
    expect(bannerIndex(html)).toBeGreaterThan(-1);
    expect(occurrences(html, BANNER)).toBe(1);
    expect(html.indexOf('id="child"')).toBeGreaterThan(bannerIndex(html));
  });

  it("the login page renders the banner", async () => {
    // Catches a login page outside the banner, a failure message that is missing, repeated or shown without error=1, or a form not posting to /api/login.
    const failed = renderToStaticMarkup(
      <RootLayout>{await LoginPage({ searchParams: Promise.resolve({ error: "1", next: "/cost" }) })}</RootLayout>,
    );
    expect(bannerIndex(failed)).toBeGreaterThan(-1);
    expect(occurrences(failed, "Sign-in failed.")).toBe(1);

    const form = failed.match(/<form\b[^>]*>/)?.[0] ?? "";
    expect(form).toMatch(/\baction="\/api\/login"/);
    expect(form).toMatch(/\bmethod="post"/i);
    expect(failed).toMatch(/<input\b(?=[^>]*\bname="password")(?=[^>]*\btype="password")[^>]*>/);
    expect(failed).toMatch(/<input\b(?=[^>]*\bname="next")(?=[^>]*\btype="hidden")[^>]*>/);

    const fresh = renderToStaticMarkup(<RootLayout>{await LoginPage({ searchParams: Promise.resolve({}) })}</RootLayout>);
    expect(bannerIndex(fresh)).toBeGreaterThan(-1);
    expect(occurrences(fresh, "Sign-in failed.")).toBe(0);
  });

  it("the nav carries four links in order with their labels", () => {
    // Catches a nav with a missing, extra, reordered, mislabelled or wrongly addressed link, or nav rendered after the page content.
    const html = renderToStaticMarkup(
      <ConsoleLayout>
        <p id="child">page content</p>
      </ConsoleLayout>,
    );
    const navs = html.match(/<nav\b[^>]*>[\s\S]*?<\/nav>/g) ?? [];
    expect(navs).toHaveLength(1);
    const links = Array.from((navs[0] ?? "").matchAll(/<a\b[^>]*\bhref="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g), (match) => ({
      href: match[1],
      label: match[2].replace(/<[^>]*>/g, "").trim(),
    }));
    expect(links).toEqual([
      { href: "/", label: "Overview" },
      { href: "/requests", label: "Requests" },
      { href: "/cost", label: "Cost" },
      { href: "/quality", label: "Quality" },
    ]);
    expect(html.indexOf('id="child"')).toBeGreaterThan(html.indexOf("</nav>"));
  });
});
