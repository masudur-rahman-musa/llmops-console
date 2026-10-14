/** The seven console pages, in nav order. Also the whole `next` allowlist. */
export const NAV: ReadonlyArray<{ href: string; label: string }> = [
  { href: "/", label: "Home" },
  { href: "/cost", label: "Cost" },
  { href: "/quality", label: "Quality" },
  { href: "/residency", label: "Residency" },
  { href: "/reliability", label: "Reliability" },
  { href: "/incidents", label: "Incidents" },
  { href: "/releases", label: "Releases" },
];

/**
 * An exact match against the nav paths, never a prefix or a URL parse, so no
 * `next` value can redirect off-site. Pinned by tests/login-route.test.ts.
 */
export function safeNext(next: string | null | undefined): string {
  return NAV.some((item) => item.href === next) ? (next as string) : "/";
}
