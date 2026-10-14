import { safeHref } from "@/lib/format";

/**
 * A link only when `safeHref` accepts the URL, the URL as plain text otherwise,
 * nothing when it is null. Every link carries `rel="noopener noreferrer"`.
 * Pinned by tests/pages.test.tsx ("external links carry rel noopener noreferrer").
 */
export function ExternalLink({ url, label }: { url: string | null; label: string }) {
  if (url === null) {
    return null;
  }
  const href = safeHref(url);
  if (href === null) {
    return <span>{url}</span>;
  }
  return (
    <a href={href} rel="noopener noreferrer" target="_blank">
      {label}
    </a>
  );
}
