"use client";

import { usePathname } from "next/navigation";

import { NAV } from "@/lib/nav";

/**
 * The six nav links, the current page marked with `aria-current="page"` by an
 * exact match, so `/` never marks on `/cost` and an unknown or null path marks
 * none. Pinned by tests/styles.test.tsx.
 */
export function NavLinks({ className }: { className?: string }) {
  const pathname = usePathname();
  return (
    <>
      {NAV.map((item) => (
        <a key={item.href} href={item.href} className={className} aria-current={item.href === pathname ? "page" : undefined}>
          {item.label}
        </a>
      ))}
    </>
  );
}
