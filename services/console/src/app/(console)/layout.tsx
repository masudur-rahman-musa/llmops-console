import type { ReactNode } from "react";

import { NAV } from "@/lib/nav";

export default function ConsoleLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <nav>
        {NAV.map((item) => (
          <a key={item.href} href={item.href}>
            {item.label}
          </a>
        ))}
      </nav>
      <form method="post" action="/api/logout">
        <button type="submit">Sign out</button>
      </form>
      <main>{children}</main>
    </>
  );
}
