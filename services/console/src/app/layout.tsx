import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = { title: "AI Operations Console" };

/**
 * The banner is on every page, `/login` included (tests/shell.test.tsx). The
 * only layout rendering `<html>`, so every page loads `globals.css`
 * (tests/styles.test.tsx).
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="banner">
          <div role="note" className="badge">
            Demo data, invented companies
          </div>
        </div>
        {children}
      </body>
    </html>
  );
}
