import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = { title: "AI Operations Console" };

/**
 * The only layout rendering `<html>`, so every page loads `globals.css`
 * (tests/styles.test.tsx).
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
