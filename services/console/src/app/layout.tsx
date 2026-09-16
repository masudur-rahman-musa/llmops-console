import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = { title: "AI Operations Console" };

/** The banner is on every page, `/login` included (tests/shell.test.tsx). */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div role="note">Demo data, invented companies</div>
        {children}
      </body>
    </html>
  );
}
