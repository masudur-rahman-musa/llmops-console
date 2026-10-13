import type { ReactNode } from "react";

import { NavLinks } from "@/components/nav-links";

import styles from "./layout.module.css";

export default function ConsoleLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <header className={styles.topbar}>
        <div className={styles.inner}>
          <nav className={styles.nav}>
            <NavLinks className={styles.link} />
          </nav>
          <form method="post" action="/api/logout" className={styles.signout}>
            <button type="submit">Sign out</button>
          </form>
        </div>
      </header>
      <main>{children}</main>
    </>
  );
}
