import type { ReactNode } from "react";

/**
 * One section of a console tab: a `section.panel` box labelled by its title, with a plain `<h2>` first.
 * Every page heading goes through here, so sections read apart by eye. Pinned by tests/panels.test.tsx.
 */
export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="panel" aria-label={title}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}
