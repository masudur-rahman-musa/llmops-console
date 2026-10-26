import type { ReactNode } from "react";

export type StatCard = {
  label: string;
  value: ReactNode;
  /** A smaller line under the value, such as the baseline it is checked against. */
  sub?: string;
  /** One line saying what the number counts, when the label alone is not enough. */
  hint?: string;
  /** An in-console page the whole card links to. */
  href?: string;
};

/**
 * The one row of number cards every tab shows under its window line: label,
 * value, then the optional sub line and hint. Pinned by tests/tab-shape.test.tsx.
 */
export function StatCards({ cards }: { cards: ReadonlyArray<StatCard> }) {
  return (
    <ul className="cards">
      {cards.map((card) => {
        const body = (
          <>
            <span className="card-label">{card.label}</span>
            <strong className="card-value">{card.value}</strong>
            {card.sub !== undefined && <span className="card-sub">{card.sub}</span>}
            {card.hint !== undefined && <span className="card-hint">{card.hint}</span>}
          </>
        );
        return (
          <li key={card.label} className="card">
            {card.href === undefined ? (
              body
            ) : (
              <a href={card.href} className="card-link">
                {body}
              </a>
            )}
          </li>
        );
      })}
    </ul>
  );
}
