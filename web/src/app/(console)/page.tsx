import { consolePage } from "@/components/layout/console-page";
import { NA, ms, percent, usd } from "@/lib/format";

import styles from "./home.module.css";

export const dynamic = "force-dynamic";

function residencyLine(violations: number | null): string {
  if (violations === null) return "Residency status unknown.";
  if (violations === 0) return "All EU data stayed in the EU: ✓";
  return `${violations} residency violations today.`;
}

export default consolePage("home", "Home", (body) => (
  <>
    <p>
      Today the gateway handled <strong>{body.requests_today ?? NA}</strong> requests costing{" "}
      <strong>{usd(body.cost_usd_today)}</strong>.
    </p>
    <ul className={styles.cards}>
      <li className={styles.card}>
        <span className={styles.label}>p95 latency:</span> <strong className={styles.value}>{ms(body.p95_seconds)}</strong>
      </li>
      <li className={styles.card}>
        <span className={styles.label}>Error rate:</span> <strong className={styles.value}>{percent(body.error_rate)}</strong>
      </li>
      <li className={styles.card}>
        <span className={styles.label}>Open incidents:</span>{" "}
        <strong className={styles.value}>{body.open_incidents ?? NA}</strong>
      </li>
      <li className={styles.card}>
        <span className={styles.value}>{residencyLine(body.residency_violations_today)}</span>
      </li>
    </ul>
    <p className="note">
      {body.last_eval === null
        ? "No eval yet."
        : `Last eval: decision accuracy ${percent(body.last_eval.decision_accuracy)}, ${body.last_eval.passed ? "passed" : "failed"}`}
    </p>
  </>
));
