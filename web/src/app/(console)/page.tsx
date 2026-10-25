import { consolePage } from "@/components/layout/console-page";
import { NA, ms, percent, usd } from "@/lib/format";
import { CHECK_SCORE_SENTENCE } from "@/lib/labels";

import styles from "./home.module.css";

export const dynamic = "force-dynamic";

export default consolePage("home", "Home", "Is the gateway healthy today, and what has it cost?", (body) => (
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
    </ul>
    <p className="note">
      {body.last_eval === null
        ? "No eval yet."
        : `Last eval: check score ${percent(body.last_eval.check_score)}, ${body.last_eval.passed ? "passed" : "failed"}`}
    </p>
    <p className="note">{CHECK_SCORE_SENTENCE}</p>
  </>
));
