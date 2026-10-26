import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { Table } from "@/components/ui/table";
import { NA, ms, percent, points, usd, utcTime } from "@/lib/format";
import { CHECK_SCORE_SENTENCE } from "@/lib/labels";

import styles from "./home.module.css";

export const dynamic = "force-dynamic";

/** The eval gate verdict the quality tab shows; null means there is no baseline. */
function verdict(passed: boolean | null): string {
  return passed === null ? "no baseline to check against" : passed ? "passed" : "failed";
}

export default consolePage("home", "Overview", "Is the gateway healthy today, and what has it cost?", (body) => (
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
    </ul>
    <p className="note">
      {body.last_eval === null
        ? "No eval yet."
        : `Last eval: check score ${points(body.last_eval.check_score)}, ${verdict(body.last_eval.passed)}`}
    </p>
    <p className="note">{CHECK_SCORE_SENTENCE}</p>
    <h2>Model routing</h2>
    <Table
      columns={["Alias", "Purpose", "Providers in fallback order"]}
      rows={body.routing.aliases.map((alias) => ({
        key: alias.name,
        cells: [
          <code key="name">{alias.name}</code>,
          alias.purpose ?? "no purpose given",
          alias.candidates.map((c, index) => `${index + 1}. ${c.provider} ${c.model}`).join(", "),
        ],
      }))}
    />
    <p className="note">Policy last loaded {utcTime(body.routing.loaded_at)}.</p>
    <p>
      {body.grafana_url === null ? (
        "Grafana link not set."
      ) : (
        <ExternalLink url={body.grafana_url} label="Health details in Grafana" />
      )}
    </p>
  </>
));
