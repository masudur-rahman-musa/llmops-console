import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { StatCards } from "@/components/ui/stat-cards";
import { Table } from "@/components/ui/table";
import { NA, ms, percent, points, smallUsd, utcTime } from "@/lib/format";
import { WINDOW_LINES } from "@/lib/labels";

export const dynamic = "force-dynamic";

/** The eval gate verdict the quality tab shows; null means there is no baseline. */
function verdict(passed: boolean | null): string {
  return passed === null ? "no baseline to check against" : passed ? "passed" : "failed";
}

export default consolePage(
  "home",
  "Overview",
  "Is the gateway healthy today, and what has it cost?",
  (body) => (
    <>
      <StatCards
        cards={[
          { label: "Requests today", value: body.requests_today?.toLocaleString("en-US") ?? NA },
          { label: "Cost today", value: smallUsd(body.cost_usd_today) },
          { label: "p95 latency", value: ms(body.p95_seconds) },
          { label: "Error rate", value: percent(body.error_rate) },
          {
            label: "Last eval",
            value:
              body.last_eval === null
                ? "No eval yet."
                : `${points(body.last_eval.check_score)} ${verdict(body.last_eval.passed)}`,
            href: "/quality",
          },
        ]}
      />
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
  ),
  WINDOW_LINES.home,
);
