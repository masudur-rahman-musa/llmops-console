import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { Panel } from "@/components/ui/panel";
import { StatCards } from "@/components/ui/stat-cards";
import { Table } from "@/components/ui/table";
import { NA, longDay, ms, percent, points, smallUsd, utcTime } from "@/lib/format";
import { WINDOW_LINES } from "@/lib/labels";

export const dynamic = "force-dynamic";

/** The eval gate verdict the quality tab shows; null means there is no baseline. */
function verdict(passed: boolean | null): string {
  return passed === null ? "no baseline to check against" : passed ? "passed" : "failed";
}

/** The newest run's score, with the baseline it is checked against when one is set, and the verdict. */
function lastEvalLine(last: { outcome_every_play: number | null; passed: boolean | null; threshold?: number | null }): string {
  const score = `Outcome every play ${points(last.outcome_every_play)}`;
  const threshold = last.threshold ?? null;
  return threshold === null
    ? `${score} ${verdict(last.passed)}`
    : `${score}, baseline ${points(threshold)}, ${verdict(last.passed)}`;
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
            value: body.last_eval === null ? "No eval yet." : lastEvalLine(body.last_eval),
            sub: body.last_eval === null ? undefined : `Last run ${longDay(body.last_eval.finished_at)}`,
            href: "/quality",
          },
        ]}
      />
      <Panel title="Model routing">
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
      </Panel>
    </>
  ),
  WINDOW_LINES.home,
);
