import { SourceNotice } from "@/components/source-notice";
import { Unavailable } from "@/components/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { NA, ms, percent, usd } from "@/lib/format";

export const dynamic = "force-dynamic";

function residencyLine(violations: number | null): string {
  if (violations === null) return "Residency status unknown.";
  if (violations === 0) return "All EU data stayed in the EU: ✓";
  return `${violations} residency violations today.`;
}

export default async function HomePage() {
  const result = await getConsolePage("home");
  if (!result.ok) {
    return <Unavailable title="Home" reason={result.reason} />;
  }
  const body = result.body;
  return (
    <>
      <h1>Home</h1>
      <SourceNotice sources={body.sources} />
      <p>
        Today the gateway handled {body.requests_today ?? NA} requests costing {usd(body.cost_usd_today)}.
      </p>
      <ul>
        <li>p95 latency: {ms(body.p95_seconds)}</li>
        <li>Error rate: {percent(body.error_rate)}</li>
        <li>Open incidents: {body.open_incidents ?? NA}</li>
      </ul>
      <p>
        {body.last_eval === null
          ? "No eval yet."
          : `Last eval: decision accuracy ${percent(body.last_eval.decision_accuracy)}, ${body.last_eval.passed ? "passed" : "failed"}`}
      </p>
      <p>{residencyLine(body.residency_violations_today)}</p>
    </>
  );
}
