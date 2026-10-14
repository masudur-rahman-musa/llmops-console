import { LineChart } from "@/components/ui/line-chart";
import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { NA, percent } from "@/lib/format";
import type { Slo } from "@/features/reliability/guards";

export const dynamic = "force-dynamic";

/** The objective as the project notes states it. */
function objective(slo: Slo): string {
  if (slo.name === "availability") {
    return percent(slo.objective);
  }
  if (slo.name === "latency") {
    return `p95 < ${slo.objective} s`;
  }
  return String(slo.objective);
}

/** Shares as percentages; the residency count as a plain number. */
function observed(slo: Slo): string {
  if (slo.observed === null) {
    return NA;
  }
  return slo.name === "residency" ? String(slo.observed) : percent(slo.observed);
}

/** Null is no traffic, never a full or an empty budget; a negative budget is overspent and shown unclamped. */
function budget(slo: Slo): string {
  if (slo.budget_remaining === null) {
    return "no traffic";
  }
  if (slo.budget_remaining < 0) {
    return `overspent (${percent(slo.budget_remaining)})`;
  }
  return percent(slo.budget_remaining);
}

export default async function ReliabilityPage() {
  const result = await getConsolePage("reliability");
  if (!result.ok) {
    return <Unavailable title="Reliability" reason={result.reason} />;
  }
  const body = result.body;
  return (
    <>
      <h1>Reliability</h1>
      <SourceNotice sources={body.sources} />

      <h2>Error budgets</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>SLO</th>
              <th>Objective</th>
              <th>Window</th>
              <th>Observed</th>
              <th>Budget remaining</th>
            </tr>
          </thead>
          <tbody>
            {body.slos.map((slo) => (
              <tr key={slo.name}>
                <td>{slo.name}</td>
                <td>{objective(slo)}</td>
                <td>{slo.window}</td>
                <td>{observed(slo)}</td>
                <td>{budget(slo)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <LineChart title="p95 latency" points={body.latency_p95} />
      <LineChart title="Error rate" points={body.error_rate} />

      <h2>Circuit breakers</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            {body.circuits.map((circuit) => (
              <tr key={circuit.provider}>
                <td>{circuit.provider}</td>
                <td>{circuit.state}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Fallbacks</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Tenant</th>
              <th>Alias</th>
              <th>Provider</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {body.fallbacks.map((row) => (
              <tr key={`${row.tenant}/${row.alias}/${row.provider}`}>
                <td>{row.tenant}</td>
                <td>{row.alias}</td>
                <td>{row.provider}</td>
                <td>{row.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
