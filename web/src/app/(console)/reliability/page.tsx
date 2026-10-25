import { consolePage } from "@/components/layout/console-page";
import { LineChart } from "@/components/ui/line-chart";
import { Table } from "@/components/ui/table";
import { NA, percent } from "@/lib/format";
import type { Slo } from "@/features/reliability/guards";

export const dynamic = "force-dynamic";

/** The objective as the reliability doc states it. */
function objective(slo: Slo): string {
  if (slo.name === "availability") {
    return percent(slo.objective);
  }
  if (slo.name === "latency") {
    return `p95 < ${slo.objective} s`;
  }
  return String(slo.objective);
}

/** The observed share as a percentage. */
function observed(slo: Slo): string {
  return slo.observed === null ? NA : percent(slo.observed);
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

export default consolePage("reliability", "Reliability", "Is the gateway up and fast enough, and which providers are failing?", (body) => (
  <>
    <h2>Error budgets</h2>
    <Table
      columns={["SLO", "Objective", "Window", "Observed", "Budget remaining"]}
      rows={body.slos.map((slo) => ({
        key: slo.name,
        cells: [slo.name, objective(slo), slo.window, observed(slo), budget(slo)],
      }))}
    />

    <LineChart title="p95 latency" points={body.latency_p95} />
    <LineChart title="Error rate" points={body.error_rate} />

    <h2>Circuit breakers</h2>
    <Table
      columns={["Provider", "State"]}
      rows={body.circuits.map((circuit) => ({
        key: circuit.provider,
        cells: [circuit.provider, circuit.state],
      }))}
    />

    <h2>Fallbacks</h2>
    <Table
      columns={["Tenant", "Alias", "Provider", "Count"]}
      rows={body.fallbacks.map((row) => ({
        key: `${row.tenant}/${row.alias}/${row.provider}`,
        cells: [row.tenant, row.alias, row.provider, row.count],
      }))}
    />
  </>
));
