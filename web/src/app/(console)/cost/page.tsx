import { consolePage } from "@/components/layout/console-page";
import { Table } from "@/components/ui/table";
import { NA, usd } from "@/lib/format";
import { TRADEOFF_METRICS, readTradeoff, type Tradeoff } from "@/lib/tradeoff";

export const dynamic = "force-dynamic";

const PER_REQUEST_DECIMALS = 4;

/** `min(ratio, 1)` as a whole-page percentage, rounded so float noise never reaches the style. */
function barWidth(ratio: number): string {
  return `${Math.round(Math.min(Math.max(ratio, 0), 1) * 10000) / 100}%`;
}

function TradeoffTable({ table }: { table: Tradeoff | null }) {
  if (table === null) {
    return <p className="note">Not measured yet.</p>;
  }
  return (
    <>
      <Table
        columns={["Option", "Kind", ...TRADEOFF_METRICS, "usd_per_1000_turns"]}
        rows={table.rows.map((row) => ({
          key: row.id,
          cells: [row.label, row.kind, ...row.metrics.map((value) => value ?? NA), row.usd_per_1000_turns ?? NA],
        }))}
      />
      <p>{table.conclusion}</p>
    </>
  );
}

export default consolePage("cost", "Cost", async (body) => {
  const tradeoff = await readTradeoff(process.env.CONSOLE_TRADEOFF_PATH);
  return (
    <>
      <h2>Daily cost</h2>
      {body.rollup_populated === false && <p className="note">Daily rollup not yet populated.</p>}
      <Table
        columns={["Day", "Tenant", "Provider", "Requests", "Cached requests", "Cost"]}
        rows={body.days.map((day) => ({
          key: `${day.day}/${day.tenant}/${day.provider}`,
          cells: [day.day, day.tenant, day.provider, day.requests, day.cached_requests, usd(day.cost_usd)],
        }))}
      />

      <h2>Cache savings</h2>
      <Table
        rows={body.savings.map((saving) => ({
          key: `${saving.day}/${saving.tenant}`,
          cells: [saving.day, saving.tenant, usd(saving.saved_usd)],
        }))}
      />
      <p>Estimated cache savings: {usd(body.savings_usd_total)}</p>

      <h2>Budgets this month</h2>
      {body.budgets.map((budget) => (
        <div key={budget.tenant}>
          <p>
            {budget.tenant}: {usd(budget.spent_usd)} of {usd(budget.cap_usd)}
          </p>
          {budget.ratio === null ? (
            <p className="note">{NA}</p>
          ) : (
            <div className="bar">
              <div style={{ width: barWidth(budget.ratio) }} />
            </div>
          )}
        </div>
      ))}

      <section>
        <h2>What one request costs</h2>
        {body.cost_per_request.map((alias) => (
          <p key={alias.alias}>
            {alias.alias}: {usd(alias.cost_usd, PER_REQUEST_DECIMALS)}
          </p>
        ))}
      </section>

      <h2>Model tradeoff</h2>
      <TradeoffTable table={tradeoff} />
    </>
  );
});
