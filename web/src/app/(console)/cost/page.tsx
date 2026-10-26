import { consolePage } from "@/components/layout/console-page";
import { Pager } from "@/components/ui/pager";
import { Table } from "@/components/ui/table";
import { NA, smallUsd, usd } from "@/lib/format";
import { CHECK_SCORE_SENTENCE, scoreLabel } from "@/lib/labels";
import { PAGE_PARAMS, pageOf } from "@/lib/paging";
import { TRADEOFF_METRICS, readTradeoff, type Tradeoff } from "@/lib/tradeoff";

export const dynamic = "force-dynamic";

const PER_REQUEST_DECIMALS = 4;

/** Newest day first, then tenant, then provider. console-api's own order is left alone. */
function newestFirst<T extends { day: string; tenant: string; provider: string }>(rows: ReadonlyArray<T>): T[] {
  return rows.toSorted(
    (a, b) => b.day.localeCompare(a.day) || a.tenant.localeCompare(b.tenant) || a.provider.localeCompare(b.provider),
  );
}

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
        columns={["Option", "Kind", ...TRADEOFF_METRICS.map(scoreLabel), "Cost per 1,000 turns"]}
        rows={table.rows.map((row) => ({
          key: row.id,
          cells: [row.label, row.kind, ...row.metrics.map((value) => value ?? NA), row.usd_per_1000_turns ?? NA],
        }))}
      />
      <p>{table.conclusion}</p>
      <p className="note">{CHECK_SCORE_SENTENCE}</p>
    </>
  );
}

export default consolePage("cost", "Cost", "What does the AI cost, per day, per tenant and per model, and what did caching save?", async (body, params) => {
  const tradeoff = await readTradeoff(process.env.CONSOLE_TRADEOFF_PATH);
  const days = pageOf(newestFirst(body.days), params, PAGE_PARAMS.days);
  const savings = pageOf(body.savings, params, PAGE_PARAMS.savings);
  return (
    <>
      <h2>Daily cost</h2>
      {body.rollup_populated === false && <p className="note">Daily rollup not yet populated.</p>}
      <Table
        columns={["Day", "Tenant", "Provider", "Requests", "Cached requests", "Cost"]}
        rows={days.rows.map((day) => ({
          key: `${day.day}/${day.tenant}/${day.provider}`,
          cells: [day.day, day.tenant, day.provider, day.requests, day.cached_requests, usd(day.cost_usd)],
        }))}
      />
      <Pager name={PAGE_PARAMS.days} page={days.page} pages={days.pages} params={params} />

      <h2>Cache savings</h2>
      <Table
        columns={["Day", "Tenant", "Saved"]}
        rows={savings.rows.map((saving) => ({
          key: `${saving.day}/${saving.tenant}`,
          cells: [saving.day, saving.tenant, smallUsd(saving.saved_usd)],
        }))}
      />
      <Pager name={PAGE_PARAMS.savings} page={savings.page} pages={savings.pages} params={params} />
      <p>Estimated cache savings: {smallUsd(body.savings_usd_total)}</p>

      <h2>Budgets this month</h2>
      {body.budgets.map((budget) => (
        <div key={budget.tenant}>
          <p>
            {budget.tenant}: {usd(budget.spent_usd)} of {usd(budget.cap_usd)}
            {budget.ratio !== null && budget.ratio > 1 && ", over"}
          </p>
          {budget.ratio === null ? (
            <p className="note">{NA}</p>
          ) : (
            <div className={budget.ratio > 1 ? "bar over" : "bar"}>
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
