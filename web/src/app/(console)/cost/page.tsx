import { consolePage } from "@/components/layout/console-page";
import { BarChart, type BarDay } from "@/components/ui/bar-chart";
import { DateField, FilterForm, SelectField } from "@/components/ui/filter-form";
import { Pager } from "@/components/ui/pager";
import { StatCards } from "@/components/ui/stat-cards";
import { Table } from "@/components/ui/table";
import type { CostBody } from "@/features/cost/guards";
import { readCostFilters } from "@/lib/filters";
import { NA, shortDay, smallUsd, usd } from "@/lib/format";
import { CHECK_SCORE_SENTENCE, WINDOW_LINES, scoreLabel } from "@/lib/labels";
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

/** One bar per day, oldest first, one part per tenant with its providers summed. */
function costByTenant(rows: CostBody["days"]): { days: BarDay[]; tenants: string[] } {
  const tenants = [...new Set(rows.map((row) => row.tenant))].sort();
  const byDay = new Map<string, Map<string, number>>();
  for (const row of rows) {
    const day = byDay.get(row.day) ?? new Map<string, number>();
    day.set(row.tenant, (day.get(row.tenant) ?? 0) + row.cost_usd);
    byDay.set(row.day, day);
  }
  const days = [...byDay.keys()].sort().map((day) => ({
    label: shortDay(day),
    parts: tenants.map((tenant) => ({ series: tenant, value: byDay.get(day)?.get(tenant) ?? 0 })),
  }));
  return { days, tenants };
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

export default consolePage(
  "cost",
  "Cost",
  "What does the AI cost, per day, per tenant and per model, and what did caching save?",
  async (body, params) => {
    const tradeoff = await readTradeoff(process.env.CONSOLE_TRADEOFF_PATH);
    const filters = readCostFilters(params);
    const filtered = body.days.filter(
      (row) =>
        (filters.from === undefined || row.day >= filters.from) &&
        (filters.to === undefined || row.day <= filters.to) &&
        (filters.tenant === undefined || row.tenant === filters.tenant),
    );
    const total = filtered.reduce(
      (sum, row) => ({
        requests: sum.requests + row.requests,
        cached: sum.cached + row.cached_requests,
        cost: sum.cost + row.cost_usd,
      }),
      { requests: 0, cached: 0, cost: 0 },
    );
    const days = pageOf(newestFirst(filtered), params, PAGE_PARAMS.days);
    const tenants = [...new Set(body.days.map((row) => row.tenant))].sort();
    const chart = costByTenant(body.days);
    return (
      <>
        <StatCards
          cards={[
            { label: "This month's spend", value: usd(body.month_spend_usd) },
            { label: "Cache saved", value: smallUsd(body.savings_usd_total) },
            ...body.cost_per_request.map((alias) => ({
              label: alias.alias,
              value: usd(alias.cost_usd, PER_REQUEST_DECIMALS),
            })),
          ]}
        />
        <p className="note">
          Cache saved is estimated: what the cached answers would have cost. Each alias card is the average cost of one
          request.
        </p>
        <BarChart title="Cost per day by tenant" days={chart.days} series={chart.tenants} yFormat={(value) => usd(value)} />

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

        <h2>Daily cost</h2>
        {body.rollup_populated === false && <p className="note">Daily rollup not yet populated.</p>}
        <FilterForm clearHref="/cost">
          <DateField name="from" label="From" value={filters.from} />
          <DateField name="to" label="To" value={filters.to} />
          <SelectField name="tenant" label="Tenant" options={tenants} value={filters.tenant} any="All tenants" />
        </FilterForm>
        <Table
          columns={["Day", "Tenant", "Provider", "Requests", "Cached requests", "Cost"]}
          rows={[
            ...days.rows.map((day) => ({
              key: `${day.day}/${day.tenant}/${day.provider}`,
              cells: [day.day, day.tenant, day.provider, day.requests, day.cached_requests, usd(day.cost_usd)],
            })),
            ...(filtered.length === 0
              ? []
              : [{ key: "total", cells: ["Total", "", "", total.requests, total.cached, usd(total.cost)] }]),
          ]}
        />
        <Pager name={PAGE_PARAMS.days} page={days.page} pages={days.pages} params={params} />

        <h2>Model tradeoff</h2>
        <TradeoffTable table={tradeoff} />
      </>
    );
  },
  WINDOW_LINES.cost,
);
