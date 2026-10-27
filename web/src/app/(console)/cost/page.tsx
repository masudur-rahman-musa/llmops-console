import { consolePage } from "@/components/layout/console-page";
import { BarChart, type BarDay } from "@/components/ui/bar-chart";
import {
  DateField,
  FilterForm,
  SelectField,
} from "@/components/ui/filter-form";
import { Pager } from "@/components/ui/pager";
import { Panel } from "@/components/ui/panel";
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
function newestFirst<
  T extends { day: string; tenant: string; provider: string },
>(rows: ReadonlyArray<T>): T[] {
  return rows.toSorted(
    (a, b) =>
      b.day.localeCompare(a.day) ||
      a.tenant.localeCompare(b.tenant) ||
      a.provider.localeCompare(b.provider),
  );
}

/** One bar per day, oldest first, one part per tenant with its providers summed. */
function costByTenant(rows: CostBody["days"]): {
  days: BarDay[];
  tenants: string[];
} {
  const tenants = [...new Set(rows.map((row) => row.tenant))].sort();
  const byDay = new Map<string, Map<string, number>>();
  for (const row of rows) {
    const day = byDay.get(row.day) ?? new Map<string, number>();
    day.set(row.tenant, (day.get(row.tenant) ?? 0) + row.cost_usd);
    byDay.set(row.day, day);
  }
  const days = [...byDay.keys()].sort().map((day) => ({
    label: shortDay(day),
    parts: tenants.map((tenant) => ({
      series: tenant,
      value: byDay.get(day)?.get(tenant) ?? 0,
    })),
  }));
  return { days, tenants };
}

/** Highest share of budget used first; a budget with no known share last; ties by tenant name. */
function worstFirst(budgets: CostBody["budgets"]): CostBody["budgets"] {
  return budgets.toSorted(
    (a, b) =>
      (b.ratio ?? Number.NEGATIVE_INFINITY) -
        (a.ratio ?? Number.NEGATIVE_INFINITY) ||
      a.tenant.localeCompare(b.tenant),
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
        columns={[
          "Option",
          "Kind",
          ...TRADEOFF_METRICS.map(scoreLabel),
          "Cost per 1,000 turns",
        ]}
        rows={table.rows.map((row) => ({
          key: row.id,
          cells: [
            row.label,
            row.kind,
            ...row.metrics.map((value) => value ?? NA),
            row.usd_per_1000_turns ?? NA,
          ],
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
    const userTenants = [
      ...new Set(body.top_users.map((user) => user.tenant)),
    ].sort();
    const users = pageOf(
      body.top_users.filter(
        (user) =>
          filters.usersTenant === undefined ||
          user.tenant === filters.usersTenant,
      ),
      params,
      PAGE_PARAMS.users,
    );
    const budgets = pageOf(
      worstFirst(body.budgets),
      params,
      PAGE_PARAMS.budgets,
    );
    const chart = costByTenant(body.days);
    return (
      <>
        <StatCards
          cards={[
            {
              label: "This month's spend",
              value: smallUsd(body.month_spend_usd),
            },
            {
              label: "Answer cache saved",
              value: smallUsd(body.savings_usd_total),
            },
            {
              label: "Prompt cache saved",
              value: smallUsd(body.prompt_cache_saved_usd),
            },
            ...body.cost_per_request.map((alias) => ({
              label: alias.alias,
              value: usd(alias.cost_usd, PER_REQUEST_DECIMALS),
            })),
          ]}
        />
        <p className="note">
          Answer cache saved is estimated: what the answers served from the
          gateway's cache would have cost. Prompt cache saved is what the input
          read from the provider's cache would have cost at the full input rate,
          less what it was billed and less the extra paid to write it. Each
          alias card is the average cost of one request.
        </p>
        <BarChart
          title="Cost per day by tenant"
          days={chart.days}
          series={chart.tenants}
          yFormat={smallUsd}
        />

        <Panel title="Budgets this month">
          {budgets.rows.map((budget) => (
            <div key={budget.tenant}>
              <p>
                {budget.tenant}: {smallUsd(budget.spent_usd)} of{" "}
                {smallUsd(budget.cap_usd)}
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
          <Pager
            name={PAGE_PARAMS.budgets}
            page={budgets.page}
            pages={budgets.pages}
            params={params}
          />
        </Panel>

        <Panel title="Top users by cost">
          {body.top_users.length === 0 ? (
            <p className="note">No user ids recorded in this window.</p>
          ) : (
            <>
              <FilterForm
                clearHref="/cost"
                keep={{
                  from: filters.from,
                  to: filters.to,
                  tenant: filters.tenant,
                }}
              >
                <SelectField
                  name="users_tenant"
                  label="Tenant"
                  options={userTenants}
                  value={filters.usersTenant}
                  any="All tenants"
                />
              </FilterForm>
              <Table
                columns={["Tenant", "User", "Requests", "Cost"]}
                rows={users.rows.map((user) => ({
                  key: `${user.tenant}/${user.user_id}`,
                  cells: [
                    user.tenant,
                    user.user_id,
                    user.requests,
                    smallUsd(user.cost_usd),
                  ],
                }))}
              />
              <Pager
                name={PAGE_PARAMS.users}
                page={users.page}
                pages={users.pages}
                params={params}
              />
            </>
          )}
        </Panel>

        <Panel title="Daily cost">
          {body.rollup_populated === false && (
            <p className="note">Daily rollup not yet populated.</p>
          )}
          <FilterForm
            clearHref="/cost"
            keep={{ users_tenant: filters.usersTenant }}
          >
            <DateField name="from" label="From" value={filters.from} />
            <DateField name="to" label="To" value={filters.to} />
            <SelectField
              name="tenant"
              label="Tenant"
              options={tenants}
              value={filters.tenant}
              any="All tenants"
            />
          </FilterForm>
          <Table
            columns={[
              "Day",
              "Tenant",
              "Provider",
              "Requests",
              "Cached requests",
              "Cost",
            ]}
            rows={[
              ...days.rows.map((day) => ({
                key: `${day.day}/${day.tenant}/${day.provider}`,
                cells: [
                  day.day,
                  day.tenant,
                  day.provider,
                  day.requests,
                  day.cached_requests,
                  smallUsd(day.cost_usd),
                ],
              })),
              ...(filtered.length === 0
                ? []
                : [
                    {
                      key: "total",
                      cells: [
                        "Total",
                        "",
                        "",
                        total.requests,
                        total.cached,
                        smallUsd(total.cost),
                      ],
                    },
                  ]),
            ]}
          />
          <Pager
            name={PAGE_PARAMS.days}
            page={days.page}
            pages={days.pages}
            params={params}
          />
        </Panel>

        <Panel title="Model tradeoff">
          <TradeoffTable table={tradeoff} />
        </Panel>
      </>
    );
  },
  WINDOW_LINES.cost,
);
