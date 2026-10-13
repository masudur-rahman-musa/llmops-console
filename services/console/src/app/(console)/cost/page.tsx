import { SourceNotice } from "@/components/source-notice";
import { Unavailable } from "@/components/unavailable";
import { getConsolePage } from "@/lib/console-api";
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
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Option</th>
              <th>Kind</th>
              {TRADEOFF_METRICS.map((name) => (
                <th key={name}>{name}</th>
              ))}
              <th>usd_per_1000_turns</th>
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => (
              <tr key={row.id}>
                <td>{row.label}</td>
                <td>{row.kind}</td>
                {row.metrics.map((value, index) => (
                  <td key={TRADEOFF_METRICS[index]}>{value ?? NA}</td>
                ))}
                <td>{row.usd_per_1000_turns ?? NA}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>{table.conclusion}</p>
    </>
  );
}

export default async function CostPage() {
  const result = await getConsolePage("cost");
  if (!result.ok) {
    return <Unavailable title="Cost" reason={result.reason} />;
  }
  const body = result.body;
  const tradeoff = await readTradeoff(process.env.CONSOLE_TRADEOFF_PATH);
  return (
    <>
      <h1>Cost</h1>
      <SourceNotice sources={body.sources} />

      <h2>Daily cost</h2>
      {body.rollup_populated === false && <p className="note">Daily rollup not yet populated.</p>}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>Tenant</th>
              <th>Provider</th>
              <th>Requests</th>
              <th>Cached requests</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {body.days.map((day) => (
              <tr key={`${day.day}/${day.tenant}/${day.provider}`}>
                <td>{day.day}</td>
                <td>{day.tenant}</td>
                <td>{day.provider}</td>
                <td>{day.requests}</td>
                <td>{day.cached_requests}</td>
                <td>{usd(day.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Cache savings</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            {body.savings.map((saving) => (
              <tr key={`${saving.day}/${saving.tenant}`}>
                <td>{saving.day}</td>
                <td>{saving.tenant}</td>
                <td>{usd(saving.saved_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
}
