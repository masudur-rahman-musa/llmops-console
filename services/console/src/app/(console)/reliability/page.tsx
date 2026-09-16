import { LineChart } from "@/components/line-chart";
import { SourceNotice } from "@/components/source-notice";
import { Unavailable } from "@/components/unavailable";
import { getConsolePage } from "@/lib/console-api";

export const dynamic = "force-dynamic";

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

      <LineChart title="p95 latency" points={body.latency_p95} />
      <LineChart title="Error rate" points={body.error_rate} />

      <h2>Circuit breakers</h2>
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

      <h2>Fallbacks</h2>
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
    </>
  );
}
