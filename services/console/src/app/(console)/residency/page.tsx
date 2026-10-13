import { SourceNotice } from "@/components/source-notice";
import { Unavailable } from "@/components/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { NA } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ResidencyPage() {
  const result = await getConsolePage("residency");
  if (!result.ok) {
    return <Unavailable title="Residency" reason={result.reason} />;
  }
  const body = result.body;
  return (
    <>
      <h1>Residency</h1>
      <SourceNotice sources={body.sources} />

      <h2>Requests by region</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Tenant</th>
              <th>Region</th>
              <th>Provider</th>
              <th>Requests</th>
            </tr>
          </thead>
          <tbody>
            {body.by_region.map((row) => (
              <tr key={`${row.tenant}/${row.region}/${row.provider}`}>
                <td>{row.tenant}</td>
                <td>{row.region}</td>
                <td>{row.provider}</td>
                <td>{row.requests}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>Violations: {body.violations ?? NA}</p>

      <h2>Redactions</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Tenant</th>
              <th>Kind</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {body.redactions.map((row) => (
              <tr key={`${row.tenant}/${row.kind}`}>
                <td>{row.tenant}</td>
                <td>{row.kind}</td>
                <td>{row.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
