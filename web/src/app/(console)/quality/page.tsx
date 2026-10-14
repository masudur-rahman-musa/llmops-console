import { ExternalLink } from "@/components/ui/external-link";
import { LineChart } from "@/components/ui/line-chart";
import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { shortSha } from "@/lib/format";

export const dynamic = "force-dynamic";

function seconds(iso: string): number {
  return Date.parse(iso) / 1000;
}

export default async function QualityPage() {
  const result = await getConsolePage("quality");
  if (!result.ok) {
    return <Unavailable title="Quality" reason={result.reason} />;
  }
  const body = result.body;
  // The chart drops a run whose accuracy is absent or null, and sorts by time itself.
  const accuracy = body.runs.map((run): [number, number] => {
    const value = run.metrics.decision_accuracy;
    return [seconds(run.started_at), typeof value === "number" ? value : Number.NaN];
  });
  return (
    <>
      <h1>Quality</h1>
      <SourceNotice sources={body.sources} />

      <LineChart
        title="Decision accuracy"
        points={accuracy}
        markers={body.release_markers.map((marker) => seconds(marker.deployed_at))}
      />

      <h2>Categories</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            {body.categories.map((category) => (
              <tr key={category.category}>
                <td>{category.category}</td>
                <td>
                  {category.passed} of {category.total}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Taxonomy</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            {body.taxonomy.map((tag) => (
              <tr key={tag.tag}>
                <td>{tag.tag}</td>
                <td>{tag.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Blocked releases</h2>
      <ul>
        {body.blocked.map((item) => (
          <li key={`${item.git_sha}/${item.deployed_at}`}>
            <code>{shortSha(item.git_sha)}</code> <ExternalLink url={item.pr_url} label="PR" />
            {item.regressed.length === 0 ? (
              <p>No regression recorded.</p>
            ) : (
              <ul>
                {item.regressed.map((regression) => (
                  <li key={regression.metric}>
                    {`${regression.metric}: ${regression.value} (was ${regression.previous})`}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </>
  );
}
