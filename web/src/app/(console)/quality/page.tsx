import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { LineChart } from "@/components/ui/line-chart";
import { Table } from "@/components/ui/table";
import { shortSha } from "@/lib/format";

export const dynamic = "force-dynamic";

function seconds(iso: string): number {
  return Date.parse(iso) / 1000;
}

export default consolePage("quality", "Quality", (body) => {
  // The chart drops a run whose accuracy is absent or null, and sorts by time itself.
  const accuracy = body.runs.map((run): [number, number] => {
    const value = run.metrics.decision_accuracy;
    return [seconds(run.started_at), typeof value === "number" ? value : Number.NaN];
  });
  return (
    <>
      <LineChart
        title="Decision accuracy"
        points={accuracy}
        markers={body.release_markers.map((marker) => seconds(marker.deployed_at))}
      />

      <h2>Categories</h2>
      <Table
        rows={body.categories.map((category) => ({
          key: category.category,
          cells: [
            category.category,
            <>
              {category.passed} of {category.total}
            </>,
          ],
        }))}
      />

      <h2>Taxonomy</h2>
      <Table
        rows={body.taxonomy.map((tag) => ({
          key: tag.tag,
          cells: [tag.tag, tag.count],
        }))}
      />

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
});
