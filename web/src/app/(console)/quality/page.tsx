import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { LineChart } from "@/components/ui/line-chart";
import { Pager } from "@/components/ui/pager";
import { NONE_IN_WINDOW, Table } from "@/components/ui/table";
import type { GateScore } from "@/features/quality/guards";
import { NA, points, shortSha } from "@/lib/format";
import { CHECK_SCORE_SENTENCE, scoreLabel } from "@/lib/labels";
import { PAGE_PARAMS, pageOf } from "@/lib/paging";

export const dynamic = "force-dynamic";

function seconds(iso: string): number {
  return Date.parse(iso) / 1000;
}

/** No verdict without a baseline: the page cannot know pass or fail. */
function gateCells(row: GateScore): string[] {
  if (row.threshold === null) {
    return [row.label, points(row.value), "Baseline not set", ""];
  }
  const result = row.passed === null ? NA : row.passed ? "passed" : "failed";
  return [row.label, points(row.value), points(row.threshold), result];
}

export default consolePage("quality", "Quality", "Does the help desk still answer well enough to ship?", (body, params) => {
  const taxonomy = pageOf(body.taxonomy, params, PAGE_PARAMS.taxonomy);
  const blocked = pageOf(body.blocked, params, PAGE_PARAMS.blocked);
  // The chart drops a run whose score is absent or null, and sorts by time itself.
  const scores = body.runs.map((run): [number, number] => {
    const value = run.metrics.check_score;
    return [seconds(run.started_at), typeof value === "number" ? value : Number.NaN];
  });
  // Each marker sits at the run that let its release through, so the time axis spans the runs only.
  // A release with no run, or whose run is outside the window, draws no marker.
  const runTimes = new Map(body.runs.map((run) => [run.id, seconds(run.started_at)]));
  const markers = body.release_markers.flatMap((marker) => {
    const at = marker.eval_run_id === null ? undefined : runTimes.get(marker.eval_run_id);
    return at === undefined ? [] : [at];
  });
  return (
    <>
      <h2>Eval gate</h2>
      <p>Each score the eval gate checks, from the newest run, beside the baseline it must reach.</p>
      <Table
        columns={["Score", "Newest run", "Baseline", "Result"]}
        rows={body.gate.map((row) => ({ key: row.name, cells: gateCells(row) }))}
      />
      {body.gate
        .filter((row) => row.note !== null)
        .map((row) => (
          <p key={row.name} className="note">
            {row.label}: {row.note}
          </p>
        ))}
      <p className="note">{CHECK_SCORE_SENTENCE}</p>

      <LineChart
        title="Check score"
        points={scores}
        markers={markers}
      />

      <h2>Categories</h2>
      <p>How many help desk cases passed in each kind of conversation, in the newest run.</p>
      <Table
        columns={["Category", "Cases passed"]}
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
      <p>Why cases failed: each failure tag and how many cases carry it.</p>
      <Table
        columns={["Failure tag", "Cases"]}
        rows={taxonomy.rows.map((tag) => ({
          key: tag.tag,
          cells: [tag.tag, tag.count],
        }))}
      />
      <Pager name={PAGE_PARAMS.taxonomy} page={taxonomy.page} pages={taxonomy.pages} params={params} />

      <h2>Blocked releases</h2>
      {blocked.rows.length === 0 && <p className="note">{NONE_IN_WINDOW}</p>}
      <ul>
        {blocked.rows.map((item) => (
          <li key={`${item.git_sha}/${item.deployed_at}`}>
            <code>{shortSha(item.git_sha)}</code> <ExternalLink url={item.pr_url} label="PR" />
            {item.regressed.length === 0 ? (
              <p>No regression recorded.</p>
            ) : (
              <ul>
                {item.regressed.map((regression) => (
                  <li key={regression.metric}>
                    {`${scoreLabel(regression.metric)}: ${regression.value} (was ${regression.previous})`}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
      <Pager name={PAGE_PARAMS.blocked} page={blocked.page} pages={blocked.pages} params={params} />
    </>
  );
});
