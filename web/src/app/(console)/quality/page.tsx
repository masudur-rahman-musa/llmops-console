import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { LineChart } from "@/components/ui/line-chart";
import { Pager } from "@/components/ui/pager";
import { Panel } from "@/components/ui/panel";
import { StatCards, type StatCard } from "@/components/ui/stat-cards";
import { NONE_IN_WINDOW, Table } from "@/components/ui/table";
import type { GateScore } from "@/features/quality/guards";
import { NA, points, shortSha } from "@/lib/format";
import { CHECK_SCORE_SENTENCE, WINDOW_LINES, plainName, scoreLabel } from "@/lib/labels";
import { PAGE_PARAMS, pageOf } from "@/lib/paging";

export const dynamic = "force-dynamic";

/** Short enough that two release labels never touch on the chart. */
const MARKER_LABEL_LENGTH = 7;

/** The check score axis runs 0% to 100% with these marked steps, so a flat line still shows its place. */
const SCORE_RANGE = [0, 100] as const;
const SCORE_STEPS = [0, 25, 50, 75, 100];

function seconds(iso: string): number {
  return Date.parse(iso) / 1000;
}

/** The newest run's score beside its baseline; no verdict without a baseline, since the page cannot know pass or fail. */
function gateCard(row: GateScore): StatCard {
  if (row.threshold === null) {
    return { label: row.label, value: points(row.value), sub: "Baseline not set" };
  }
  const result = row.passed === null ? NA : row.passed ? "passed" : "failed";
  return { label: row.label, value: points(row.value), sub: `Baseline ${points(row.threshold)} ${result}` };
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
    return at === undefined ? [] : [{ at, label: marker.git_sha.slice(0, MARKER_LABEL_LENGTH) }];
  });
  // The baseline is the gate's own check score threshold; none set draws no line.
  const threshold = body.gate.find((row) => row.name === "check_score")?.threshold ?? null;
  const baseline = threshold === null ? undefined : { value: threshold, label: `Baseline ${points(threshold)}` };
  return (
    <>
      <StatCards cards={body.gate.map(gateCard)} />
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
        yFormat={points}
        range={SCORE_RANGE}
        steps={SCORE_STEPS}
        baseline={baseline}
      />

      <Panel title="Categories">
        <p>How many help desk cases passed in each kind of conversation, in the newest run.</p>
        <Table
          columns={["Category", "Cases passed"]}
          rows={body.categories.map((category) => ({
            key: category.category,
            cells: [
              plainName(category.category),
              <>
                {category.passed} of {category.total}
              </>,
            ],
          }))}
        />
      </Panel>

      <Panel title="Taxonomy">
        <p>Why cases failed: each failure tag and how many cases carry it.</p>
        <Table
          columns={["Failure tag", "Cases"]}
          rows={taxonomy.rows.map((tag) => ({
            key: tag.tag,
            cells: [plainName(tag.tag), tag.count],
          }))}
        />
        <Pager name={PAGE_PARAMS.taxonomy} page={taxonomy.page} pages={taxonomy.pages} params={params} />
      </Panel>

      <Panel title="Blocked releases">
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
      </Panel>
    </>
  );
}, WINDOW_LINES.quality);
