import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { LineChart } from "@/components/ui/line-chart";
import { Pager } from "@/components/ui/pager";
import { Panel } from "@/components/ui/panel";
import { StatCards, type StatCard } from "@/components/ui/stat-cards";
import { NONE_IN_WINDOW, Table } from "@/components/ui/table";
import type { GateScore } from "@/features/quality/guards";
import { NA, points, rounded, shortSha, usd } from "@/lib/format";
import { OUTCOME_SENTENCE, WINDOW_LINES, plainName, scoreLabel } from "@/lib/labels";
import { PAGE_PARAMS, pageOf } from "@/lib/paging";

export const dynamic = "force-dynamic";

/** The outcome every play axis runs 0% to 100% with these marked steps, so a flat line still shows its place. */
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

/** The newest run's scores beside the gated two, and its effort; each read from the run as stored. */
const RUN_SCORES = ["outcome_at_least_once", "reply_quality"] as const;
const RUN_EFFORT = ["mean_turns", "mean_tool_calls", "mean_seconds"] as const;
/** A mean cost per play is cents or less, so it keeps four decimals. */
const COST_DECIMALS = 4;
/** The judge's fault words in the words the page prints. */
const FAULT_WORDS: Record<string, string> = { bot: "The bot", case: "The case", eval: "The test", fine: "Fine" };
const RESULT_COUNTS = ["pass", "fail", "needs_a_look", "setup_wrong", "eval_bug"] as const;

function numberOf(metrics: Record<string, unknown> | undefined, key: string): number | null {
  const value = metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** The newest run's own cards: two scores, its effort, then one card of the five play results. */
function runCards(metrics: Record<string, unknown> | undefined): StatCard[] {
  const counts = RESULT_COUNTS.map((key) => {
    const value = numberOf(metrics, key);
    return `${scoreLabel(key)} ${value === null ? NA : value.toLocaleString("en-US")}`;
  });
  return [
    ...RUN_SCORES.map((key) => ({ label: scoreLabel(key), value: points(numberOf(metrics, key)) })),
    ...RUN_EFFORT.map((key) => {
      const value = numberOf(metrics, key);
      return { label: scoreLabel(key), value: value === null ? NA : rounded(value) };
    }),
    { label: scoreLabel("mean_cost_usd"), value: usd(numberOf(metrics, "mean_cost_usd"), COST_DECIMALS) },
    {
      label: "Results",
      value: counts.join(" · "),
      hint: "Plays in the newest run, by result. An eval bug is a play the test itself cut short; it is not scored.",
    },
  ];
}

export default consolePage("quality", "Quality", "Does the help desk still answer well enough to ship?", (body, params) => {
  const taxonomy = pageOf(body.taxonomy, params, PAGE_PARAMS.taxonomy);
  const blocked = pageOf(body.blocked, params, PAGE_PARAMS.blocked);
  // console-api sends the runs newest first.
  const newest = body.runs[0]?.metrics;
  // The chart drops a run whose score is absent or null, and sorts by time itself.
  const scores = body.runs.map((run): [number, number] => {
    const value = run.metrics.outcome_every_play;
    return [seconds(run.started_at), typeof value === "number" ? value : Number.NaN];
  });
  // Each marker sits at the run that let its release through, so the time axis spans the runs only.
  // A release with no run, or whose run is outside the window, draws no marker.
  const runTimes = new Map(body.runs.map((run) => [run.id, seconds(run.started_at)]));
  const markers = body.release_markers.flatMap((marker) => {
    const at = marker.eval_run_id === null ? undefined : runTimes.get(marker.eval_run_id);
    return at === undefined ? [] : [{ at }];
  });
  // The baseline is the gate's own outcome every play threshold; none set draws no line.
  const threshold = body.gate.find((row) => row.name === "outcome_every_play")?.threshold ?? null;
  const baseline = threshold === null ? undefined : { value: threshold, label: `Baseline ${points(threshold)}` };
  return (
    <>
      <StatCards cards={[...body.gate.map(gateCard), ...runCards(newest)]} />
      {body.gate
        .filter((row) => row.note !== null)
        .map((row) => (
          <p key={row.name} className="note">
            {row.label}: {row.note}
          </p>
        ))}
      <p className="note">{OUTCOME_SENTENCE}</p>

      <LineChart
        title="Outcome every play"
        points={scores}
        markers={markers}
        yFormat={points}
        range={SCORE_RANGE}
        steps={SCORE_STEPS}
        baseline={baseline}
      />

      <Panel title="Needs a look">
        <p>
          Plays in the newest run that broke no rule but need a person to read them, and why. Whose fault is the
          judge&apos;s guess (the bot, the case, the test, or fine); it changes no score.
        </p>
        <Table
          columns={["Case", "Play", "Why", "Whose fault"]}
          rows={body.needs_a_look.map((item) => ({
            key: `${item.case_id}/${item.play}`,
            cells: [
              <code key="case">{item.case_id}</code>,
              item.play,
              item.why,
              item.fault ? `${FAULT_WORDS[item.fault] ?? plainName(item.fault)}: ${item.fault_reason ?? ""}` : NA,
            ],
          }))}
        />
      </Panel>

      <Panel title="Categories">
        <p>
          How many help desk cases passed in each kind of conversation, in the newest run.
          Unstable cases count as not passed here.
        </p>
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
                      {`${scoreLabel(regression.metric)}: ${rounded(regression.value)} (was ${rounded(regression.previous)})`}
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
