import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { Table } from "@/components/ui/table";
import { NA, percent, shortSha, usd, utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

function count(value: number | null): string {
  return value === null ? NA : Math.round(value).toLocaleString("en-US");
}

export default consolePage(
  "requests",
  "Requests",
  "What did the gateway hide, how many tokens went in and out, how often did the cache answer, and where is each request's trace?",
  (body) => (
    <>
      <p>Cache hit rate: {percent(body.cache_hit_rate)}</p>
      <p>Guardrail blocks (one person asking for and approving the same thing): {count(body.guardrail_blocks)}</p>
      <h2>Redactions by kind</h2>
      <Table
        columns={["Kind", "Values hidden"]}
        rows={body.redactions.map((row) => ({ key: row.kind, cells: [row.kind, count(row.count)] }))}
      />
      <h2>Help desk refusals by reason</h2>
      <Table
        columns={["Reason", "Refusals"]}
        rows={body.refusals.map((row) => ({ key: row.reason, cells: [row.reason, count(row.count)] }))}
      />
      <h2>Tokens per model</h2>
      <Table
        columns={["Model", "Tokens in", "Tokens out"]}
        rows={body.tokens.map((row) => ({ key: row.model, cells: [row.model, count(row.input), count(row.output)] }))}
      />
      <h2>Newest requests</h2>
      <Table
        columns={["Time", "Tenant", "Model", "Tokens in / out", "Cost", "Cache", "Prompt version", "Trace"]}
        rows={body.requests.map((row, index) => ({
          key: `${row.ts}/${index}`,
          cells: [
            utcTime(row.ts),
            row.tenant,
            row.model,
            `${count(row.input_tokens)} / ${count(row.output_tokens)}`,
            usd(row.cost_usd, 6),
            row.cached ? "answered from cache" : "sent to the model",
            row.prompt_version === null ? "none sent" : <code key="pv">{shortSha(row.prompt_version)}</code>,
            <ExternalLink key="trace" url={row.trace_url} label="Open trace" />,
          ],
        }))}
      />
    </>
  ),
);
