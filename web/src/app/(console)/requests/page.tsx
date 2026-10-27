import { consolePage } from "@/components/layout/console-page";
import { BarChart } from "@/components/ui/bar-chart";
import {
  CheckField,
  DateField,
  FilterForm,
  SelectField,
} from "@/components/ui/filter-form";
import { ExternalLink } from "@/components/ui/external-link";
import { Pager } from "@/components/ui/pager";
import { StatCards } from "@/components/ui/stat-cards";
import { Table } from "@/components/ui/table";
import { readRequestFilters } from "@/lib/filters";
import { NA, percent, shortDay, shortSha, usd, utcTime } from "@/lib/format";
import { WINDOW_LINES, plainName, redactionLabel } from "@/lib/labels";
import { PAGE_PARAMS } from "@/lib/paging";

export const dynamic = "force-dynamic";

const ANSWERED = "Answered";
const FAILED = "Failed (5xx)";

function latency(value: number | null): string {
  return value === null
    ? NA
    : `${Math.round(value).toLocaleString("en-US")} ms`;
}

function count(value: number | null): string {
  return value === null ? NA : Math.round(value).toLocaleString("en-US");
}

const filters = readRequestFilters;

export default consolePage(
  "requests",
  "Requests",
  "What did the gateway hide, how many tokens went in and out, how often did the cache answer, and where is each request's trace?",
  (body, params) => (
    <>
      <StatCards
        cards={[
          {
            label: "Requests",
            value:
              body.per_day.length === 0
                ? NA
                : count(
                    body.per_day.reduce((sum, day) => sum + day.requests, 0),
                  ),
          },
          {
            label: "Answer cache hit rate",
            value: percent(body.cache_hit_rate),
            hint: "Answered by the gateway's own cache, with no model call.",
          },
          {
            label: "Prompt cache share",
            value: percent(body.prompt_cache_share),
            hint: "Input tokens the provider read from its cache, billed cheaper.",
          },
          {
            label: "Values hidden",
            value: count(
              body.redactions.reduce((sum, row) => sum + row.count, 0),
            ),
          },
          {
            label: "Self-approvals blocked",
            value: count(body.guardrail_blocks),
            hint: "The help desk stops a person approving their own request.",
          },
        ]}
      />
      {/* A null errors value (Prometheus down) draws no Failed part; the source notice says why. */}
      <BarChart
        title="Requests and errors per day"
        series={[ANSWERED, FAILED]}
        days={body.per_day.map((day) => ({
          label: shortDay(day.day),
          parts: [
            { series: ANSWERED, value: day.requests },
            { series: FAILED, value: day.errors ?? 0 },
          ],
        }))}
      />
      <h2>Redactions by kind</h2>
      <Table
        columns={["Kind", "Values hidden"]}
        rows={body.redactions.map((row) => ({
          key: row.kind,
          cells: [redactionLabel(row.kind), count(row.count)],
        }))}
      />
      <h2>Help desk refusals by reason</h2>
      <Table
        columns={["Reason", "Refusals"]}
        rows={body.refusals.map((row) => ({
          key: row.reason,
          cells: [plainName(row.reason), count(row.count)],
        }))}
      />
      <h2>Tokens per model</h2>
      <Table
        columns={["Model", "Tokens in", "Tokens out"]}
        rows={body.tokens.map((row) => ({
          key: row.model,
          cells: [row.model, count(row.input), count(row.output)],
        }))}
      />
      <h2>Newest requests</h2>
      <FilterForm clearHref="/requests">
        <DateField name="day" label="Day" value={filters(params).day} />
        <SelectField
          name="tenant"
          label="Tenant"
          options={body.tenants}
          value={filters(params).tenant}
          any="All tenants"
        />
        <SelectField
          name="model"
          label="Model"
          options={body.models}
          value={filters(params).model}
          any="All models"
        />
        <CheckField
          name="errors"
          label="Errors only"
          checked={filters(params).errors}
        />
        <CheckField
          name="slow"
          label="Slow only"
          checked={filters(params).slow}
        />
        <CheckField
          name="cached"
          label="Cache hits"
          checked={filters(params).cached}
        />
      </FilterForm>
      {body.slow_ms !== null && (
        <p className="note">Slow means above {latency(body.slow_ms)}.</p>
      )}
      <ExternalLink url={body.langfuse_url} label="Open these in Langfuse" />
      <Table
        columns={[
          "Time",
          "Trace",
          "Tenant",
          "Model",
          "Status",
          "Latency",
          "Tokens in / out",
          "Cost",
          "Cache",
          "Prompt version",
        ]}
        rows={body.requests.map((row, index) => ({
          key: `${row.ts}/${index}`,
          cells: [
            utcTime(row.ts),
            <ExternalLink key="trace" url={row.trace_url} label="Open trace" />,
            row.tenant,
            row.model,
            String(row.status),
            latency(row.latency_ms),
            `${count(row.input_tokens)} / ${count(row.output_tokens)}`,
            usd(row.cost_usd, 6),
            row.cached ? "answered from cache" : "sent to the model",
            row.prompt_version === null ? (
              "none sent"
            ) : (
              <code key="pv">{shortSha(row.prompt_version)}</code>
            ),
          ],
        }))}
      />
      <Pager
        name={PAGE_PARAMS.requests}
        page={body.requests_page}
        pages={body.requests_pages}
        params={params}
      />
    </>
  ),
  WINDOW_LINES.requests,
);
