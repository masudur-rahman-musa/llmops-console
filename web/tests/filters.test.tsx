import { readFileSync } from "node:fs";

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CostPage from "@/app/(console)/cost/page";
import RequestsPage from "@/app/(console)/requests/page";
import { readCostFilters, readRequestFilters } from "@/lib/filters";

type Fixture = Record<string, unknown>;
type Params = Record<string, string | string[] | undefined>;
type Row = Record<string, unknown>;

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const LANGFUSE = "https://cloud.langfuse.com/project/demo/traces?filter=Level;string;;%3D;ERROR";

function fixture(name: string): Fixture {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8")) as Fixture;
}

function stubFetch(body: unknown): ReturnType<typeof vi.fn> {
  const fetch = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function props(params: Params): { searchParams: Promise<Params> } {
  return { searchParams: Promise.resolve(params) };
}

function requestedUrl(fetch: ReturnType<typeof vi.fn>): string {
  expect(fetch).toHaveBeenCalledTimes(1);
  return String(fetch.mock.calls[0][0]);
}

function tables(html: string): string[] {
  return Array.from(html.matchAll(/<table\b[\s\S]*?<\/table>/g), (match) => match[0]);
}

function headers(table: string): string[] {
  return Array.from(table.matchAll(/<th\b[^>]*>([\s\S]*?)<\/th>/g), (match) => text(match[1]));
}

function bodyRows(table: string): string[][] {
  const body = table.split("<tbody>")[1] ?? "";
  return Array.from(body.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), (row) =>
    Array.from(row[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g), (cell) => text(cell[1])),
  );
}

function tableWithHeader(html: string, header: string): string {
  const found = tables(html).filter((table) => headers(table).includes(header));
  expect(found, header).toHaveLength(1);
  return found[0];
}

function anchors(html: string): Array<{ href: string; label: string; tag: string }> {
  return Array.from(html.matchAll(/<a\b([^>]*)\bhref="([^"]*)"([^>]*)>([\s\S]*?)<\/a>/g), (match) => ({
    href: match[2].replace(/&amp;/g, "&"),
    label: text(match[4]),
    tag: match[0],
  }));
}

function labels(html: string): string[] {
  return Array.from(html.matchAll(/<label\b[^>]*>([\s\S]*?)<\/label>/g), (match) => text(match[1]));
}

function formOf(html: string): string {
  const forms = Array.from(html.matchAll(/<form\b[\s\S]*?<\/form>/g), (match) => match[0]);
  expect(forms).toHaveLength(1);
  return forms[0];
}

function controlNames(form: string): string[] {
  return Array.from(form.matchAll(/<(?:input|select|textarea)\b[^>]*\bname="([^"]*)"/g), (match) => match[1]);
}

function optionValues(form: string, name: string): string[] {
  const select = form.match(new RegExp(`<select\\b[^>]*name="${name}"[^>]*>([\\s\\S]*?)</select>`));
  expect(select, `select ${name}`).not.toBeNull();
  return Array.from((select?.[1] ?? "").matchAll(/<option\b[^>]*value="([^"]*)"/g), (match) => match[1]);
}

function selectedOption(form: string, name: string): string | undefined {
  const select = form.match(new RegExp(`<select\\b[^>]*name="${name}"[^>]*>([\\s\\S]*?)</select>`));
  const chosen = (select?.[1] ?? "").match(/<option\b(?=[^>]*\bselected\b)[^>]*value="([^"]*)"/);
  return chosen?.[1];
}

function inputTag(form: string, name: string): string {
  const tag = form.match(new RegExp(`<input\\b[^>]*name="${name}"[^>]*>`));
  expect(tag, `input ${name}`).not.toBeNull();
  return tag?.[0] ?? "";
}

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.stubEnv("CONSOLE_TRADEOFF_PATH", "");
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const EVERYTHING: Params = {
  requests_page: "1",
  day: "2026-10-27",
  tenant: "acme",
  model: "gpt-4o-mini",
  errors: "true",
  slow: "true",
  cached: "true",
};

describe("requests filters", () => {
  it("requests form shows every filter with its options and current values", async () => {
    // Catches a form that is missing a filter, builds the tenant and model choices from anything but the body's lists, loses the values in the address, or draws a box ticked when its value is not "true".
    const body = fixture("requests");
    body.tenants = ["acme", "nordlicht"];
    body.models = ["claude-haiku-5-5", "gpt-4o-mini"];
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props(EVERYTHING)));

    const form = formOf(html);
    expect(labels(form)).toEqual(
      expect.arrayContaining(["Day", "Tenant", "Model", "Errors only", "Slow only", "Cache hits"]),
    );
    expect(controlNames(form).sort()).toEqual(["cached", "day", "errors", "model", "slow", "tenant"]);
    expect(optionValues(form, "tenant")).toEqual(expect.arrayContaining(["acme", "nordlicht"]));
    expect(optionValues(form, "model")).toEqual(expect.arrayContaining(["claude-haiku-5-5", "gpt-4o-mini"]));
    expect(optionValues(form, "tenant")).not.toContain("cuan");
    expect(selectedOption(form, "tenant")).toBe("acme");
    expect(selectedOption(form, "model")).toBe("gpt-4o-mini");
    expect(inputTag(form, "day")).toContain('value="2026-10-27"');
    for (const name of ["errors", "slow", "cached"]) {
      expect(inputTag(form, name), name).toContain('type="checkbox"');
      expect(inputTag(form, name), name).toContain('value="true"');
      expect(inputTag(form, name), name).toMatch(/\bchecked\b/);
    }
    expect(form).toContain('method="get"');
    expect(text(form)).toContain("Apply");

    const bare = renderToStaticMarkup(await RequestsPage(props({})));
    const emptyForm = formOf(bare);
    expect(selectedOption(emptyForm, "tenant") ?? "").toBe("");
    expect(inputTag(emptyForm, "errors")).not.toMatch(/\bchecked\b/);
    expect(inputTag(emptyForm, "day")).not.toContain("2026");
  });

  it("filters reach console-api", async () => {
    // Catches a page that never sends a filter, sends one under another name, sends a false box as a filter, or orders the query differently from the one console-api is documented with.
    const fetch = stubFetch(fixture("requests"));

    await RequestsPage(props(EVERYTHING));

    expect(requestedUrl(fetch)).toBe(
      `${API}/console/requests?page=1&day=2026-10-27&tenant=acme&model=gpt-4o-mini&errors=true&slow=true&cached=true`,
    );
  });

  it("only the filters that are set are sent, beside the page", async () => {
    // Catches a page that sends empty values (`tenant=`) or false booleans for filters nobody set.
    const fetch = stubFetch(fixture("requests"));

    await RequestsPage(props({ requests_page: "3", slow: "true" }));

    expect(requestedUrl(fetch)).toBe(`${API}/console/requests?page=3&slow=true`);
  });

  it("a model with slashes and colons is sent, encoded as a query value", async () => {
    // Catches a model check copied from the tenant rule (dots, colons and slashes are legal in a model id) and a value pasted into the query without encoding.
    const fetch = stubFetch(fixture("requests"));

    await RequestsPage(props({ model: "org/qwen2.5:3b" }));

    const url = new URL(requestedUrl(fetch));
    expect(url.searchParams.get("model")).toBe("org/qwen2.5:3b");
  });

  const BAD: Array<[string, Params, string]> = [
    ["a tenant with a space", { tenant: "Bad Tenant" }, "tenant"],
    ["an uppercase tenant", { tenant: "Acme" }, "tenant"],
    ["a tenant with a trailing newline", { tenant: "acme\n" }, "tenant"],
    ["a tenant with a trailing space", { tenant: "acme " }, "tenant"],
    ["a tenant of 65 characters", { tenant: "t".repeat(65) }, "tenant"],
    ["a month 13", { day: "2026-13-01" }, "day"],
    ["a day that does not exist", { day: "2026-02-30" }, "day"],
    ["a day without zeros", { day: "2026-1-5" }, "day"],
    ["a day with a trailing newline", { day: "2026-10-27\n" }, "day"],
    ["a model with a semicolon", { model: "gpt;4o" }, "model"],
    ["a model with a comma", { model: "gpt,4o" }, "model"],
    ["a model of 129 characters", { model: "m".repeat(129) }, "model"],
    ["a model with a trailing newline", { model: "gpt-4o\n" }, "model"],
    ["errors=1", { errors: "1" }, "errors"],
    ["errors=TRUE", { errors: "TRUE" }, "errors"],
    ["slow=yes", { slow: "yes" }, "slow"],
    ["cached=false", { cached: "false" }, "cached"],
  ];

  it.each(BAD)("a bad value in the address is dropped: %s", async (_name, params, dropped) => {
    // Catches a page that sends a value the checks would refuse (an error page for a hand-typed link) or that fails to render: the bad filter leaves the call, a good one beside it stays, and the page still draws.
    const fetch = stubFetch(fixture("requests"));

    const html = renderToStaticMarkup(await RequestsPage(props({ ...params, tenant: params.tenant ?? "acme" })));

    const url = new URL(requestedUrl(fetch));
    if (dropped === "tenant") {
      expect(url.searchParams.has("tenant")).toBe(false);
    } else {
      expect(url.searchParams.has(dropped)).toBe(false);
      expect(url.searchParams.get("tenant")).toBe("acme");
    }
    expect(url.searchParams.get("page")).toBe("1");
    expect(text(html)).toContain("Newest requests");
    expect(text(html)).not.toContain("Console data unavailable");
  });

  it("a model of 128 characters and a tenant of 64 are kept", async () => {
    // Catches an off-by-one on either cap: the inclusive edge passes the web check, one more is dropped above.
    const fetch = stubFetch(fixture("requests"));

    await RequestsPage(props({ model: "m".repeat(128), tenant: "t".repeat(64) }));

    const url = new URL(requestedUrl(fetch));
    expect(url.searchParams.get("model")).toBe("m".repeat(128));
    expect(url.searchParams.get("tenant")).toBe("t".repeat(64));
  });

  it("the filter readers are exported", () => {
    // Catches a missing or renamed lib: the pages and the address checks share these two readers.
    expect(typeof readRequestFilters).toBe("function");
    expect(typeof readCostFilters).toBe("function");
  });

  it("request list shows status and latency", async () => {
    // Catches a list without the two columns, a failed request shown as served, a latency printed bare or without the thousands comma, and a missing latency printed as 0 or blank instead of n/a.
    const body = fixture("requests");
    const base = (body.requests as Row[])[0];
    body.requests = [
      { ...base, tenant: "rowfailed", status: 502, latency_ms: 412 },
      { ...base, tenant: "rowoldest", status: 200, latency_ms: null },
      { ...base, tenant: "rowslowest", status: 200, latency_ms: 2500 },
    ];
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({})));

    const table = tableWithHeader(html, "Status");
    const head = headers(table);
    expect(head).toContain("Latency");
    const status = head.indexOf("Status");
    const latency = head.indexOf("Latency");
    const rows = bodyRows(table);
    const cells = (tenant: string): string[] => {
      const found = rows.find((row) => row.includes(tenant));
      expect(found, tenant).toBeDefined();
      return found ?? [];
    };
    expect([cells("rowfailed")[status], cells("rowfailed")[latency]]).toEqual(["502", "412 ms"]);
    expect([cells("rowoldest")[status], cells("rowoldest")[latency]]).toEqual(["200", "n/a"]);
    expect(cells("rowslowest")[latency]).toBe("2,500 ms");
  });

  it("slow line and Langfuse link", async () => {
    // Catches a slow line that is missing, in seconds, not rounded to whole milliseconds or without the thousands comma; a link that goes anywhere but the body's langfuse_url, loses rel, or is drawn when the body has none.
    const body = fixture("requests");
    body.slow_ms = 1904.6;
    body.langfuse_url = LANGFUSE;
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({})));

    expect(text(html)).toContain("Slow means above 1,905 ms.");
    const links = anchors(html).filter((anchor) => anchor.label === "Open these in Langfuse");
    expect(links).toHaveLength(1);
    expect(links[0].href).toBe(LANGFUSE);
    expect(links[0].tag).toMatch(/rel="noopener noreferrer"/);

    body.langfuse_url = null;
    body.slow_ms = null;
    stubFetch(body);
    const bare = renderToStaticMarkup(await RequestsPage(props({})));
    expect(anchors(bare).filter((anchor) => anchor.label === "Open these in Langfuse")).toEqual([]);
    expect(text(bare)).not.toContain("Open these in Langfuse");
    expect(text(bare)).not.toContain("Slow means above");
  });

  it("the pager keeps the filters and the form drops the page", async () => {
    // Catches a pager that forgets the filters (Next would clear them), a pager that loses its own page number, and a form that sends the page back (a new filter would open on a page that may not exist).
    const body = fixture("requests");
    body.requests_page = 1;
    body.requests_pages = 3;
    stubFetch(body);

    const html = renderToStaticMarkup(await RequestsPage(props({ tenant: "acme", errors: "true", requests_page: "1" })));

    const next = anchors(html).filter((anchor) => anchor.label === "Next");
    expect(next).toHaveLength(1);
    const query = new URLSearchParams(next[0].href.replace(/^\?/, ""));
    expect(query.get("tenant")).toBe("acme");
    expect(query.get("errors")).toBe("true");
    expect(query.get("requests_page")).toBe("2");
    expect(controlNames(formOf(html)).filter((name) => /page/i.test(name))).toEqual([]);
    expect(formOf(html)).not.toMatch(/name="[^"]*page/i);
  });

  it("the form has a Clear link back to the bare page", async () => {
    // Catches a Clear that keeps the filters in its address, or a form with no way back.
    stubFetch(fixture("requests"));

    const html = renderToStaticMarkup(await RequestsPage(props({ tenant: "acme", slow: "true" })));

    const clear = anchors(formOf(html)).filter((anchor) => anchor.label === "Clear");
    expect(clear).toHaveLength(1);
    expect(clear[0].href).toBe("/requests");
  });
});

describe("daily cost filters", () => {
  // 25 days times two tenants. On day d each row has d requests, 1 cached and costs d dollars.
  function costBody(): Fixture {
    const body = fixture("cost");
    body.days = ["acme-eu", "globex-eu"].flatMap((tenant) =>
      Array.from({ length: 25 }, (_, i) => ({
        day: `2026-08-${String(i + 1).padStart(2, "0")}`,
        tenant,
        provider: "mistral",
        requests: i + 1,
        cached_requests: 1,
        cost_usd: i + 1,
      })),
    );
    return body;
  }

  function dailyTable(html: string): string {
    return tableWithHeader(html, "Cached requests");
  }

  function totalRow(html: string): string[] {
    const rows = bodyRows(dailyTable(html)).filter((row) => row[0] === "Total");
    expect(rows).toHaveLength(1);
    return rows[0];
  }

  function dayRows(html: string): string[][] {
    return bodyRows(dailyTable(html)).filter((row) => row[0] !== "Total");
  }

  it("daily cost range and tenant narrow the table with a total row", async () => {
    // Catches a range that is ignored or off by one at either end, a tenant filter that is ignored, a total that sums only the page shown (20 rows of 40), and filters sent to console-api (which reads the 30 days itself).
    const fetch = stubFetch(costBody());

    const narrow = renderToStaticMarkup(
      await CostPage(props({ from: "2026-08-05", to: "2026-08-24", tenant: "acme-eu" })),
    );

    expect(requestedUrl(fetch)).toBe(`${API}/console/cost`);
    const rows = dayRows(narrow);
    expect(rows).toHaveLength(20);
    expect(new Set(rows.map((row) => row[1]))).toEqual(new Set(["acme-eu"]));
    expect(rows.map((row) => row[0])).toContain("2026-08-05");
    expect(rows.map((row) => row[0])).toContain("2026-08-24");
    expect(rows.map((row) => row[0])).not.toContain("2026-08-04");
    expect(rows.map((row) => row[0])).not.toContain("2026-08-25");
    // Days 5 to 24 sum to 290 requests; 20 rows with 1 cached each; $290.00.
    expect(totalRow(narrow)).toEqual(expect.arrayContaining(["Total", "290", "20", "$290.00"]));

    const both = stubFetch(costBody());
    const wide = renderToStaticMarkup(await CostPage(props({ from: "2026-08-05", to: "2026-08-24" })));
    expect(requestedUrl(both)).toBe(`${API}/console/cost`);
    expect(dayRows(wide)).toHaveLength(20);
    expect(text(wide)).toContain("Page 1 of 2");
    // 40 rows across two pages, 20 shown: the total still covers all 40.
    expect(totalRow(wide)).toEqual(expect.arrayContaining(["Total", "580", "40", "$580.00"]));
  });

  it("the daily cost form shows its range and tenant and keeps them", async () => {
    // Catches a form without From, To or Tenant, options not taken from the days in the body, values lost from the address, and a form that carries the page.
    stubFetch(costBody());

    const html = renderToStaticMarkup(
      await CostPage(props({ from: "2026-08-05", to: "2026-08-24", tenant: "globex-eu", days_page: "2" })),
    );

    const form = formOf(html);
    expect(labels(form)).toEqual(expect.arrayContaining(["From", "To", "Tenant"]));
    expect(controlNames(form).sort()).toEqual(["from", "tenant", "to"]);
    expect(inputTag(form, "from")).toContain('value="2026-08-05"');
    expect(inputTag(form, "to")).toContain('value="2026-08-24"');
    expect(optionValues(form, "tenant")).toEqual(expect.arrayContaining(["acme-eu", "globex-eu"]));
    expect(selectedOption(form, "tenant")).toBe("globex-eu");
    expect(form).toContain('method="get"');
    expect(anchors(form).filter((anchor) => anchor.label === "Clear").map((anchor) => anchor.href)).toEqual([
      "/cost",
    ]);
  });

  it("a reversed range is swapped and a bad date ignored", async () => {
    // Catches a from after to that shows an empty table, a bad date that blanks the page or is read as a real day, and a bad tenant that filters everything out. The reversed range gives the same 20 rows as the right way round, and the form shows the earlier day as From.
    stubFetch(costBody());
    const right = renderToStaticMarkup(await CostPage(props({ from: "2026-08-05", to: "2026-08-24" })));
    stubFetch(costBody());

    const swapped = renderToStaticMarkup(await CostPage(props({ from: "2026-08-24", to: "2026-08-05" })));

    expect(dayRows(swapped)).toEqual(dayRows(right));
    expect(totalRow(swapped)).toEqual(totalRow(right));
    expect(inputTag(formOf(swapped), "from")).toContain('value="2026-08-05"');
    expect(inputTag(formOf(swapped), "to")).toContain('value="2026-08-24"');

    stubFetch(costBody());
    const bad = renderToStaticMarkup(
      await CostPage(props({ from: "2026-13-01", to: "2026-08-10", tenant: "Bad Tenant" })),
    );

    // The bad From is ignored (no lower end), the good To holds, the bad tenant filters nothing.
    const days = dayRows(bad).map((row) => row[0]);
    expect(days.length).toBeGreaterThan(0);
    expect(days.every((day) => day <= "2026-08-10")).toBe(true);
    expect(new Set(dayRows(bad).map((row) => row[1])).size).toBeGreaterThan(0);
    expect(totalRow(bad)).toEqual(expect.arrayContaining(["Total", "110"]));
    expect(inputTag(formOf(bad), "from")).not.toContain("2026-13-01");
    expect(text(bad)).not.toContain("Console data unavailable");
  });

  it("a range with no rows shows none in this window and a zero total", async () => {
    // Catches a page that draws a Total row of NaN, or crashes, when the filters match nothing.
    stubFetch(costBody());

    const html = renderToStaticMarkup(await CostPage(props({ from: "2026-09-01", to: "2026-09-10" })));

    expect(text(html)).toContain("None in this window.");
    expect(text(html)).not.toContain("NaN");
  });
});
