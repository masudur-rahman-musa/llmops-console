import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PlayPage from "@/app/(console)/quality/plays/[run]/[case]/[play]/page";

const API = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const RUN = "0123456789abcdef0123456789abcdef";

type Params = { run: string; case: string; play: string };

/** The play page the way Next calls it: the path parts arrive as a promise. */
const Page = PlayPage as (props: { params: Promise<Params> }) => Promise<ReactElement>;

const PLAY = {
  run_id: RUN,
  case_id: "case-012",
  play: 2,
  turns: [
    { message: "Hi, I need 2.5 refunds <b>now</b> & fast", reply: "Let me look at order 4411.", tools: ["lookup_order"] },
    { message: "Yes please", reply: "Done, $12.50 is back on your card.", tools: ["lookup_order", "approve_refund"] },
    { message: "Thanks", reply: "You are welcome.", tools: [] },
  ],
};

let fetchMock: ReturnType<typeof vi.fn>;

async function render(play: unknown): Promise<string> {
  const body = { sources: { postgres: "ok", prometheus: null }, play };
  fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  // renderToStaticMarkup runs the shared junk check on what it renders (tests/setup.ts).
  return renderToStaticMarkup(await Page({ params: Promise.resolve({ run: RUN, case: "case-012", play: "2" }) }));
}

function text(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

beforeEach(() => {
  vi.stubEnv("CONSOLE_API_URL", API);
  vi.stubEnv("CONSOLE_ADMIN_KEY", KEY);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("play page", () => {
  it("play page shows each turn", async () => {
    // Catches a page that drops a reply, shows the turns out of order, leaves out the tool names, reads the wrong play (the address is checked), or prints a message as markup instead of text.
    const html = await render(PLAY);
    const flat = text(html);

    const order = [
      "Hi, I need 2.5 refunds <b>now</b> & fast",
      "Let me look at order 4411.",
      "lookup_order",
      "Yes please",
      "Done, $12.50 is back on your card.",
      "approve_refund",
      "Thanks",
      "You are welcome.",
    ];
    let cursor = 0;
    for (const piece of order) {
      const at = flat.indexOf(piece, cursor);
      expect(at, piece).toBeGreaterThan(-1);
      cursor = at + piece.length;
    }
    // The message is data: it is escaped, never a tag.
    expect(html).not.toContain("<b>now</b>");
    expect(String(fetchMock.mock.calls[0][0])).toBe(`${API}/console/quality/plays/${RUN}/case-012/2`);
  });

  it("play page with no play", async () => {
    // Catches a page that crashes or prints an empty conversation when the play is not stored (`play: null`).
    const flat = text(await render(null));

    expect(flat).toContain("This play is not in the store.");
    expect(flat).not.toContain("Let me look");
  });
});
