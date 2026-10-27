import { expect, it } from "vitest";

import { checkConsole } from "./check";

/** Needs the running console (`make up`); started by `make console-check`, never by the normal run. */
it("every console tab is free of junk text", async () => {
  const secret = process.env.CONSOLE_SESSION_SECRET ?? "";
  const baseUrl = process.env.CONSOLE_CHECK_URL ?? "http://localhost:3001";
  const failures = await checkConsole({ baseUrl, secret });
  expect(failures, failures.join("\n")).toEqual([]);
});
