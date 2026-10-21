import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, describe, expect, it } from "vitest";

import { TRADEOFF_METRICS, readTradeoff } from "@/lib/tradeoff";

const FIXTURE = fileURLToPath(new URL("./fixtures/tradeoff.json", import.meta.url));
const SCRATCH = mkdtempSync(join(tmpdir(), "console-tradeoff-"));

afterAll(() => {
  rmSync(SCRATCH, { recursive: true, force: true });
});

describe("readTradeoff", () => {
  it("rows follow order with the two scores in order", async () => {
    // Catches rows in the options object's key order, metrics in the file's key order, a missing metric other than null, or a dropped conclusion.
    expect(TRADEOFF_METRICS).toEqual(["check_score", "judge_score"]);

    const table = await readTradeoff(FIXTURE);
    expect(table).not.toBeNull();
    const rows = (table?.rows ?? []).map(({ id, label, kind, metrics, usd_per_1000_turns }) => ({
      id,
      label,
      kind,
      metrics,
      usd_per_1000_turns,
    }));
    expect(rows).toEqual([
      {
        id: "hosted-small",
        label: "Hosted small model",
        kind: "hosted",
        metrics: [0.93, 0.95],
        usd_per_1000_turns: "1.90",
      },
      {
        id: "hosted-large",
        label: "Hosted large model",
        kind: "hosted",
        metrics: [0.94, 0.97],
        usd_per_1000_turns: null,
      },
      {
        id: "local-8b",
        label: "Local 8B on one GPU",
        kind: "self_hosted",
        metrics: [0.86, null],
        usd_per_1000_turns: "0.67",
      },
    ]);
    expect(table?.conclusion).toBe("Hosted small model matches the large model within tolerance at a fraction of the cost.");
  });

  it("an unset path, a missing file, invalid json and an unknown order id are not measured", async () => {
    // Catches a reader that throws on any of these, or renders a table with a row for an id the options do not hold.
    const invalid = join(SCRATCH, "invalid.json");
    writeFileSync(invalid, "{not json", "utf8");
    const unknownId = join(SCRATCH, "unknown-id.json");
    const document = JSON.parse(readFileSync(FIXTURE, "utf8")) as { order: string[] };
    document.order = [...document.order, "ghost-option"];
    writeFileSync(unknownId, JSON.stringify(document), "utf8");

    const cases: Array<[string, string | undefined]> = [
      ["unset", undefined],
      ["empty", ""],
      ["missing", join(SCRATCH, "absent.json")],
      ["invalid json", invalid],
      ["unknown order id", unknownId],
    ];
    for (const [label, path] of cases) {
      expect(await readTradeoff(path), label).toBeNull();
    }
    expect(await readTradeoff(FIXTURE)).not.toBeNull();
  });
});
