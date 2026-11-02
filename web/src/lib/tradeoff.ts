import { readFile } from "node:fs/promises";

/**
 * The model tradeoff table, read from `CONSOLE_TRADEOFF_PATH` on every request.
 * Anything short of a whole, consistent document is `null`, which
 * the cost page shows as `Not measured yet.`
 */

export const TRADEOFF_METRICS = [
  "outcome_every_play",
  "safety",
] as const;

export type TradeoffRow = {
  id: string;
  label: string;
  kind: string;
  metrics: (number | null)[];
  usd_per_1000_turns: string | null;
};

export type Tradeoff = { rows: TradeoffRow[]; conclusion: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function row(option: unknown): TradeoffRow | null {
  if (!isRecord(option) || typeof option.id !== "string" || typeof option.label !== "string" || typeof option.kind !== "string") {
    return null;
  }
  const metrics = isRecord(option.metrics) ? option.metrics : {};
  const cost = option.usd_per_1000_turns;
  return {
    id: option.id,
    label: option.label,
    kind: option.kind,
    metrics: TRADEOFF_METRICS.map((name) => {
      const value = metrics[name];
      return typeof value === "number" && Number.isFinite(value) ? value : null;
    }),
    usd_per_1000_turns: typeof cost === "string" ? cost : null,
  };
}

export async function readTradeoff(path: string | undefined): Promise<Tradeoff | null> {
  if (!path) {
    return null;
  }
  let document: unknown;
  try {
    document = JSON.parse(await readFile(path, "utf8"));
  } catch {
    return null;
  }
  if (
    !isRecord(document) ||
    !isRecord(document.options) ||
    !Array.isArray(document.order) ||
    typeof document.conclusion !== "string"
  ) {
    return null;
  }
  const options = document.options;
  const rows: TradeoffRow[] = [];
  for (const id of document.order) {
    const found = typeof id === "string" && Object.hasOwn(options, id) ? row(options[id]) : null;
    if (found === null) {
      return null;
    }
    rows.push(found);
  }
  return { rows, conclusion: document.conclusion };
}
