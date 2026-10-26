/**
 * The plain names the pages print for score keys, redaction kinds and code names,
 * the one sentence that explains the check score wherever it shows, and each tab's
 * window line. Pinned by tests/clarity.test.tsx and tests/labels.test.ts.
 */

export const CHECK_SCORE_SENTENCE =
  "Check score: the share of help desk conversations where every turn passed all three checks (the right tools in order, no change before the user says yes, no made-up name).";

const SCORE_LABELS: Record<string, string> = {
  check_score: "Check score",
  judge_score: "Judge score",
};

/** A code name as a sentence start: `wrong_tool` reads `Wrong tool`, `KIND_0` reads `Kind 0`. */
export function plainName(key: string): string {
  const words = key.toLowerCase().replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** A score key in words; an unknown key loses its underscores rather than showing raw. */
export function scoreLabel(key: string): string {
  return SCORE_LABELS[key] ?? plainName(key);
}

const REDACTION_LABELS: Record<string, string> = {
  email: "Emails",
  card: "Card numbers",
  phone: "Phone numbers",
  iban: "Bank accounts",
};

/** A redaction kind in words, in either case: the gateway sends `email`, the help desk `EMAIL`. */
export function redactionLabel(kind: string): string {
  return REDACTION_LABELS[kind.toLowerCase()] ?? plainName(kind);
}

/** Days console-api covers when the console sends no `days`; pinned to its `DEFAULT_DAYS` by tests/console-clarity.test.tsx. */
export const WINDOW_DAYS = 30;

export const WINDOW_SENTENCE = `Last ${WINDOW_DAYS} days.`;

/** The line under each tab's question saying what time it covers. */
export const WINDOW_LINES = {
  home: "Today so far (UTC). p95 latency and error rate cover the last 24 hours.",
  requests: `${WINDOW_SENTENCE} Newest requests lists every request, newest first.`,
  cost: `${WINDOW_SENTENCE} Budgets and this month's spend cover the calendar month so far (UTC).`,
  quality: WINDOW_SENTENCE,
} as const;
