/**
 * The plain names the pages print for score keys, and the one sentence that
 * explains the check score wherever it shows. Pinned by tests/clarity.test.tsx.
 */

export const CHECK_SCORE_SENTENCE =
  "Check score: the share of help desk conversations where every turn passed all three checks (the right tools in order, no change before the user says yes, no made-up name).";

const SCORE_LABELS: Record<string, string> = {
  check_score: "Check score",
  judge_score: "Judge score",
};

/** A score key in words; an unknown key loses its underscores rather than showing raw. */
export function scoreLabel(key: string): string {
  const known = SCORE_LABELS[key];
  if (known !== undefined) return known;
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Days console-api covers when the console sends no `days`; pinned to its `DEFAULT_DAYS` by tests/console-clarity.test.tsx. */
export const WINDOW_DAYS = 30;

export const WINDOW_SENTENCE = `Last ${WINDOW_DAYS} days.`;
