/**
 * The shared junk check for the console's rendered text.
 *
 * It reads the text a reader sees, never attributes, scripts or styles, and names
 * each find: a number with more than 6 decimals, a bare 7 to 40 character commit
 * hash (anything in `<code>` is shown on purpose), the words `undefined`, `NaN`,
 * `null` and `[object Object]`, and a table row whose only count is 0. Every render
 * in the page and chart tests runs it through `tests/setup.ts`, and `make
 * console-check` runs it on the live pages. Pinned by tests/junk.test.ts.
 */

const LONG_DECIMAL = /(?<![\d.])\d+\.\d{7,}(?!\d)/g;
// `\w` as the edge keeps a prefixed id such as a user id (`u_` and 32 hex) from counting as a bare hash.
const HASH = /(?<!\w)[0-9a-f]{7,40}(?!\w)/gi;
const WORD = /(?<!\w)(?:undefined|NaN|null)(?!\w)|\[object Object\]/g;
const ROW = /<tr\b[^>]*>([\s\S]*?)<\/tr>/g;
const CELL = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/g;

function visible(markup: string): string {
  return markup.replace(/<(script|style)\b[\s\S]*?<\/\1>/g, " ");
}

function text(markup: string): string {
  return markup.replace(/<[^>]*>/g, " ");
}

function isHash(value: string): boolean {
  return /\d/.test(value) && /[a-f]/i.test(value);
}

/** Every piece of junk in the markup, one message each, in the order of the checks. */
export function findJunk(markup: string): string[] {
  const page = visible(markup);
  const body = text(page.replace(/<code\b[^>]*>[\s\S]*?<\/code>/g, " "));
  const found: string[] = [];
  for (const match of body.matchAll(LONG_DECIMAL)) found.push(`long decimal: ${match[0]}`);
  for (const match of body.matchAll(HASH)) if (isHash(match[0])) found.push(`bare hash: ${match[0]}`);
  for (const match of body.matchAll(WORD)) found.push(`word: ${match[0]}`);
  for (const section of page.match(/<tbody\b[\s\S]*?<\/tbody>/g) ?? []) {
    for (const row of section.matchAll(ROW)) {
      const cells = [...row[1].matchAll(CELL)].map((cell) => text(cell[1]).trim());
      const counts = cells.filter((cell) => /^\d+$/.test(cell));
      const others = cells.filter((cell) => !/^\d+$/.test(cell));
      if (counts.length === 1 && counts[0] === "0" && others.every((cell) => !/\d/.test(cell))) {
        found.push(`zero-only row: ${cells.join(" | ")}`);
      }
    }
  }
  return found;
}

/** Throws naming every find; returns quietly on clean markup. */
export function assertNoJunk(markup: string): void {
  const found = findJunk(markup);
  if (found.length > 0) throw new Error(`Junk in rendered text: ${found.join("; ")}`);
}
