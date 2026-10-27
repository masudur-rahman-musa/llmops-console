import { describe, expect, it } from "vitest";

import { assertNoJunk, findJunk } from "./junk";

/** A table body row, the way the shared Table draws one. */
function row(...cells: string[]): string {
  return `<table><tbody><tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr></tbody></table>`;
}

const SHA_40 = "5e2d8c1b9a7f6e5d4c3b2a19f8e7d6c5b4a39281";

describe("findJunk", () => {
  it("six decimals pass, seven fail", () => {
    // Catches a decimal check with the limit off by one either way: 1.123456 (six places) must pass, 13.0128054 (seven) must be named whole.
    expect(findJunk("<p>1.123456</p>")).toEqual([]);
    expect(findJunk("<p>13.0128054</p>")).toEqual(["long decimal: 13.0128054"]);
    expect(findJunk("<p>Failed 13.012805401607991 on Oct 26</p>")).toEqual(["long decimal: 13.012805401607991"]);
  });

  it("a hash of 6 passes, 7 and 40 fail", () => {
    // Catches a hash check with the wrong length range (6 named, or 7 or 40 let through), or one that names a hex-looking word with no digit.
    expect(findJunk("<p>release 1f6210</p>")).toEqual([]);
    expect(findJunk("<p>release 1f62101</p>")).toEqual(["bare hash: 1f62101"]);
    expect(findJunk(`<p>release ${SHA_40}</p>`)).toEqual([`bare hash: ${SHA_40}`]);
    expect(findJunk("<p>defaced</p>")).toEqual([]);
  });

  it("41 hex characters is past the hash range", () => {
    // Catches a hash check with no upper end, so the 7 to 40 range the plan names is not what the code does.
    expect(findJunk(`<p>${SHA_40}a</p>`).filter((item) => item.startsWith("bare hash"))).toEqual([]);
  });

  it("a hash with a trailing newline, CRLF or space is still named", () => {
    // Catches a hash check anchored to the whole cell text, so a sha followed by a line break or a space slips past.
    for (const tail of ["\n", "\r\n", " "]) {
      expect(findJunk(`<p>1f62101${tail}</p>`), JSON.stringify(tail)).toEqual(["bare hash: 1f62101"]);
    }
  });

  it("a hash inside code passes", () => {
    // Catches a hash check that also names the short ids the console shows on purpose in code (prompt version, blocked release).
    expect(findJunk("<p><code>1f62101</code></p>")).toEqual([]);
    expect(findJunk(`<li><code>${SHA_40.slice(0, 12)}</code> PR</li>`)).toEqual([]);
  });

  it("undefined, NaN, null and [object Object] fail", () => {
    // Catches a word check missing one of the four, or one matching inside a longer word so nullable or undefinedness would be named.
    for (const word of ["undefined", "NaN", "null", "[object Object]"]) {
      expect(findJunk(`<p>Score: ${word}</p>`), word).toEqual([`word: ${word}`]);
    }
    expect(findJunk("<p>nullable</p>")).toEqual([]);
  });

  it("a row whose only count is 0 fails", () => {
    // Catches a zero-row check that names any row holding a 0 (an empty day with its $0.00 cost is a real row) or one that names a row whose count is not 0.
    expect(findJunk(row("Internal", "0"))).toEqual(["zero-only row: Internal | 0"]);
    expect(findJunk(row("Oct 26", "0", "$0.00"))).toEqual([]);
    expect(findJunk(row("Internal", "3"))).toEqual([]);
  });

  it("only the text a reader sees is checked", () => {
    // Catches a check that reads attributes, scripts or styles, so every chart's raw coordinates and the page's own script would fail it.
    expect(findJunk('<svg><circle cx="173.33333333333334" cy="20" r="3"></circle></svg>')).toEqual([]);
    expect(findJunk('<a href="https://cloud.langfuse.com/traces/0af7651916cd43dd8448eb211c80319c">Open trace</a>')).toEqual([]);
    expect(findJunk("<script>self.__next_f.push([1,null,\"$undefined\"])</script><p>ok</p>")).toEqual([]);
    expect(findJunk("<style>.a{width:33.33333333%}</style><p>ok</p>")).toEqual([]);
  });

  it("every piece of junk on a page is named", () => {
    // Catches a check that stops at the first find, so a page with two faults reports only one.
    expect(findJunk("<p>13.0128054</p><p>NaN</p>").sort()).toEqual(["long decimal: 13.0128054", "word: NaN"]);
  });
});

describe("assertNoJunk", () => {
  it("clean text does not throw", () => {
    // Catches an assert that throws on every page, which would make every render fail whatever it shows.
    expect(() => assertNoJunk("<p>Answered 3 of 10 · 48.48%</p>")).not.toThrow();
  });

  it("junk throws naming it", () => {
    // Catches an assert that swallows what findJunk names, or throws without saying which text was bad.
    expect(() => assertNoJunk("<p>1.0833333333333335</p>")).toThrow("long decimal: 1.0833333333333335");
  });
});
