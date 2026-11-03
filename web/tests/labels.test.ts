import { describe, expect, it } from "vitest";

import * as labels from "@/lib/labels";
import { OUTCOME_SENTENCE, WINDOW_LINES, WINDOW_SENTENCE, plainName, redactionLabel, scoreLabel } from "@/lib/labels";

describe("plain labels", () => {
  it("redaction kinds read in plain words in either case", () => {
    // Catches a label table keyed on one case only (the gateway sends lowercase, the help desk sends upper), and a kind left raw.
    const rows: Array<[string, string]> = [
      ["EMAIL", "Emails"],
      ["email", "Emails"],
      ["CARD", "Card numbers"],
      ["card", "Card numbers"],
      ["PHONE", "Phone numbers"],
      ["phone", "Phone numbers"],
      ["IBAN", "Bank accounts"],
      ["iban", "Bank accounts"],
    ];
    for (const [kind, label] of rows) {
      expect(redactionLabel(kind), kind).toBe(label);
    }
  });

  it("an unknown redaction kind loses its underscores and gets a capital", () => {
    // Catches an unknown kind shown raw (kind_0), in capitals, or with the underscore turned into nothing (Kind0).
    expect(redactionLabel("kind_0")).toBe("Kind 0");
    expect(redactionLabel("KIND_0")).toBe("Kind 0");
    expect(redactionLabel("passport_number")).toBe("Passport number");
  });

  it("plainName turns snake case into a sentence start", () => {
    // Catches names left with underscores, every word capitalised, or a single word left in lower case.
    expect(plainName("wrong_tool")).toBe("Wrong tool");
    expect(plainName("out_of_scope")).toBe("Out of scope");
    expect(plainName("missed_refusal")).toBe("Missed refusal");
    expect(plainName("refusal")).toBe("Refusal");
  });

  it("each tab has its window line", () => {
    // Catches a window line reworded or moved to another tab; the strings are pinned word for word, and Requests and Quality keep the shared sentence.
    expect(WINDOW_LINES.home).toBe("Today so far (UTC). p95 latency and error rate cover the last 24 hours.");
    expect(WINDOW_LINES.cost).toBe(
      "Last 30 days. Budgets and this month's spend cover the calendar month so far (UTC).",
    );
    expect(WINDOW_LINES.requests).toBe("Last 30 days. Newest requests lists every request, newest first.");
    expect(WINDOW_LINES.quality).toBe(WINDOW_SENTENCE);
    expect(WINDOW_SENTENCE).toBe("Last 30 days.");
  });

  it("each score key reads in the words the pages print", () => {
    // Catches a score key shown raw or with its old name: the four gate and quality scores, the effort means, and the four result counts.
    const rows: Array<[string, string]> = [
      ["safety", "Safety"],
      ["outcome_every_play", "Outcome every play"],
      ["outcome_at_least_once", "Outcome at least once"],
      ["reply_quality", "Reply quality"],
      ["mean_turns", "Mean turns"],
      ["mean_tool_calls", "Mean tool calls"],
      ["mean_seconds", "Mean seconds"],
      ["mean_cost_usd", "Mean cost"],
      ["pass", "Pass"],
      ["fail", "Fail"],
      ["needs_a_look", "Needs a look"],
      ["setup_wrong", "Setup wrong"],
    ];
    for (const [key, label] of rows) {
      expect(scoreLabel(key), key).toBe(label);
    }
  });

  it("one outcome sentence replaces the check score sentence", () => {
    // Catches the old sentence kept beside the new one, or a new one that does not explain the score it sits under: it is one sentence, names the score, and mentions no check score.
    expect("CHECK_SCORE_SENTENCE" in labels).toBe(false);
    expect(OUTCOME_SENTENCE).toContain("Outcome every play");
    expect(OUTCOME_SENTENCE.endsWith(".")).toBe(true);
    expect(OUTCOME_SENTENCE.slice(0, -1)).not.toContain(". ");
    expect(OUTCOME_SENTENCE).not.toMatch(/check score|judge score/i);
  });

  it("eval_bug reads Eval bug", () => {
    // Catches a result key shown raw (eval_bug) or with a stray capital (Eval Bug): the fifth result reads in the words the Quality page prints, next to its four neighbours.
    expect(scoreLabel("eval_bug")).toBe("Eval bug");
    expect(scoreLabel("setup_wrong")).toBe("Setup wrong");
    expect(scoreLabel("needs_a_look")).toBe("Needs a look");
  });
});
