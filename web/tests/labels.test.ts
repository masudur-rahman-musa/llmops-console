import { describe, expect, it } from "vitest";

import { WINDOW_LINES, WINDOW_SENTENCE, plainName, redactionLabel } from "@/lib/labels";

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
});
