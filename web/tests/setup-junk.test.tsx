import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

describe("junk check on every render", () => {
  it("any render with junk throws", () => {
    // Catches a setup file that is not loaded, or that wraps renderToStaticMarkup without running the junk check, so a page test could render NaN and pass.
    expect(() => renderToStaticMarkup(<p>{"NaN"}</p>)).toThrow("NaN");
  });

  it("a clean render still returns its markup", () => {
    // Catches a wrapper that throws on every render, or that checks but drops the markup it was meant to hand back.
    expect(renderToStaticMarkup(<p>{"Answered 3"}</p>)).toBe("<p>Answered 3</p>");
  });
});
