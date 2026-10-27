import { vi } from "vitest";

import { assertNoJunk } from "./junk";

/**
 * Every `renderToStaticMarkup` in every test runs the junk check on what it renders,
 * so a new page or chart is covered without anyone remembering. Pinned by
 * tests/setup-junk.test.tsx.
 */
vi.mock("react-dom/server", async (importOriginal) => {
  const real = await importOriginal<typeof import("react-dom/server")>();
  const checked: typeof real.renderToStaticMarkup = (...args) => {
    const markup = real.renderToStaticMarkup(...args);
    assertNoJunk(markup);
    return markup;
  };
  return { ...real, default: { ...real, renderToStaticMarkup: checked }, renderToStaticMarkup: checked };
});
