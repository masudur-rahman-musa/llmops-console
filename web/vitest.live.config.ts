import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

/** Runs only `tests/live/`, against the running console. Started by `make console-check`. */
export default defineConfig({
  esbuild: { jsx: "automatic" },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["tests/live/**/*.test.ts"],
  },
});
