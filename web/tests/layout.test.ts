import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// web/tests -> web -> repo root.
const REPO_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const WEB_ROOT = fileURLToPath(new URL("..", import.meta.url));

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

describe("console layout", () => {
  it("services/console is gone and web/src holds the console", () => {
    // Catches a half-done move: the old tree left behind, or the new one missing.
    expect(existsSync(join(REPO_ROOT, "services", "console"))).toBe(false);
    expect(existsSync(join(WEB_ROOT, "src", "lib", "console-api.ts"))).toBe(true);
    expect(existsSync(join(WEB_ROOT, "src", "app", "(console)", "page.tsx"))).toBe(true);
  });

  it("no file imports @/lib/guards", () => {
    // Catches a leftover import of the module that was split into features/<page>/guards.ts and lib/shapes.ts.
    const files = [...filesUnder(join(WEB_ROOT, "src")), ...filesUnder(join(WEB_ROOT, "tests"))].filter((file) =>
      /\.(ts|tsx)$/.test(file),
    );
    const offenders = files.filter((file) => /from "@\/lib\/guards"/.test(readFileSync(file, "utf8")));
    expect(offenders).toEqual([]);
  });
});
