import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

function readJson(name: string): unknown {
  return JSON.parse(readFileSync(join(ROOT, name), "utf8"));
}

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

describe("console config", () => {
  it("tsconfig is strict", () => {
    // Catches a tsconfig with strict off or absent.
    const tsconfig = readJson("tsconfig.json") as { compilerOptions?: { strict?: unknown } };
    expect(tsconfig.compilerOptions?.strict).toBe(true);
  });

  it("dev and start serve on 3001", () => {
    // Catches a dev or start script left on Next's default port 3000, or on any port other than 3001.
    const pkg = readJson("package.json") as { scripts?: Record<string, string> };
    for (const script of ["dev", "start"]) {
      expect(pkg.scripts?.[script], script).toMatch(/(^|\s)(--port|-p)(\s+|=)3001(\s|$)/);
    }
  });

  it("no source file names a NEXT_PUBLIC_ variable", () => {
    // Catches a source file exposing a variable to the browser bundle through Next's public prefix; this file sits outside src, so it cannot match itself.
    const files = filesUnder(join(ROOT, "src"));
    // An empty or missing src must fail, not pass vacuously.
    expect(files.length).toBeGreaterThan(0);
    const offenders = files.filter((file) => /NEXT_PUBLIC_/.test(readFileSync(file, "utf8")));
    expect(offenders).toEqual([]);
  });
});
