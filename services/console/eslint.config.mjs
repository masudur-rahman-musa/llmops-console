// eslint-config-next 15 ships eslintrc-style configs only. Its plugins are
// resolved from its own install location rather than listed as direct
// dependencies, so the flat config below needs nothing the lockfile does not
// already carry.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const fromNextConfig = createRequire(require.resolve("eslint-config-next"));
const nextPlugin = fromNextConfig("@next/eslint-plugin-next");
const tsParser = fromNextConfig("@typescript-eslint/parser");
const tsPlugin = fromNextConfig("@typescript-eslint/eslint-plugin");
const reactHooks = fromNextConfig("eslint-plugin-react-hooks");

export default [
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
  ...tsPlugin.configs["flat/recommended"],
  {
    files: ["**/*.ts", "**/*.tsx"],
    languageOptions: {
      parser: tsParser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { "@next/next": nextPlugin, "react-hooks": reactHooks },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,
      ...reactHooks.configs.recommended.rules,
      // An app-router project has no pages directory for this rule to read.
      "@next/next/no-html-link-for-pages": "off",
    },
  },
];
