import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      exclude: [
        "src/main.ts",
        "src/index.ts",
        "src/types.ts",
        "**/*.test.ts",
        "vitest.config.ts",
      ],
      reporter: ["text", "json-summary"],
      thresholds: {
        statements: 83,
        branches: 66,
        functions: 98,
        lines: 85,
      },
    },
  },
});
