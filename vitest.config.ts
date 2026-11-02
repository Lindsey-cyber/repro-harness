import { configDefaults, defineConfig } from "vitest/config";

// .claude/skills holds pstack's own bun tests; they are not this project's suite.
export default defineConfig({ test: { exclude: [...configDefaults.exclude, ".claude/**"] } });
