import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({ test: { env: { MLH_BACKOFF_MS: "0" }, exclude: [...configDefaults.exclude] } });
