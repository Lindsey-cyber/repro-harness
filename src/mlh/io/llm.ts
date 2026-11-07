// OpenAI-compatible chat client (DeepSeek by default) with token/cost accounting
// and a record/replay cache: a benchmark rerun with the same prompts costs
// nothing and returns the same answers, and the cached entry keeps the original
// usage so cost is still reported as what a cold run would pay.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface ToolCall { id: string; type: "function"; function: { name: string; arguments: string } }
export type ChatMsg =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[]; reasoning_content?: string }
  | { role: "tool"; tool_call_id: string; content: string };
export interface ToolDef { type: "function"; function: { name: string; description: string; parameters: Record<string, unknown> } }

export interface Usage { calls: number; promptTokens: number; completionTokens: number; cachedPromptTokens: number; costUsd: number; ms: number; replayed: number }
export const zeroUsage = (): Usage => ({ calls: 0, promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, costUsd: 0, ms: 0, replayed: 0 });
export const addUsage = (a: Usage, b: Usage): Usage => ({ calls: a.calls + b.calls, promptTokens: a.promptTokens + b.promptTokens, completionTokens: a.completionTokens + b.completionTokens, cachedPromptTokens: a.cachedPromptTokens + b.cachedPromptTokens, costUsd: a.costUsd + b.costUsd, ms: a.ms + b.ms, replayed: a.replayed + b.replayed });

// USD per 1M tokens: [input cache miss, input cache hit, output]. List prices as
// published by third-party trackers in 2026-11 (the official page did not show
// rates); override with MLH_PRICE_<MODEL>="in,hit,out". Costs are estimates.
const PRICES: Record<string, [number, number, number]> = {
  "deepseek-flash": [0.14, 0.028, 0.28],
  "deepseek-v4-pro": [1.74, 0.145, 3.48],
};
function price(model: string): [number, number, number] {
  const env = process.env[`MLH_PRICE_${model.toUpperCase().replace(/[^A-Z0-9]/g, "_")}`];
  if (env) { const p = env.split(",").map(Number); if (p.length === 3 && p.every(Number.isFinite)) return p as [number, number, number]; }
  return PRICES[model] ?? [0, 0, 0];
}

export interface LlmOptions { model: string; apiKey?: string; baseUrl?: string; cacheDir?: string; timeoutMs?: number }

export class Llm {
  readonly usage: Usage = zeroUsage();
  readonly model: string;
  private readonly key: string;
  private readonly base: string;
  private readonly cacheDir: string | undefined;
  private readonly timeoutMs: number;

  constructor(o: LlmOptions) {
    this.model = o.model;
    this.key = o.apiKey ?? process.env.DEEPSEEK_API_KEY ?? "";
    this.base = o.baseUrl ?? process.env.MLH_LLM_BASE_URL ?? "https://api.deepseek.com";
    this.cacheDir = o.cacheDir;
    this.timeoutMs = o.timeoutMs ?? 240_000;
  }

  static available(): boolean {
    return !!process.env.DEEPSEEK_API_KEY;
  }

  async chat(messages: ChatMsg[], opts: { tools?: ToolDef[]; maxTokens?: number; json?: boolean; effort?: "low" | "high" | "max"; salt?: string } = {}): Promise<{ message: Extract<ChatMsg, { role: "assistant" }>; usage: Usage }> {
    const body: Record<string, unknown> = { model: this.model, messages, temperature: 0, max_tokens: opts.maxTokens ?? 4000 };
    if (opts.tools?.length) body.tools = opts.tools;
    if (opts.json) body.response_format = { type: "json_object" };
    if (opts.effort) body.reasoning_effort = opts.effort;
    // The salt changes only the cache key, so a deliberate retry is not answered by the cached failure.
    // MLH_LLM_SALT separates benchmark repeats so each repeat samples the model afresh.
    const key = createHash("sha256").update(JSON.stringify(body) + (opts.salt ?? "") + (process.env.MLH_LLM_SALT ?? "")).digest("hex");
    const cached = this.cacheDir ? join(this.cacheDir, `${key}.json`) : undefined;
    let data: { choices: { message: Extract<ChatMsg, { role: "assistant" }> }[]; usage: Record<string, unknown> }, ms: number, replayed = 0;
    if (cached && existsSync(cached)) {
      ({ data, ms } = JSON.parse(readFileSync(cached, "utf8")));
      replayed = 1;
    } else {
      if (!this.key) throw new Error("no DEEPSEEK_API_KEY");
      const t0 = performance.now();
      let lastErr = "";
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          const r = await fetch(`${this.base}/chat/completions`, { method: "POST", headers: { Authorization: `Bearer ${this.key}`, "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(this.timeoutMs) });
          const text = await r.text();
          if (r.status === 429 || r.status >= 500) { lastErr = `${r.status} ${text.slice(0, 200)}`; await new Promise((s) => setTimeout(s, 2000 * 2 ** attempt)); continue; }
          if (!r.ok) throw new Error(`LLM ${r.status}: ${text.slice(0, 500)}`);
          data = JSON.parse(text);
          lastErr = "";
          break;
        } catch (e) {
          if (String(e).includes("LLM 4")) throw e;
          lastErr = String(e);
          await new Promise((s) => setTimeout(s, 2000 * 2 ** attempt));
        }
      }
      if (lastErr) throw new Error(`LLM unavailable after retries: ${lastErr}`);
      ms = performance.now() - t0;
      if (cached) { mkdirSync(this.cacheDir!, { recursive: true }); writeFileSync(cached, JSON.stringify({ data: data!, ms })); }
    }
    const u = data!.usage;
    const prompt = Number(u.prompt_tokens ?? 0), completion = Number(u.completion_tokens ?? 0), hit = Number(u.prompt_cache_hit_tokens ?? 0);
    const [pin, phit, pout] = price(this.model);
    const usage: Usage = { calls: 1, promptTokens: prompt, completionTokens: completion, cachedPromptTokens: hit, costUsd: ((prompt - hit) * pin + hit * phit + completion * pout) / 1e6, ms: ms!, replayed };
    Object.assign(this.usage, addUsage(this.usage, usage));
    return { message: data!.choices[0]!.message, usage };
  }

  // One JSON object from the model; tolerant of fences around it.
  json<T>(system: string, user: string, maxTokens = 4000): Promise<T> {
    return jsonVia(this, system, user, maxTokens);
  }
}

// Reasoning models can spend the whole budget thinking and return nothing; the
// one retry uses low effort and a fresh cache key.
export async function jsonVia<T>(m: Pick<Llm, "chat">, system: string, user: string, maxTokens: number): Promise<T> {
  let last = "";
  for (const retry of [false, true]) {
    const { message } = await m.chat([{ role: "system", content: system }, { role: "user", content: user }], { maxTokens, json: true, ...(retry ? { effort: "low" as const, salt: "retry-low-effort" } : {}) });
    last = (message.content ?? "").trim().replace(/^```(json)?\s*|\s*```$/g, "");
    try { return JSON.parse(last) as T; } catch { /* retry once */ }
  }
  throw new Error(`model returned no JSON object: ${last.slice(0, 200)}`);
}

// A per-node view of a model that counts only that node's usage.
export function meter(m: Pick<Llm, "model" | "chat">): Pick<Llm, "model" | "usage" | "chat" | "json"> {
  const usage = zeroUsage();
  const view = {
    model: m.model,
    usage,
    async chat(...a: Parameters<Llm["chat"]>) {
      const r = await m.chat(...a);
      Object.assign(usage, addUsage(usage, r.usage));
      return r;
    },
    json<T>(system: string, user: string, maxTokens = 4000): Promise<T> { return jsonVia<T>(view, system, user, maxTokens); },
  };
  return view;
}
