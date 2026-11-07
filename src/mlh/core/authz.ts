// Risk-based authorization. Pure. Separates "may this action happen" from the
// code that performs it. A model's opinion is recorded but never authorizes
// anything above low risk: medium needs a deterministic check, high a human.

export type Action =
  | "read_workspace" // read files of the copied repository
  | "fetch_source" // clone the repository from its URL
  | "exec_sandboxed" // run repository code inside the isolated sandbox
  | "exec_unsandboxed" // run repository code on the host
  | "network_egress" // repository code reaching the network
  | "write_outside_workspace"
  | "publish"; // push, open a PR, post anywhere

export type Risk = "low" | "medium" | "high";

export const RISK: Record<Action, Risk> = {
  read_workspace: "low",
  fetch_source: "medium",
  exec_sandboxed: "medium",
  exec_unsandboxed: "high",
  network_egress: "high",
  write_outside_workspace: "high",
  publish: "high",
};

export interface Evidence {
  kind: "deterministic" | "llm" | "human";
  check: string; // what was checked, e.g. "sandbox.selftest" or "approval:node-id"
  ok: boolean;
  action?: Action; // a human approval names the action it approves
}

export interface AuthzPolicy {
  // Actions refused outright, whatever the evidence.
  deny: Action[];
}

export const DEFAULT_POLICY: AuthzPolicy = { deny: ["publish", "write_outside_workspace"] };

export type Decision = { allow: true; basis: string } | { allow: false; hold: boolean; reason: string };

export function authorize(action: Action, evidence: Evidence[], policy: AuthzPolicy = DEFAULT_POLICY): Decision {
  if (policy.deny.includes(action)) return { allow: false, hold: false, reason: `${action} is denied by policy` };
  const risk = RISK[action];
  if (risk === "low") return { allow: true, basis: "low risk" };
  const human = evidence.find((e) => e.kind === "human" && e.ok && (e.action === undefined || e.action === action));
  if (risk === "high") {
    if (human) return { allow: true, basis: `human approval (${human.check})` };
    const llm = evidence.some((e) => e.kind === "llm" && e.ok);
    return { allow: false, hold: true, reason: `${action} is high risk and needs a human approval${llm ? "; a model's approval does not count" : ""}` };
  }
  const failed = evidence.find((e) => e.kind === "deterministic" && !e.ok);
  if (failed) return human ? { allow: true, basis: `human approval despite failed check ${failed.check}` } : { allow: false, hold: true, reason: `deterministic check ${failed.check} failed` };
  const det = evidence.find((e) => e.kind === "deterministic" && e.ok);
  if (det) return { allow: true, basis: `deterministic check ${det.check}` };
  if (human) return { allow: true, basis: `human approval (${human.check})` };
  return { allow: false, hold: true, reason: `${action} is medium risk and no deterministic check passed` };
}
