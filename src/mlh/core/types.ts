// Shared vocabulary of the ML harness. Pure.

export type FailureCategory =
  | "sandbox_transient" // the sandbox failed to start or died for reasons outside the experiment
  | "timeout"
  | "resource_limit" // cpu, memory or file-size limit hit
  | "experiment_error" // the repository's own code exited non-zero
  | "contract_violation" // a skill's output did not match its declared schema
  | "precondition" // a required input (manifest field, artifact) is missing
  | "policy_denied" // authorization refused the action
  | "harness_bug";

// Only failures that a fresh attempt can plausibly fix are retried.
export const RETRYABLE: readonly FailureCategory[] = ["sandbox_transient", "timeout"];

export interface Manifest {
  train?: string[];
  eval?: string[];
  data?: { train: string; test: string; label?: string };
  outputs?: { metrics?: string; predictions?: string; checkpoint?: string; evalMetrics?: string };
  metric?: string;
  claim?: { metric: string; value: number };
  seedEnv?: string;
}

export type Severity = "high" | "medium" | "low";

export interface FindingDraft {
  category: string;
  severity: Severity;
  summary: string;
  evidence: Record<string, unknown>;
  // "fact": an exact measurement (e.g. a row count) that needs no second opinion.
  // "suspicion": must be confirmed by independent evidence before it is reported as confirmed.
  kind: "fact" | "suspicion";
}

export type FindingStatus = "confirmed" | "suspected" | "refuted" | "disputed";

export interface Confirmation {
  node: string;
  supports: boolean;
  note: string;
}

export interface Finding extends FindingDraft {
  id: string;
  node: string;
  status: FindingStatus;
  confirmations: Confirmation[];
}

export interface ArtifactRef {
  name: string;
  sha256: string;
  bytes: number;
}
