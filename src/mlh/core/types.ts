// Shared vocabulary of the ML harness. Pure.

export type FailureCategory =
  | "sandbox_transient" // the sandbox failed to start or died for reasons outside the experiment
  | "timeout"
  | "provider_transient" // the model provider failed or returned an unusable answer
  | "provider_fatal" // out of balance or bad credentials: retrying cannot help, the run stops
  | "resource_limit" // cpu, memory or file-size limit hit
  | "experiment_error" // the repository's own code exited non-zero
  | "contract_violation" // a skill's output did not match its declared schema
  | "precondition" // a required input (manifest field, artifact) is missing
  | "policy_denied" // authorization refused the action
  | "harness_bug";

// Only failures that a fresh attempt can plausibly fix are retried.
export const RETRYABLE: readonly FailureCategory[] = ["sandbox_transient", "timeout", "provider_transient"];
// Failures of the platform, not of the work: a resumed run gives such nodes a fresh attempt.
export const INFRA_FAILURES: readonly FailureCategory[] = ["sandbox_transient", "provider_transient", "provider_fatal"];

export interface Manifest {
  train?: string[];
  eval?: string[];
  data?: { train: string; test: string; label?: string };
  outputs?: { metrics?: string; predictions?: string; checkpoint?: string; evalMetrics?: string };
  metric?: string;
  // Keys in the metrics files when they are not the metric's own name.
  metricKey?: string;
  trainMetricKey?: string;
  claim?: { metric: string; value: number };
  seedEnv?: string;
  // CLI flag the training script reads its seed from (e.g. --seed).
  seedArg?: string;
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

// confirmed: an independent deterministic measurement supports it.
// supported: only a reproducible, harness-judged effect measurement written by a model supports it.
export type FindingStatus = "confirmed" | "supported" | "suspected" | "refuted" | "disputed";

export interface Confirmation {
  node: string;
  supports: boolean;
  note: string;
  // deterministic: measured by harness code. measured: an effect the harness judged from a model-written measurement.
  tier?: "deterministic" | "measured";
  // The check could not decide either way (e.g. a perfect-separability test on a leak that is not a single feature).
  inconclusive?: boolean;
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
