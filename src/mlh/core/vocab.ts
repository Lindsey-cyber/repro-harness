// The finding vocabulary shared by the harness, the LLM components and the agent baseline. Pure.
export const VOCABULARY = `Finding categories (use exactly these names):
- data.train_test_overlap: the same rows or entities appear in train and test
- data.target_leak: a feature or preprocessing statistic carries label information not available at prediction time
- pipeline.eval_path_mismatch: train-time and eval-time preprocessing/feature paths differ
- eval.metric_mismatch: the reported metric is not what the model's predictions actually score
- ckpt.roundtrip_mismatch: the saved/loaded model does not reproduce the reported metric
- repro.nondeterministic: the same command and seed gives different results
- repro.claim_not_met: the README's claimed number is not reproduced by running the code
- other:<short-slug>: any other defect that changes the results; use it only when none of the categories above fits (e.g. a metric computed wrongly is eval.metric_mismatch, not other)`;
