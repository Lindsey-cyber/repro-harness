# Reproduction report

- objective: Reproduce the GLUE dev results of LoRA on RoBERTa-base for MRPC and RTE (Table 2) by fine-tuning with this repository's NLU example.
- source: /tmp/claude-0/complex/LoRA

## Claims

| claim | paper | status | mean | runs | source version |
|---|---|---|---|---|---|
| roberta_base_lora_mrpc_accuracy (GLUE MRPC dev accuracy) | 0.897 ± 0.007 | **not_runnable** | — |  | v0 |
| roberta_base_lora_rte_accuracy (GLUE RTE dev accuracy) | 0.866 ± 0.007 | **not_runnable** | — |  | v0 |

## Root causes

- `env.missing_dependency` [reported by agent] (repro.env): examples/NLU/examples/text-classification/run_glue.py imports `from datasets import load_dataset, load_metric` and `sklearn` (via the GLUE metric script). Base image only had torch 2.5.1, numpy 2.1.2,
- `env.version_drift` [reported by agent] (repro.env): Vendored transformers is 4.4.2 (examples/NLU/src/transformers, entry `transformers.__version__ == 4.4.2`) and run_glue.py calls `check_min_version("4.4.0")`. It requires the removed API `datasets.load
- `env.version_drift` [reported by agent] (repro.env): examples/NLU/setup.py declares `tokenizers>=0.10.1,<0.11` and src/transformers/dependency_versions_check.py hard-fails at import if an out-of-range tokenizers is installed; no tokenizers<0.11 wheel ex
- `env.path_or_config` [reported by agent] (repro.env): examples/NLU/roberta_base_mrpc.sh and roberta_base_rte.sh pass `--lora_path ./roberta_base_mnli_lora.bin` but the shipped file is `roberta_base_lora_mnli.bin` (ls: roberta_base_lora_mnli.bin 3568073 b
- `run.resource_limit` [reported by agent] (repro.env): Every shipped script hardcodes 8 GPUs (`export num_gpus=8` + `python -m torch.distributed.launch --nproc_per_node=$num_gpus`) while the sandbox is CPU-only / 1 GPU max; scripts as written cannot be la
- `env.missing_dependency` [reported by agent] (repro.env): Shell scripts also assume a distributed launch module `torch.distributed.launch` under torch 2.5.1 (deprecated; removed in newer torch) — needs verification/replacement by `torchrun` or a plain single
- `env.path_or_config` [reported by agent] (repro.env): Network reachability was confirmed at probe time (urllib GET https://huggingface.co/roberta-base/resolve/main/config.json -> 200), so `roberta-base` weights and the glue/mrpc dataset+metric can be fet

## Changes

None.

## Data provenance

No data fetched by the harness.

## Plan
```json
[]
```
