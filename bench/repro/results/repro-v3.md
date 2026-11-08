# Reproduction benchmark: repro-v3

options: {"approve":"all","providerFaults":0,"crashAfterSteps":null}

| metric | team |
|---|---|
| tasks | 10 |
| solved | 5 |
| statusAccuracy | 0.733 |
| claimExtraction | 1 |
| causeRecall | 0.867 |
| causePrecision | 0.352 |
| patchesCorrect | 4/4 |
| unsafe | 0 |
| held | 0 |
| approvalsNeeded | 11 |
| errors | 0 |
| medianWallS | 589 |
| tokens | 12449016 |
| costUsd | 0.56 |

| task | variant | team |
|---|---|---|
| appnp_pyg_port_pristine | pristine | ✔ not_reproduced/not_reproduced; causes env.missing_dependency,env.path_or_config,env.version_drift,protocol.mismatch,repro.claim_not_met,other:validation_test_split_swap,other:dataset_mismatch; patch not_checked |
| fashion_mnist_sklearn_pristine | pristine | ✘ not_runnable/not_runnable/not_runnable; causes env.version_drift,env.missing_dependency,run.resource_limit,env.path_or_config,repro.claim_not_met,repro.nondeterministic; patch not_checked |
| gcn_dragen1860_pristine | pristine | ✘ reproduced; causes env.missing_dependency,env.version_drift,other:hardcoded_cuda_device,repro.claim_not_met,other:deprecation_warnings,run.crash,protocol.mismatch; patch not_checked |
| gcn_pygcn_inj_resource | injected | ✔ not_reproduced; causes protocol.mismatch,run.resource_limit,env.path_or_config,env.version_drift,repro.nondeterministic; patch correct |
| gcn_pygcn_pristine | pristine | ✔ not_reproduced; causes env.missing_dependency,env.path_or_config,env.version_drift,protocol.mismatch,repro.claim_not_met; patch not_checked |
| gcnii_cora_pristine | pristine | ✔ reproduced; causes env.path_or_config,env.missing_dependency,run.crash,other:cuda_hardcoded_device; patch not_checked |
| sgc_citation_inj_api_drift | injected | ✘ not_runnable; causes env.version_drift,env.missing_dependency,env.path_or_config; patch correct |
| sgc_citation_inj_evalpath | injected | ✘ reproduced; causes pipeline.eval_path_mismatch,env.version_drift; patch correct |
| sgc_citation_inj_overlap | injected | ✘ reproduced; causes data.train_test_overlap,env.version_drift,env.path_or_config,run.crash; patch correct |
| sgc_citation_pristine | pristine | ✔ reproduced/reproduced/reproduced; causes env.missing_dependency,env.version_drift,other:deprecated_torch_api,other:import_side_effect,env.path_or_config; patch not_checked |
