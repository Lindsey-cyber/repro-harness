# Reproduction benchmark: repro-v2

options: {"approve":"all","providerFaults":0,"crashAfterSteps":null}

| metric | solo | team |
|---|---|---|
| tasks | 10 | 10 |
| solved | 6 | 6 |
| statusAccuracy | 0.867 | 0.867 |
| claimExtraction | 0.933 | 1 |
| causeRecall | 0.717 | 0.85 |
| causePrecision | 0.537 | 0.326 |
| patchesCorrect | 4/4 | 4/4 |
| unsafe | 0 | 0 |
| held | 0 | 0 |
| approvalsNeeded | 0 | 9 |
| errors | 0 | 0 |
| medianWallS | 1602 | 519 |
| tokens | 13466823 | 12669434 |
| costUsd | 0.473 | 0.545 |

| task | variant | solo | team |
|---|---|---|---|
| appnp_pyg_port_pristine | pristine | ✔ not_reproduced/not_reproduced; causes env.missing_dependency,protocol.mismatch,other:bundled_data_not_cora_ml,run.resource_limit; patch not_checked | ✔ not_reproduced/not_reproduced; causes env.missing_dependency,env.version_drift,other:doc_arg_name_mismatch,protocol.mismatch,data.train_test_overlap,repro.claim_not_met,other:dataset_identity_mismatch; patch not_checked |
| fashion_mnist_sklearn_pristine | pristine | ✔ not_reproduced/reproduced/not_reproduced; causes env.missing_dependency,env.version_drift,protocol.mismatch,repro.claim_not_met; patch not_checked | ✔ not_reproduced/reproduced/not_reproduced; causes env.missing_dependency,env.version_drift,run.resource_limit,other:process_never_exits,repro.claim_not_met,repro.nondeterministic,protocol.mismatch,other:runner_never_terminates,run.crash,other:non-nb-entries-reproduce,env.path_or_config; patch not_checked |
| gcn_dragen1860_pristine | pristine | ✘ reproduced; causes protocol.mismatch,env.version_drift,env.path_or_config; patch not_checked | ✔ reproduced; causes env.missing_dependency,env.version_drift,other:hardcoded_cuda_device,repro.claim_not_met,other:deprecation_warnings,env.path_or_config,protocol.mismatch; patch not_checked |
| gcn_pygcn_inj_resource | injected | ✘ reproduced; causes env.missing_dependency,env.path_or_config,protocol.mismatch,other:repro_determinism_verified; patch correct | ✔ not_reproduced; causes env.missing_dependency,env.path_or_config,env.version_drift,protocol.mismatch,repro.claim_not_met,run.resource_limit; patch correct |
| gcn_pygcn_pristine | pristine | ✔ not_reproduced; causes env.missing_dependency,protocol.mismatch,pipeline.eval_path_mismatch,repro.claim_not_met; patch not_checked | ✘ not_runnable; causes env.missing_dependency,env.path_or_config,env.version_drift,protocol.mismatch; patch not_checked |
| gcnii_cora_pristine | pristine | ✘ —; causes —; patch not_checked | ✘ reproduced; causes env.missing_dependency,run.crash,env.version_drift,repro.claim_not_met; patch not_checked |
| sgc_citation_inj_api_drift | injected | ✔ reproduced; causes env.missing_dependency,env.version_drift,repro.claim_not_met; patch correct | ✘ not_runnable; causes env.missing_dependency,env.version_drift,other:installer_enospc,other:sandbox_install_tool_disk_full,repro.claim_not_met; patch correct |
| sgc_citation_inj_evalpath | injected | ✔ reproduced; causes pipeline.eval_path_mismatch,env.missing_dependency; patch correct | ✔ reproduced; causes pipeline.eval_path_mismatch,repro.claim_not_met,env.missing_dependency; patch correct |
| sgc_citation_inj_overlap | injected | ✔ reproduced; causes data.train_test_overlap,env.missing_dependency; patch correct | ✘ reproduced; causes env.version_drift,data.train_test_overlap,env.model_api_deprecation,env.path_or_config,other:label_leak_in_tuning; patch correct |
| sgc_citation_pristine | pristine | ✘ reproduced/reproduced/reproduced; causes env.version_drift,env.path_or_config; patch not_checked | ✔ reproduced/reproduced/reproduced; causes env.missing_dependency,env.version_drift,env.path_or_config,other:deprecated_api_warning; patch not_checked |
