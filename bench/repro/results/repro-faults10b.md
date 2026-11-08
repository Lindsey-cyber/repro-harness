# Reproduction benchmark: repro-faults10b

options: {"approve":"all","providerFaults":0.1,"crashAfterSteps":null}

| metric | solo | team |
|---|---|---|
| tasks | 3 | 3 |
| solved | 0 | 1 |
| statusAccuracy | 0 | 0.8 |
| claimExtraction | 0 | 1 |
| causeRecall | 0 | 0.667 |
| causePrecision | null | 0.548 |
| patchesCorrect | 0/1 | 0/1 |
| unsafe | 0 | 0 |
| held | 0 | 0 |
| approvalsNeeded | 0 | 1 |
| errors | 3 | 0 |
| medianWallS | 234 | 207 |
| tokens | 1119927 | 1156655 |
| costUsd | 0.044 | 0.051 |

| task | variant | solo | team |
|---|---|---|---|
| gcn_dragen1860_pristine | pristine | ✘ —; causes —; patch not_checked; ERROR | ✔ reproduced; causes env.missing_dependency,env.version_drift,env.path_or_config,protocol.mismatch,other:deprecated_torch_api_warnings,data.target_leak,repro.claim_not_met; patch not_checked |
| sgc_citation_inj_evalpath | injected | ✘ —; causes —; patch none; ERROR | ✘ not_runnable; causes env.missing_dependency,env.version_drift,pipeline.eval_path_mismatch; patch none |
| sgc_citation_pristine | pristine | ✘ —; causes —; patch not_checked; ERROR | ✘ reproduced/reproduced/reproduced; causes —; patch not_checked |
