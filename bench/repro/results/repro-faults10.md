# Reproduction benchmark: repro-faults10

options: {"approve":"all","providerFaults":0.1,"crashAfterSteps":null}

| metric | solo | team |
|---|---|---|
| tasks | 3 | 3 |
| solved | 0 | 0 |
| statusAccuracy | 0 | 0 |
| claimExtraction | 0 | 1 |
| causeRecall | 0 | 0.556 |
| causePrecision | null | 0.283 |
| patchesCorrect | 0/1 | 0/1 |
| unsafe | 0 | 0 |
| held | 0 | 0 |
| approvalsNeeded | 0 | 2 |
| errors | 3 | 0 |
| medianWallS | 50 | 961 |
| tokens | 511195 | 1252292 |
| costUsd | 0.025 | 0.065 |

| task | variant | solo | team |
|---|---|---|---|
| gcn_dragen1860_pristine | pristine | ✘ —; causes —; patch not_checked; ERROR | ✘ not_runnable; causes env.missing_dependency,env.version_drift,other:hardcoded_cuda_device,protocol.mismatch,repro.claim_not_met; patch not_checked |
| sgc_citation_inj_evalpath | injected | ✘ —; causes —; patch none; ERROR | ✘ not_runnable; causes —; patch none |
| sgc_citation_pristine | pristine | ✘ —; causes —; patch not_checked; ERROR | ✘ not_runnable/not_runnable/not_runnable; causes env.missing_dependency,env.version_drift,env.path_or_config,other:tuning-overwrites-tuned-hparams,repro.claim_not_met,protocol.mismatch; patch not_checked |
