# Campaign summary (2026-11-09T04:22:30.122Z)

papers 1; claims 1: reproduced 0, not reproduced 0, not runnable 1; GPU $0.04; LLM $0.04 (940680 tokens); resumes 0

| paper | claims (mean vs paper) | fidelity | env smoke | GPU $ | LLM tokens | min | resumes | failures |
|---|---|---|---|---|---|---|---|---|
| detr (2005.12872) | ✘ detr_r50_coco_val2017_box_ap: — vs 0.42 | eval_only | pass | 0.039 | 940680 | 10 | 0 | experiment_error×1 |

- detr: repro.fetch: fetch failed: "items", [])]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/fetch_data.py", line 69, 

Pipeline success (every claim definitive, not stopped): 0/1; median 10 min/paper

scale gated: roles on probation (claims, env, planner) -> at most 3 papers, 1 at a time

| role | level | runs | ok | success-rate lower bound | last outcome |
|---|---|---|---|---|---|
| claims | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: every extracted number is in the PDF |
| env | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: smoke test passed on the agent's own report |
| planner | probation | 1 | 1 | 0.21 | ok p5detr-clean1/detr: plan accepted |
| debugger | probation | 0 | 0 | 0.00 | — |

Standard for promotion: ≥ 5 audited runs and a 95% Wilson lower bound ≥ 0.6. One failure while trusted demotes and resets the count. The debugger blocks scale only once it has been needed.