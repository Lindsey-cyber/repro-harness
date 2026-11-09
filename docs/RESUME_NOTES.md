# Verifiable Multi-Agent Harness for Autonomous ML Reproduction — project record

Source material for a CV / PhD-application entry. Every number below was measured in this
project (journals, traces and test runs are in the repository, branch `mlh`). Claims that are
**not** yet established are marked as such in section 9; keep them out of the CV or phrase them
as ongoing.

## 1. One-paragraph summary

Built, from scratch, a stateful multi-agent system that takes an ML paper (arXiv PDF) and its
code repository and autonomously reproduces the paper's reported numbers on cloud GPUs (Modal),
under a fixed per-paper budget, with independent verification of every reported result. A team
of LLM agents (claim extraction, environment setup, experiment planning, debugging) works inside
a harness that owns all facts: it extracts claims only if the number appears in the PDF, runs
every experiment itself in isolated containers, reads results with value-blind rules, compares
them to the paper with a statistical tolerance, and records every action in an event-sourced
journal from which any run can be resumed after a crash. Reliability was treated as the primary
objective: every failure observed in real runs became a deterministic regression test, tests were
mutation-checked, and controlled failure injection, preflight probes and strict stop rules were
added before further paid runs. Four published papers across four domains were reproduced
end to end (object detection, sentence embeddings, graph learning, tabular deep learning), all
within tolerance of the published values, at about $0.3–1.2 of GPU per paper.

## 2. Results (measured)

| Paper | Domain / type | Reproduced numbers (ours vs paper) | Modal cost | Notes |
|---|---|---|---|---|
| DETR (Carion et al., 2020) | detection, eval of released checkpoint | COCO val2017 box AP **0.420 vs 0.420** | ~$0.5 | the run's first recorded verdict was wrong (metric-reading bug); the output contains 0.420 exactly; the bug was found, fixed and regression-tested (§6, #13) |
| SimCSE (Gao et al., 2021) | NLP, eval of released checkpoint | avg STS **76.25 vs 76.25** (all 7 tasks printed: 68.40/82.41/74.38/80.91/78.56/76.85/72.23) | $0.27 | stage 107 s |
| GraphSAINT (Zeng et al., 2020) | graph learning, **training** of 8 configurations (4 samplers × PPI/Flickr) | F1-micro, 8/8 within tolerance: PPI node 0.9634 vs 0.960, edge 0.9809 vs 0.981, rw 0.9816 vs 0.981, mrw 0.9795 vs 0.980; Flickr node 0.5084 vs 0.507, edge 0.5067 vs 0.510, rw 0.5132 vs 0.511, mrw 0.5092 vs 0.510 | ~$0.9 | required compiling Cython extensions, a compiler toolchain, recovery from an out-of-memory kill; 7 verdicts recorded by the system, the 8th (PPI-edge) measured and within tolerance |
| FT-Transformer (Gorishniy et al., 2021) | tabular DL, **training**, 5 seeds × 2 datasets | California Housing RMSE **0.4574 ± 0.0030 vs 0.459**; Adult accuracy **0.8574 ± 0.0015 vs 0.854** | $1.21 total (incl. measured 26 % waste) | data from the paper's official host, checksummed |

Tolerance rule: |mean − paper| ≤ max(0.01, 2·paper std, 2·standard error of our runs).

Other measured results along the way:
- Early synthetic benchmark (M1, 12 tasks with injected bugs): harness 12/12.
- 10 real small reproduction tasks (bench/repro): single agent 6/10, multi-agent team 6/10 — equal
  accuracy; under injected transient faults the team finished 3/3 runs, the single agent 0/3
  (checkpointed, resumable nodes vs one long agent session).
- First campaign on complex papers (c1): only 1 of 4 finished papers gave a definitive verdict;
  analysis showed ~88 % of Modal spend (~$3.9) and ~45 % of LLM spend produced no result, and that
  the failures were platform/harness faults, not agent reasoning. This triggered the reliability
  programme (§5).
- Live canary on the real platform (synthetic repo with a known metric, full pipeline, real T4):
  5/5 exact, including a run killed mid-way and resumed; ~$0.001 per run.
- Offline suite: 22 test files, 177 tests; mutation check: 17/17 deliberately reintroduced bugs
  were caught (plus later fixes mutation-checked individually).
- LLM cost is small: c1 used ~$0.36 of DeepSeek for 4 papers; FT-Transformer $0.14.

## 3. System architecture

**Orchestration (stateful, fault-tolerant).** A task DAG whose state is a fold over an
append-only, fsync'd event journal (`node`, `start`, `ok`, `fail`, `hold`, `approve`, `skip`,
`reopen`). Any run resumes from its journal after a crash; agent loops checkpoint every step
(`state.json`). Failures carry one of 10 categories; only transient ones are retried, with
exponential backoff. Nodes are added as results arrive (paper → claims → environment → plan →
fetch → stages → comparisons → debugger → patch → re-run).

**Agents (specialised, bounded).** One LLM loop implementation with per-role tools, step budgets,
per-command time caps and wall-clock budgets:
- *claims*: extracts the objective's claims from the PDF text; a number is kept only if it
  appears in the PDF.
- *environment*: makes the repository runnable in a remote container: picks a base image (the
  harness suggests the one matching the repo's torch pin), pip packages, system layers, launch
  command; declares data it needs through a harness tool.
- *planner*: writes a staged GPU plan (stages, dependencies, per-seed instances, GPUs, timeouts,
  data downloads with checksums, links, memory, the regex that reads each claim's number); it
  can call the harness's own validator (`check_plan`, budget of 8) before reporting.
- *debugger*: diagnoses a failed or mismatching stage and proposes a code change or a changed
  command; it may not edit measurement code without approval.

**Execution (sandboxed).** Untrusted repository code never runs on the host: locally in Linux
user/net/mount namespaces with a read-only root and resource limits; remotely in Modal sandboxes
(one per command) built from a base image + pip packages + declared system layers, with network
blocked or restricted to approved data hosts, a per-run volume at `/cache` (no run can see
another's files), stage outputs persisted and restored across stages, and a per-paper budget that
reserves each run's worst case before admission and charges actual cost per node attempt.

**Verification (independent of agent claims).** The harness, not an agent, (1) checks claims
against the PDF, (2) smoke-tests the environment itself, (3) validates plans (scripts must exist,
no inline programs, no placeholders except `{seed}`, every stage must serve a claim, data must
come from declared downloads, cost within budget, download URLs probed for reachability),
(4) runs every stage, (5) reads each number with the claim's regex — never "the last match":
lines that differ only in measured values are one series (final value counts), differently
labelled lines are ambiguous and resolved by a model that sees the labels with every value
masked (it cannot steer toward the paper's number), else no verdict — and (6) compares with the
tolerance rule.

**Risk-adaptive authorisation.** Every action passes an authorisation function that weighs
action risk against evidence: sandboxed execution needs a passing isolation self-test;
code patches are graded (medium: static checks must pass; high — touching evaluation, metrics,
splits or seeds — needs a human or a standing policy); downloads from unknown hosts are held
for a human; publishing is denied. Held items go to a review list; runs never block on a human.

**Trust per agent role.** Each role earns trust from harness facts only (claims in the PDF; smoke
test passed without repair; plan accepted; an accepted change made the failing step succeed).
Failures are attributed to one of five classes (infrastructure, resource, agent, paper,
unclassified) and only agent failures count. Promotion requires a 95 % Wilson lower bound ≥ 0.6
(in practice ≥ 6 clean runs); one failure while trusted demotes and resets the count. While any
role is unproven, campaigns run at most 3 papers, strictly one at a time. Trust is recomputed
from journals, so crashes or concurrent rounds cannot corrupt it.

**Operations and observability.** Campaign driver (pinned clones, child processes, watchdog that
kills a run without progress and resumes it), round chaining with done/stopped markers,
automatic per-round retrospective (metric vs previous round, top failure and owner, stop
predicate), a platform probe (`doctor`) that must pass before any round, per-attempt cost
accounting (productive / retried / failed by class / killed / unattributed), and strict stop
rules (same infrastructure fault twice, an unclassifiable failure, >30 % wasted spend, LLM cap).

Scale: ~6.2k lines of TypeScript/Python, 200 commits, 22 test files / 177 tests.

## 4. Evaluation framework

- **Benchmarks**: synthetic tasks with injected bugs (M1); held-out blind tasks; 10 real small
  reproduction tasks; complex-paper campaigns.
- **Baselines**: deterministic pipeline (no LLM), single agent with all tools and the same budget
  (its reported numbers re-verified by re-running its commands), multi-agent team.
- **Reliability**: controlled failure injection end to end with a fake cloud backend and a
  scripted agent team (capacity shortage, spend limit, transport error, timeout, image-build
  failure, GPU OOM, malformed model output, process killed after each node and inside an agent,
  torn journal/budget files, unapproved data host, failed download); acceptance: each run ends
  in the clean result or a named stop, no paid stage re-runs after resume, no infrastructure
  failure counted against an agent.
- **Regression**: every failure observed in a real run became a deterministic test (recorded
  error corpus, replay of recorded planner drafts, recorded journals and logs as fixtures).
- **Mutation checking**: fixed bugs were reintroduced one at a time to prove the tests catch them.
- **Execution traces**: per node attempt: wall time, LLM tokens/cost, Modal cost, sandbox runs;
  per agent step: LLM time and per-tool time.

## 5. Chronology and turning points

1. **Harness core (M1–M2).** Event-sourced DAG, namespace sandbox, risk-based authorisation,
   verified skills; LLM components under verification; single-agent and deterministic baselines;
   blind held-out benchmark.
2. **From toy tasks to real papers.** Real repositories and real PDFs; claims must appear in the
   PDF; Modal GPU backend with budgets; env/planner/debugger roles; independent re-verification of
   the single agent's numbers.
3. **First complex campaign (c1) — the reality check.** 1/4 definitive verdicts. Cost analysis
   showed most spend wasted on platform faults: wrong image labels (3 of 6 base images had the
   wrong Python), a capacity shortage misclassified as fatal, resume that left platform-failed
   nodes failed, validator rules that rejected correct plans, agents running training on CPU.
4. **Reliability-first programme** (owner's directive: no concurrency, no performance work, no
   paid run as an infrastructure test). Plan P0–P6 with acceptance criteria: accounting and
   failure attribution; mock end-to-end pipeline; failure injection; scheduler tests; live
   canary; reliability baseline. All offline phases and the live canary passed before resuming.
5. **Paper-by-paper hardening.** Each paper exposed a class of problem that was fixed
   structurally and tested (§6), after which the next paper ran on the improved system.
6. **Four papers reproduced** (§2), strictly sequentially, with a 5-minute review cadence of live
   runs.

## 6. Difficulties found in real runs and how each was fixed (selection, in order)

Each item: what happened → structural fix → test.

1. Modal capacity shortage read as "out of money" → error classifier with a recorded corpus;
   only spend limits stop. 2. Retries waited 50 ms → minutes of backoff. 3. Base-image labels
   named the wrong Python → `doctor` asks each image what it contains; a round needs a fresh pass.
4. Resume left platform-failed nodes failed → platform failures reopen on resume (tested on a
   real journal). 5. Validator rejected a correct model-directory download → replay of recorded
   planner drafts as tests. 6. Agents timed training on CPU and backgrounded jobs to escape time
   limits → per-role time caps, no background processes. 7. Env agent ignored the repo's torch pin
   → harness maps the pin to the matching image. 8. An optimisation (local "read-only" commands)
   gave agents two filesystem views → removed after measuring it saved only 2.4 s per call.
9. Runs read other runs' leftovers in a shared volume → per-run volumes. 10. https-only rule made
   a planner write an unusable COCO URL → http allowed with a checksum; URLs probed at planning.
11. Debugger credited for other steps / demotion undone by one success / gate deadlock (found by
   an independent code review) → fixed, tested. 12. A debugger run for an OOM kill re-ran all 9
   stages for a one-stage change → re-run scope follows what changed. 13. **A confident wrong
   verdict**: DETR's regex matched both AP (0.420) and AR@100 (0.574) and the harness took the
   last → no "last match"; value-masked label disambiguation; tested on the real COCO output.
14. A planner encoded expected numbers in its regex → rejected. 15. Build outputs not visible to
   the next stage (volume commit race) → runner waits for the commit; missing declared outputs
   fail by name; verified live 5/5. 16. 28 of 30 planned stages served no claim; a stage read the
   whole volume → every stage must serve a claim; no whole-volume references. 17. Env agent
   downloaded data outside the harness, so a cached environment broke in a fresh volume → data is
   declared through a harness tool, checksummed, carried with the environment cache.
18. `{seed}` placeholder passed literally → substituted per instance, other placeholders rejected.
19. Per-epoch log lines taken as different metrics → numbers outside `[...]` treated as
   measurements. 20. A stopped paper could not be resumed → no result row for strict stops.
21. Build toolchains, host memory and data links had no declared form → system image layers,
   `memGB`, `links` (verified live).

## 7. What made it work (keys to success)

- **Facts come from the harness, never from agents**: claims checked against the PDF, results
  read by the harness from its own runs, value-blind disambiguation.
- **Every incident becomes structure**: a deterministic regression test (or validator rule)
  before the next paid run; tests proven by mutation.
- **Prove the platform before spending on it**: preflight probe, live canary, offline failure
  injection with a fake backend.
- **Make failures attributable and costs visible**: five-class attribution, per-attempt cost
  buckets, strict stop rules instead of silent retries.
- **Isolation and provenance**: per-run volumes, declared and checksummed downloads, recorded
  environments, a journal that explains every number.
- **Measure before optimising**: an optimisation was reverted when measurement showed no benefit.
- **Short review cycles on live runs**: a 5-minute status cadence caught several problems within
  minutes (e.g. an agent looking for its repository for 12 steps; 28 useless planned stages).

## 8. Draft CV bullets (to be polished)

- Built a stateful multi-agent harness (claim extraction, environment, planning, debugging agents)
  that autonomously reproduces ML papers from PDF + code on cloud GPUs; event-sourced journal with
  crash-safe resume, dependency-aware staged plans, sandboxed execution, per-paper budgets.
- Reproduced 4 published papers end to end across 4 domains (DETR AP 42.0; SimCSE 76.25 STS;
  GraphSAINT 8/8 F1 within tolerance; FT-Transformer RMSE 0.457 / acc 0.857 over 5 seeds), all
  within tolerance of the published numbers at ≈$0.3–1.2 GPU per paper.
- Designed independent verification and risk-adaptive authorisation: claims must appear in the
  paper, results are re-measured by the harness with value-blind metric extraction, code changes
  are risk-graded (evaluation/metric edits need human approval), and per-role agent trust is
  earned from audited outcomes (Wilson lower bound) before the system scales.
- Built an evaluation and reliability framework: failure injection across the full pipeline,
  regression tests from every real incident (177 tests, 17/17 mutation catch rate), live canary,
  per-attempt cost/waste accounting and strict stop rules; single-agent and deterministic
  baselines (equal accuracy on 10 real tasks; 3/3 vs 0/3 completion under injected faults).

## 9. Not yet established (do not claim as done)

- Repeatability of a complex paper over several identical runs (the planned 5-run baseline was
  stopped in favour of covering more papers).
- Single-agent baseline on the complex papers (only on the 10 small tasks).
- Trust promotion: no role has reached the promotion standard yet (too few audited runs).
- LoRA and SimSiam (running at the time of writing).
- GraphSAINT PPI-edge's verdict was measured but not recorded by the system's comparison step;
  DETR's system verdict was recorded before the extraction fix (the correct value is in the log).

## 10. If LoRA and SimSiam are reproduced: what to add

- **6 papers across 6 domains**: detection, sentence embeddings, graph learning, tabular DL,
  parameter-efficient fine-tuning of language models (LoRA on RoBERTa-base, GLUE MRPC/RTE),
  self-supervised representation learning (SimSiam on CIFAR-10).
- **Claim types**: beyond matching a number, a *qualitative ablation claim* (SimSiam: with
  stop-gradient kNN accuracy is high, without it the representation collapses) — verification of
  a scientific conclusion, not just a metric.
- **Harder engineering**: LoRA needs a patched, vendored `transformers` and older library pins;
  SimSiam needs a training run from scratch (scaled epochs) — covering eval-only, fine-tuning,
  multi-config training and from-scratch training.
- **Aggregate metrics to report**: papers reproduced / attempted, claims reproduced / total,
  median wall time and GPU cost per paper, waste share after the reliability programme vs c1
  (≈88 % of Modal spend wasted in c1).
- Optionally, if run: single-agent baseline on the same complex papers; repeated runs of one paper
  (variance of the verdict and of cost); first agent roles promoted under the trust standard.
