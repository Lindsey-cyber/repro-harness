# Reliability validation plan (complex-paper reproduction)

Owner's priority (2026-11-09): reliability, reproducibility and trust before speed. No added
concurrency, no performance work, no paid experiment used as an infrastructure test. Every run
is stopped until the gates below are met; paid phases need the owner's go.

## 1. Evidence today

### 1.1 Observed failures and their deterministic regression tests

All tests are offline ($0) and deterministic. Last run: 16 files, 115 tests, 115 passed.

| # | Observed failure (where) | Class | Test that reproduces it | Result |
|---|---|---|---|---|
| F1 | Capacity shortage read as fatal; lora stopped (c1) | infra | `mlh-errors` "only a spend limit stops the campaign; a capacity shortage is retried" + recorded corpus | pass |
| F2 | Spend limit read as transient; retries burned budget (c1) | infra | `mlh-errors` provider_fatal case (recorded message) | pass |
| F3 | Resume left platform-failed nodes failed; lora "finished" in 0 min (c1) | harness | `mlh-reopen` on lora's real journal | pass |
| F4 | Validator rejected a model dir filled by fetch items (simcse, c1) | harness | `mlh-replay` simcse drafts 1-2 | pass |
| F5 | Validator gave a misleading message for outputs outside /cache/runs (graphsaint) | harness | `mlh-replay` graphsaint draft 1 | pass |
| F6 | 3 of 6 image labels wrong | harness | `mlh-doctor` label check (unit) + live probe (`mlh doctor`, PASS 02:19Z) | pass |
| F7 | Env agent stayed on Python 3.11 despite torch==1.7.0 pin (simsiam) | agent, aided by harness | `mlh-doctor` imageForPins | pass |
| F8 | Agents ran long CPU training to time it (planner 163 min, simsiam env) | agent | `mlh-contracts` time limits (maxRunS, wallMin) | pass |
| F9 | Chain started the next round before the previous one started | harness | **none** (bash script untested) | GAP |
| F10 | Trust: debugger credited for other steps; demotion undone by one success; gate deadlock | harness | `mlh-trust` (6 tests) | pass |
| F11 | Infra failures counted against agents | harness | `mlh-trust` "infrastructure failures are not blamed on the agent" | pass |
| F12 | CPU containers billed $0 | harness | `mlh-trust` container cost | pass |
| F13 | Concurrent trust writers / torn file | harness | `mlh-retro` rebuild from journals | pass |

These tests prove each *observed* failure stays fixed. They do **not** prove the workflow is
stable: there is no end-to-end test of the GPU reproduction flow with a mocked platform, no
failure injection on that flow, and no repeated run of one paper. That is what section 3 adds.

### 1.2 Cost so far, by outcome (all runs, from traces and budgets)

| Node outcome | Nodes | LLM $ | Wall min |
|---|---|---|---|
| succeeded | 28 | 0.384 | 515.6 |
| failed: provider_fatal (spend limit / capacity) | 4 | 0.119 | 103.6 |
| failed: contract_violation (plan rejected) | 2 | 0.138 | 40.7 |
| failed: sandbox_transient | 15 | 0.113 | 31.9 |
| failed: experiment_error | 1 | 0.000 | 0.0 |
| **total** | 50 | 0.754 | 691.7 |

Modal: about $3.89 actually spent. Only DETR ($0.46) produced a definitive result, so about
**88 % of Modal spend and ~45 % of LLM spend produced no result**. "Succeeded" nodes also hide
waste: the 163-min planner, the discarded simsiam env session.

Accounting gaps found while writing this:
- GPU spend is recorded per run, not per node, so it cannot be attributed to failed attempts.
- Copying lora's run into c2 duplicated its budget file: the ledger would count $2.72 twice (sum
  shown by the ledger: $6.61).
- There is no category for "productive but wasted" (replanning, discarded sessions).

## 1.3 Evidence added by the validation phases (2026-11-09)

| Phase | Status | Evidence |
|---|---|---|
| P0 accounting & attribution | done | `tests/mlh-accounting` (class table, cost buckets, per-attempt Modal spend under concurrency, ledger dedupe, killed-attempt LLM spend, stop rules); `mlh cost <round>`; c1 re-accounted in `bench/complex/results/c1.cost.md` (old runs' Modal spend shown as unattributed, not guessed) |
| P1 mock end-to-end | done | `tests/mlh-e2e-repro`, `tests/mlh-faults` P1: 10 runs from scratch identical (claims, cost, one stage run); debugger diagnoses a failed command and the patch is authorized and re-run; data fetched with provenance; journal never has two nodes in flight |
| P2 failure injection | done | `tests/mlh-faults` P2: capacity once / twice (stop, resume), spend limit, transport error, stage timeout, image build failure, GPU OOM, malformed model answer, kill after each of 5 nodes and inside the env agent, torn journal and budget, unapproved data host held, failed download named |
| P3 scheduler & driver | done | `tests/mlh-chain` (6 chain cases, strictly sequential), `tests/mlh-campaign` (doctor gate, pinned clone, watchdog kill of a hung child and resume, result row, trust, retro, done marker). Found and fixed: a child that died before its journal was "resumed" (fails); the watchdog never killed a child hung before its first journal write |
| Mutation check | done | 17 of 17 reintroduced bugs are caught by the suite (`bench/complex/results/reliability/mutations-2026-11-09.txt`) |
| P4 live canary | **pass** | doctor PASS; canary reproduced exactly on real Modal (image, /cache, T4) in 4 of 4 runs, one killed after 4 nodes and resumed; about $0.001 per run |
| P5 attempt 1 | **failed**, stopped after run 1 | Run 1 (10 min, $0.08) ended without a verdict: the validator allowed https only, so the planner wrote `https://images.cocodataset.org/...`, a host whose certificate does not match (it serves http). Fixed: http allowed with a sha256 (verified by the fetch); every declared URL is probed during planning and an unreachable one is a plan rejection the planner sees; deterministic fetch failures are the planner's, transient ones are retried; the driver now stops at the first run without a definitive verdict |
| P5 attempt 2 | **stopped at 5 min** by the 5-minute review | The env agent spent steps 11-22 looking for the repository: "read-only" commands ran in the local sandbox and the rest remotely, two filesystem views (a speed optimization worth 2.4 s per call). It also read other runs' leftovers in the shared /cache volume. Fixed: every command of a remote agent runs remotely; each run gets its own /cache volume (`mlh-run-<runKey>`), so nothing one run leaves reaches another; the env cache is off unless configured; watchdog silence 30 min (was 2 h). Live canary re-passed with per-run volumes |
| P5 attempt 3 | running | DETR, 3 clean + 2 killed-and-resumed, sequential, reviewed every 5 min |

Offline suite: 21 files, 168 tests, all passing.

## 2. Failure taxonomy (to be enforced in code, P0)

Every failed attempt gets exactly one class, recorded in the journal:

| Class | Examples | Counts against agents? |
|---|---|---|
| infrastructure | Modal transport, capacity, image registry, harness bug | no |
| resource | spend limit, LLM balance, budget refusal, GPU OOM at planned size | no |
| agent | plan rejected, smoke failed on the agent's own report, change did not fix the step, report missing | yes |
| paper | claim not reproducible within tolerance after a correct run, missing data or weights upstream | no (it is a finding) |

Unclassifiable failures (`harness_bug` or an unknown message) stop the round.

## 3. Validation plan

Phases P0-P3 are offline ($0); P4 costs cents; P5-P6 are paid and need the owner's go.

| Phase | What | Acceptance criteria |
|---|---|---|
| **P0 Accounting & attribution** | Per-node Modal spend and LLM tokens/cost in the journal; taxonomy class per failure; ledger keyed by run id (no duplicates from copies); report splits productive vs failed-attempt vs infra vs replanning/retry cost | Unit tests on recorded journals; c1 re-accounted with every dollar assigned to a class |
| **P1 Mock end-to-end** | Full repro DAG (paper -> claims -> env -> plan -> fetch -> stage -> compare -> report) with a fake Modal backend (deterministic outputs) and a scripted LLM (recorded responses). Each stage also tested alone: planning, env setup, execution, validation, checkpointing, resume, result collection | Same final report on 10 repeated runs (byte-identical claims table); each stage's contract test passes |
| **P2 Failure injection** | Into the P1 flow: capacity, spend limit, transport error, timeout, image build error, OOM, LLM 5xx, malformed LLM output, process kill at every node boundary and mid-agent, torn journal/budget/state files, stall (watchdog) | For every injection: expected class recorded; run either completes with the **same** report as the clean run, or stops with a named reason; no paid stage re-executed after resume; no infra/resource failure in agent trust |
| **P3 Scheduler/chain** | Chain and campaign driver against fake child processes: not started yet, ended between polls, stopped marker, stale marker on relaunch, crash of the driver | Each case behaves as specified; sequential only (one paper at a time, asserted) |
| **P4 Live canary** (~$0.05) | `mlh doctor` plus one synthetic repository with a known deterministic metric run through the real stack on Modal CPU and T4 | Doctor PASS; canary metric exact; per-node cost recorded |
| **P5 Reliability baseline** (paid, ~$2-3) | One paper, DETR eval-only (cheapest, reproduced twice before): 3 clean runs from scratch with identical config, sequential; then 2 runs with controlled real failures (kill the driver mid-stage; kill it during the env agent) | 5/5 complete; same claim status every run; metric spread within tolerance; resume reuses finished nodes (no duplicate paid stage); 0 agent-attributed infra failures; waste <= 20 % of spend |
| **P6 Second paper** (paid) | SimCSE eval, same protocol | Same criteria; only then strictly sequential trust rounds resume |

## 4. Budgets and stop conditions (enforced in code, P0)

- The same infrastructure signature twice in one round stops the round for investigation.
- Any unclassified failure stops the round.
- Per paper: GPU budget as configured and an LLM cap (to be set from P5 data); a paper whose
  failed-attempt cost exceeds 30 % of its spend stops.
- No round without a doctor PASS from the last 24 h.
- One paper at a time; no concurrency anywhere in the pipeline until P6 passes.

## Handoff (2026-11-09, end of session)

State:
- Branch `mlh`, everything committed. Offline suite: 21 files, 153 tests passing. Mutation check:
  17/17 reintroduced bugs caught. Live canary on real Modal: 5/5 pass (one killed and resumed).
- P5 attempt 3 (DETR, sequential: 3 clean + 2 killed-and-resumed) was started at 04:33Z by
  `scripts/mlh-reliability.sh` and left running unattended. It stops itself at the first run that
  is stopped or has no definitive verdict, on a repeated infrastructure fault, waste > 30 %, or the
  LLM cap. At handoff run 1 was in the env step (13 steps, $0.002 Modal), behaving normally.
- Log: `bench/complex/results/reliability/p5c.log`; per run: `mlh campaign-status p5c-clean1`,
  cost: `mlh cost p5c-clean1`.

To stop it: `pkill -f mlh-reliability.sh; pkill -f -- "--label p5c"`.

Next, in order: read p5c.log; if P5 passes (5/5 definitive, same verdict, resumes rerun no paid
stage, waste <= 20 %), run the same protocol on SimCSE (P6); only then resume sequential rounds.
If it stops, the reason is in the log and the run's stopped.json: fix, add a regression test,
rerun P5 from scratch.

Process lesson recorded: a long run must be reviewed every few minutes until it is known to be
normal; P5 attempt 2 was caught at 5 minutes this way.
