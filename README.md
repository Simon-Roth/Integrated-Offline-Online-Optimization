# Offline/Online Allocation Framework (Generic + BGAP)

This repository implements a two-stage allocation framework:

1) Offline phase: assign a known set of offline steps to options (one-hot actions) by solving a MILP or using heuristics.
2) Online phase: assign arriving steps sequentially using online policies that can optionally evict or reassign offline steps (BGAP is a specialization).

The system is designed to be generic (A x <= b MILP interface) and also provide BGAP-specific heuristics, policies, and experiments.

BGAP means **Block-structured Generalized Assignment Problem**. We keep bin/item terminology in algorithms and analysis for readability.

The documentation below is organized by module and describes:
- what each file does,
- when it is used in the overall flow,
- and why the structures exist.

---

## Repository layout

- `generic/`
  Core, problem-agnostic framework:
  `core/` (config, models, utils), `data/`, `offline/`, `online/`, `experiments/`.
- `bgap/`
  BGAP-specific heuristics, online policies, pricing, and experiments.
- `configs/`
  Base YAML configs used by generic and bgap workflows.
- `analysis/`
  Analysis notebooks and scripts.
- `scripts/`
  Thin CLI entrypoints that call the experiment modules.
- `outputs/`
  Generated artifacts (results, plots, analysis outputs). Ignored by git.
- `requirements.txt`
  Python dependencies (NumPy, Gurobi, etc).
- `README.md`
  Hola

---

## Core workflow (high level)

The typical flow for a single experiment run is:

1) Load config (generic or bgap override).
2) Generate an instance (offline steps, online steps, local feasibility constraints, costs) from the config.
3) Assemble offline MILP data (A, b, c) from the instance.
4) Solve the offline problem (MILP or heuristic).
5) Run the online policy sequentially on online steps.
6) Aggregate results and write a JSON summary.

This flow is implemented in `generic/experiments/run_eval.py`, and is repeated
for multiple pipelines and seeds in `generic/experiments/run_multiple_evals.py` and `bgap/experiments/run_param_sweep.py`.
CLI wrappers live in `scripts/` (e.g., `scripts/run_eval.py`).

---

## Configuration

### `generic/core/config.py`
Defines all configuration dataclasses and YAML parsing.

- `ProblemConfig`: n options, T_off steps, m capacity constraints, b (or b_mean/b_std), fallback flags, allow_reassignment.
- `CapCoeffGenerationConfig`: Beta distribution and bounds for offline/online
  A_t^{cap} coefficients.
- `FeasibilityGenerationConfig`: feasibility edge probabilities.
- `CostConfig`: assignment costs, fallback cost, eviction penalty model.
- `SlackConfig`: global slack, optionally applied to online phase.
- `UtilizationPricingConfig`: settings for utilization-priced heuristic.
- `DLAConfig`: settings for Dynamic Learning Algorithm (DLA).
- `SolverConfig`: MILP settings and warm-start heuristic selection.
- `HeuristicConfig`: scalarization for vector sizes and residuals.
- `EvalConfig`: list of seeds.

### `configs/generic/generic.yaml`
Default configuration for the generic framework.

Key flags:
- `fallback_is_enabled`: whether a fallback option exists.
- `fallback_allowed_offline`: whether offline steps may use fallback.
- `fallback_allowed_online`: whether online steps may use fallback.
- `allow_reassignment`: if false, online evictions/reassignments are disallowed.

### `configs/bgap/bgap.yaml` + `bgap/core/config.py`
`bgap/core/config.py` merges `configs/generic/generic.yaml` with bgap overrides,
so bgap experiments can change only what is needed.
Defaults in `generic/core/config.py` mirror `configs/generic/generic.yaml`.

---

## Core data structures

### `generic/core/models.py`
Defines the core types that all phases share.

- `StepSpec`: id, A_t^{cap}, and local feasibility (A_t^{feas}, b_t); used for both offline and online steps.
- `Costs`: assignment costs matrix, fallback cost, eviction penalty settings.
- `Instance`: the full problem instance:
  - n, m, b
  - offline_steps
  - costs
  - fallback_option_index
  - online_steps (optional)
- `AssignmentState`: mutable state of loads and assignments.
- `Decision`: what a policy decided for one online step (placement, evictions,
  reassignments, incremental objective).

### Solution summaries (`generic/core/models.py`)
- `OfflineSolutionInfo`: algorithm name, status, objective, runtime, feasible.
- `_status_name`: maps Gurobi status codes to strings.
- `OnlineSolutionInfo`: status, runtime, total objective, fallback count,
  eviction count, and optional decisions log.

---

## Instance generation and MILP assembly

### `generic/data/instance_generators.py`
Responsible for sampling A_t^{cap}, local feasibility (A_t^{feas}, b_t), and costs from the config.

Key functions:
- `BaseInstanceGenerator.from_config(cfg).generate_full_instance(cfg, seed, T_onl=...)`:
  - samples capacity vector b (or uses provided list),
  - samples offline A_t^{cap} coefficients,
  - samples offline feasible options with probability `p_off` and encodes them as
    explicit local constraints A_t^{feas} x_t = b_t (one-hot + forbidden options),
  - samples offline assignment costs,
  - samples online A_t^{cap} coefficients and feasible options with probability `p_onl`,
    then builds A_t^{feas} x_t = b_t and calls `_ensure_row_feasible` to guarantee at least
    one feasible regular option,
  - appends online costs to the cost matrix.
- `BaseInstanceGenerator.generate_offline_instance(cfg, seed)`:
  - wrapper that calls `generate_full_instance` with `T_onl=0`.

This file is the source of truth for how local feasibility is generated:
- Offline feasibility uses `cfg.feasibility.p_off` and is encoded in A_t^{feas}, b_t.
- Online feasibility uses `cfg.feasibility.p_onl` and `_ensure_row_feasible`.
- Fallback feasibility for offline and online is controlled by
  `fallback_allowed_offline` and `fallback_allowed_online` via A_t^{feas} rows.

### BGAP vs. generic problems
BGAP is encoded in the instance generator choice and block utilities, not in
the experiment runner. In particular:

- `generation.generator: "generic"` uses `GenericInstanceGenerator` and samples
  full `m x n` capacity matrices.
- `generation.generator: "bgap"` uses `BGAPInstanceGenerator` and
  enforces the block structure (`m = n * d`).
- `bgap/core/block_utils.py` provides bgap-specific helpers such as
  `extract_volume` and `split_capacities`.
- `bgap/online/policies/*` are heuristics that assume the block structure.

`bgap/experiments/run_param_sweep.py` only orchestrates scenarios and pipelines;
it does not define the problem structure itself.

To run a generic pipeline, use `scripts/run_eval.py` (or `python -m generic.experiments.run_eval`)
with `configs/generic/generic.yaml`. To run bgap experiments, use
`scripts/run_param_sweep.py` and the merged config
(`configs/generic/generic.yaml` + `configs/bgap/bgap.yaml` via `bgap/core/config.py`).
For a generic (non-bgap) instance, set `generation.generator: "generic"`.

### `generic/data/offline_milp_assembly.py`
Builds the canonical MILP for the offline stage:
minimize c^T x subject to A x <= b.

Key functions:
- `build_offline_milp_data_from_arrays(...)`:
  - builds capacity constraints over m resources,
  - does not add capacity constraints for the fallback option,
  - adds local feasibility constraints A_t^{feas} x_t = b_t (as two inequalities).
- `build_offline_milp_data(instance, cfg)`:
  - convenience wrapper: pulls arrays from `Instance` and config.

### `generic/data/__init__.py` and `bgap/data/instance_generators.py`
`generic/data/__init__.py` re-exports the generator base and `GenericInstanceGenerator`.
`bgap/data/instance_generators.py` exposes `BGAPInstanceGenerator`.
The old block-structured implementation lives in `bgap/data/instance_generators_legacy.py`
as a backup.

---

## Offline phase

### `generic/offline/solver.py`
Generic MILP solver that works directly on A, b, c.

Flow:
- `solve(instance)`:
  - builds MILP data via `build_offline_milp_data`,
  - optionally creates a warm start,
  - calls `solve_from_data`.
- `solve_from_data(data)`:
  - builds Gurobi model from A, b, c,
  - sets solver parameters,
  - solves and extracts an `AssignmentState` and `OfflineSolutionInfo`.

### `bgap/offline/solver.py`
Extends the generic solver by adding warm-start heuristics from bgap
offline heuristics. The warm-start is optional and configured by
`cfg.solver.warm_start_heuristic`.

### `generic/offline/policies.py`
Defines the interface for any offline heuristic (`solve(instance)`).

### `bgap/offline/policies/*`
BGAP-specific offline heuristics used for warm-starts and baselines:

- `first_fit_decreasing.py` (FFD):
  sort by size and place in first feasible bin that fits.
- `best_fit_decreasing.py` (BFD):
  sort by size and place in bin with smallest residual that fits.
- `cost_best_fit_decreasing.py` (CABFD):
  sort by size and choose lowest assignment cost, tie-break by residual.
- `utilization_priced.py`:
  dynamic pricing based on bin utilization (penalize high load bins).

---

## Online phase

### `generic/online/policies.py`
Defines the `BaseOnlinePolicy` interface (`select_action`), plus
`PolicyInfeasibleError` for signaling infeasible placement.

### `generic/online/solver.py`
Orchestrates the online phase:

- iterates through online steps,
- calls `policy.select_action(...)` to get a `Decision`,
- if policy raises `PolicyInfeasibleError`, the solver can place in fallback
  if fallback is enabled and allowed for online steps,
- applies the decision to the live `AssignmentState`,
- collects `OnlineSolutionInfo`.

Important behavior:
- If `cfg.problem.allow_reassignment` is false, any decision with evictions or
  reassignments triggers an error.
- Fallback placement is controlled by:
  - `fallback_is_enabled`
  - `fallback_allowed_online`
  - per-step A_t^{feas} rows that allow or forbid the fallback option.

### `generic/online/state_utils.py` (generic core)
These are generic state mutation helpers used by the online solver:
- `clone_state`, `build_cap_lookup`
- `apply_decision`, `add_to_load`, `remove_from_load`
- `count_fallback_steps`

### `bgap/online/state_utils.py` (bgap helpers)
These are bgap-specific simulation helpers used by online heuristics:
- `PlacementContext`: a snapshot of loads and assignments for planning.
- `build_context`: builds a context with effective capacities (slack-aware).
- `execute_placement`: simulate placing an item into a bin; can evict/reassign.
- `eviction_penalty`: compute eviction penalty (per item or per usage).
- `effective_capacities`, `offline_volumes`, `TOLERANCE`.

Note: `execute_placement` mutates the context on success, so policies should
either rebuild the context per candidate (as the current policies do) or keep
this behavior in mind if new policies are added.

---

## Online policies (bgap)

Located in `bgap/online/policies/`:


- `cost_best_fit.py`:
  chooses bin with lowest incremental cost (cost aware), tie-break by residual (best fit), two-pass (-> no eviction then eviction).
- `sim_base.py`:
  legacy policy kept for reproducibility of older runs; it expects file-based
  precomputed dual prices and is not used in default pipelines.
- `dynamic_learning.py`:
  dynamic learning algorithm with phase schedule. Recomputes prices by solving
  a fractional LP on observed steps (scaled capacities).

---

## Pricing modules

### `generic/online/pricing.py`
Computes sampled LP dual prices used by price-based policies:

- uses the realized offline state to compute residual capacities,
- builds a fractional LP for a sampled set of online steps (same horizon, new seed),
- supports separate switches for sampling online A/feasibility vs. online costs:
  - `cfg.pricing_sim.sample_online_caps` controls resampling of A_t^{cap} and feasibility,
  - `cfg.pricing_sim.sample_online_costs` controls whether online costs are resampled or taken from the realized instance,
- allows fallback in the pricing LP (to guarantee successful computation) when a fallback option exists, penalized by `cfg.costs.huge_fallback`,
- pricing is in-memory (`compute_resource_prices`), used by `SimDual`
  and other policies that initialize prices from sampled LPs.

### Dynamic Learning pricing
`dynamic_learning.py` has its own LP-based pricing for each phase. It also
allows fallback in the pricing LP (to avoid infeasibility) and uses
`cfg.costs.huge_fallback` as penalty.

---

## Experiments and evaluation

### `generic/experiments/pipeline_registry.py`
Defines available offline solvers and online policies as import strings and
registers the default pipelines. Provides:
- `PipelineSpec` (name + offline solver + online policy).
- `default_registry()` to build all combinations.

### `generic/experiments/run_eval.py`
Run a single pipeline across multiple seeds and output an aggregated JSON.

Flow:
- load config,
- generate instance (offline + online),
- solve offline (MILP from A,b,c if solver supports `solve_from_data` (which is the case for our generic_solver, but e.g. nor for some bgap-specific offline heuristics like UTIL as they do not work on A,b,c but Instance)),
- run online solver,
- aggregate results and write JSON to `outputs/generic/results`.

Key CLI args:
```
python scripts/run_eval.py \
  --config configs/generic/generic.yaml \
  --offline-solver generic.offline.solver.OfflineMILPSolver \
  --online-policy bgap.online.policies.cost_best_fit.CostAwareBestFitOnlinePolicy \
  --seeds 1 2 3 \
  --m-onl 100
```

### `generic/experiments/run_multiple_evals.py`
Run multiple pipelines from the registry and output one JSON per pipeline.
Optional: compute a full-horizon optimal benchmark once per seed.

Key CLI args:
```
python scripts/run_multiple_evals.py \
  --config configs/generic/generic.yaml \
  --pipelines bgap_milp+sim_dual util+cost_best_fit \
  --seeds 1 2 3 \
  --m-onl 100 \
  --compute-optimal
```

### `generic/experiments/optimal_benchmark.py`
Builds a full-horizon MILP by merging offline and online steps, then solves it
offline. This gives a lower bound or benchmark for evaluation.

---

## BGAP experiments

### `bgap/experiments/scenarios.py`
Defines scenario families for parameter sweeps:
- ratio sweeps (T_off vs T_onl),
- coefficient variance families,
- graph sparsity variations,
- load regimes and other controls.

### `bgap/experiments/run_param_sweep.py`
Uses bgap config + scenarios to run the generic evaluation pipeline across
many settings and writes outputs to `outputs/bgap/results/param_sweep/<scenario>/`.

Example:
```
python scripts/run_param_sweep.py \
  --base-config configs/bgap/bgap.yaml \
  --scenarios baseline_midvar_off40_on60 \
  --pipelines bgap_milp+cost_best_fit \
  --seeds 1 2 3
```

---

## Output JSON structure (overview)

`generic/experiments/run_eval.py` produces summaries that include:
- per-run offline/online status, objective, runtime,
- aggregate means across seeds,
- status counts and failure counts.

`run_multiple_evals.py` adds a `pipeline` field and writes one JSON per pipeline.
`optimal_benchmark.py` writes an optimal full-horizon JSON.

---

## Setup and dependencies

Install dependencies:
```
pip install -r requirements.txt
```

Notes:
- Gurobi is required (`gurobipy`). You need a valid license set up for your
  environment.
- `scipy` is required by some Gurobi helpers when using matrix constraints (is in requirements.txt).

---

## Legacy or deprecated components 

default.yaml 
