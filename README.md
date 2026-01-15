# Offline/Online Allocation Framework (Generic + Binpacking)

This repository implements a two-stage allocation framework:

1) Offline phase: assign a known set of items to bins by solving a MILP or using heuristics.
2) Online phase: assign arriving items sequentially using online policies that can optionally evict or reassign offline items (binpacking setting).

The system is designed to be generic (A x <= b MILP interface) and also provide binpacking-specific heuristics, policies, and experiments.

The documentation below is organized by module and describes:
- what each file does,
- when it is used in the overall flow,
- and why the structures exist.

---

## Repository layout

- `generic/`
  Core, problem-agnostic framework: config, data generation, MILP assembly,
  generic offline/online solvers, experiment runners, and shared models.
- `binpacking/`
  Binpacking-specific heuristics, online policies, pricing, experiments, and analysis.
- `configs/`
  Base YAML configs used by generic and binpacking workflows.
- `requirements.txt`
  Python dependencies (NumPy, Gurobi, etc).
- `README.md`
  Hola

---

## Core workflow (high level)

The typical flow for a single experiment run is:

1) Load config (generic or binpacking override).
2) Generate an instance (offline items, online items, feasibility graphs, costs) from the config.
3) Assemble offline MILP data (A, b, c) from the instance.
4) Solve the offline problem (MILP or heuristic).
5) Run the online policy sequentially on online items.
6) Aggregate results and write a JSON summary.

This flow is implemented in `generic/experiments/run_eval.py`, and is repeated
for multiple pipelines and seeds in `generic/experiments/run_multiple_evals.py` and `binpacking/experiments/run_param_sweep.py`.

---

## Configuration

### `generic/config.py`
Defines all configuration dataclasses and YAML parsing.

- `ProblemConfig`: N bins, M_off items, dimensions, capacities or distribution, fallback flags, binpacking flag, fallback capacities.
- `VolumeGenerationConfig`: Beta distribution and bounds for offline/online
  item sizes.
- `GraphGenerationConfig`: feasibility graph edge probabilities.
- `CostConfig`: assignment costs, fallback cost, eviction penalty model.
- `SlackConfig`: global slack, optionally applied to online phase.
- `UtilizationPricingConfig`: settings for utilization-priced heuristic.
- `DLAConfig`: settings for Dynamic Learning Algorithm (DLA).
- `SolverConfig`: MILP settings and warm-start heuristic selection.
- `HeuristicConfig`: scalarization for vector sizes and residuals.
- `EvalConfig`: list of seeds.

### `configs/generic.yaml`
Default configuration for the generic framework.

Key flags:
- `fallback_is_enabled`: whether a fallback bin exists.
- `fallback_allowed_offline`: whether offline items may use fallback.
- `fallback_allowed_online`: whether online items may use fallback.
- `binpacking`: if false, online evictions/reassignments are disallowed, as this is specific logic implemented via the policies in /binpacking.

### `configs/binpacking.yaml` + `binpacking/config.py`
`binpacking/config.py` merges `configs/generic.yaml` with binpacking overrides,
so binpacking experiments can change only what is needed.

---

## Core data structures

### `generic/models.py`
Defines the core types that all phases share.

- `BinSpec`: id and capacity vector.
- `ItemSpec`: id and volume vector.
- `OnlineItem`: arriving online item, with feasible regular bins list.
- `Costs`: assignment costs matrix, fallback cost, eviction penalty settings.
- `FeasibleGraph`: item-by-bin feasibility mask (0/1).
- `Instance`: the full problem instance:
  - bins
  - offline_items
  - costs
  - offline_feasible
  - fallback_bin_index
  - online_items (optional)
  - online_feasible (optional)
- `AssignmentState`: mutable state of loads and assignments.
- `Decision`: what a policy decided for one online item (placement, evictions,
  reassignments, incremental objective).

### `generic/offline/models.py`
- `OfflineSolutionInfo`: algorithm name, status, objective, runtime, feasible.
- `_status_name`: maps Gurobi status codes to strings.

### `generic/online/models.py`
- `OnlineSolutionInfo`: status, runtime, total objective, fallback count,
  eviction count, and optional decisions log.

---

## Instance generation and MILP assembly

### `generic/data/instance_generators.py`
Responsible for sampling bins, items, feasibility, and costs from the config.

Key functions:
- `generate_instance_with_online(cfg, seed, M_onl=...)`:
  - samples bin capacities (or uses provided list),
  - samples offline volumes,
  - builds offline feasibility mask with probability `p_off`,
  - adds fallback column if fallback is enabled and allowed,
  - samples offline assignment costs,
  - samples online volumes and online feasibility mask with probability `p_onl`, then calls `_ensure_row_feasible` to guarantee each online item has at least one feasible regular bin,
  - appends online costs to the cost matrix.
- `generate_offline_instance(cfg, seed)`:
  - wrapper that calls `generate_instance_with_online` with `M_onl=0`.

This file is the source of truth for how feasibility is generated:
- Offline feasibility uses `cfg.graphs.p_off`.
- Online feasibility uses `cfg.graphs.p_onl` and `_ensure_row_feasible`.
- Fallback feasibility for offline and online is controlled by
  `fallback_allowed_offline` and `fallback_allowed_online`.

### `generic/data/offline_milp_assembly.py`
Builds the canonical MILP for the offline stage:
minimize c^T x subject to A x <= b.

Key functions:
- `build_offline_milp_data_from_arrays(...)`:
  - builds capacity constraints (per bin, per dimension),
  - adds fallback capacity constraints when fallback is enabled,
  - adds assignment constraints using two inequalities (sum == 1),
  - adds infeasibility constraints (sum of infeasible edges == 0).
- `build_offline_milp_data(instance, cfg)`:
  - convenience wrapper: pulls arrays from `Instance` and config.

### `generic/data/__init__.py` and `binpacking/data/instance_generators.py`
These are re-exports to keep imports clean. Binpacking does not maintain its own generator logic.

---

## Offline phase

### `generic/offline/offline_solver.py`
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

### `binpacking/offline/offline_solver.py`
Extends the generic solver by adding warm-start heuristics from binpacking
offline heuristics. The warm-start is optional and configured by
`cfg.solver.warm_start_heuristic`.

### `generic/offline/offline_policies.py`
Defines the interface for any offline heuristic (`solve(instance)`).

### `binpacking/offline/offline_heuristics/*`
Binpacking-specific offline heuristics used for warm-starts and baselines:

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
Defines the `BaseOnlinePolicy` interface (`select_bin`), plus
`PolicyInfeasibleError` for signaling infeasible placement.

### `generic/online/online_solver.py`
Orchestrates the online phase:

- iterates through online items,
- calls `policy.select_bin(...)` to get a `Decision`,
- if policy raises `PolicyInfeasibleError`, the solver can place in fallback
  if fallback is enabled and allowed for online items,
- applies the decision to the live `AssignmentState`,
- collects `OnlineSolutionInfo`.

Important behavior:
- If `cfg.problem.binpacking` is false, any decision with evictions or
  reassignments triggers an error. This keeps generic runs eviction-free.
- Fallback placement is controlled by:
  - `fallback_is_enabled`
  - `fallback_allowed_online`
  - online feasibility mask (fallback column).

### `generic/online/state_utils.py` (generic core)
These are generic state mutation helpers used by the online solver:
- `clone_state`, `build_volume_lookup`
- `apply_decision`, `add_to_bin`, `remove_from_bin`
- `count_fallback_items`

### `binpacking/online/state_utils.py` (binpacking helpers)
These are binpacking-specific simulation helpers used by online heuristics:
- `PlacementContext`: a snapshot of loads and assignments for planning.
- `build_context`: builds a context with effective capacities (slack-aware).
- `execute_placement`: simulate placing an item into a bin; can evict/reassign.
- `eviction_penalty`: compute eviction penalty (per item or per volume).
- `effective_capacities`, `offline_volumes`, `TOLERANCE`.

Note: `execute_placement` mutates the context on success, so policies should
either rebuild the context per candidate (as the current policies do) or keep
this behavior in mind if new policies are added.

---

## Online policies (binpacking)

Located in `binpacking/online/online_heuristics/`:


- `cost_best_fit.py`:
  chooses bin with lowest incremental cost (cost aware), tie-break by residual (best fit), two-pass (-> no eviction then eviction).
- `primal_dual.py`:
  uses precomputed dual prices (lambda) to score bins:
  cost + lambda * volume. Fallback is handled by the solver.
- `dynamic_learning.py`:
  dynamic learning algorithm with phase schedule. Recomputes prices by solving
  a fractional LP on observed items (scaled capacities).

---

## Pricing modules

### `binpacking/online/prices.py`
Computes dual prices for primal-dual:

- uses the realized offline state to compute residual capacities,
- builds a fractional LP for online items,
- allows fallback in the pricing LP (to guarantee sucessfull computation) when a fallback bin exists, penalized by `cfg.costs.huge_fallback`,
- writes prices to `binpacking/results/primal_dual.json`.

This is called per seed inside `generic/experiments/run_eval.py` when the
online policy requires prices.

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
- `online_policy_needs_prices(...)` to identify policies that need pricing.

### `generic/experiments/run_eval.py`
Run a single pipeline across multiple seeds and output an aggregated JSON.

Flow:
- load config,
- generate instance (offline + online),
- solve offline (MILP from A,b,c if solver supports `solve_from_data` (which is the case for our generic_solver, but e.g. nor for some binpacking-specific offline heuristics like UTIL as they do not work on A,b,c but Instance)),
- compute prices if needed (primal-dual),
- run online solver,
- aggregate results and write JSON to `generic/results`.

Key CLI args:
```
python -m generic.experiments.run_eval \
  --config configs/generic.yaml \
  --offline-solver generic.offline.offline_solver.OfflineMILPSolver \
  --online-policy binpacking.online.online_heuristics.cost_best_fit.CostAwareBestFitOnlinePolicy \
  --seeds 1 2 3 \
  --m-onl 100
```

### `generic/experiments/run_multiple_evals.py`
Run multiple pipelines from the registry and output one JSON per pipeline.
Optional: compute a full-horizon optimal benchmark once per seed.

Key CLI args:
```
python -m generic.experiments.run_multiple_evals \
  --config configs/generic.yaml \
  --pipelines binpacking_milp+primal_dual util+cost_best_fit \
  --seeds 1 2 3 \
  --m-onl 100 \
  --compute-optimal
```

### `generic/experiments/optimal_benchmark.py`
Builds a full-horizon MILP by merging offline and online items, then solves it
offline. This gives a lower bound or benchmark for evaluation.

---

## Binpacking experiments

### `binpacking/experiments/scenarios.py`
Defines scenario families for parameter sweeps:
- ratio sweeps (M_off vs M_onl),
- volume variance families,
- graph sparsity variations,
- load regimes and other controls.

### `binpacking/experiments/run_param_sweep.py`
Uses binpacking config + scenarios to run the generic evaluation pipeline across
many settings and writes outputs to `binpacking/results/param_sweep/<scenario>/`.

Example:
```
python -m binpacking.experiments.run_param_sweep \
  --base-config configs/binpacking.yaml \
  --scenarios baseline_midvar_off40_on60 \
  --pipelines binpacking_milp+cost_best_fit \
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

