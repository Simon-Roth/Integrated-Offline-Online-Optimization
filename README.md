# Offline/Online Allocation Framework (Generic + BGAP)

This repository implements a integrated optimization framework:

1) Offline phase: assign a known set of offline steps to options (one-hot actions) by solving a MILP or using heuristics.
2) Online phase: assign arriving steps sequentially using online policies.

The system is designed to be generic (A x <= b MILP interface) and also provide BGAP-specific heuristics, policies, and experiments.

BGAP means **Block-structured Generalized Assignment Problem** and serves as the formulation we use for a hospital case study, where items (patients) have to be assigned to bins (operating/surgery rooms).

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
  BGAP-specific heuristics, online policies, and experiments.
- `configs/`
  Runtime YAML configs used by generic and bgap workflows:
  `generic/` (generic base), `bgap/` (bgap base override), `presets/` (optional tuning presets).
- `analysis/`
  Analysis notebooks and scripts.
- `scripts/`
  Thin CLI entrypoints that call the experiment modules.
- `outputs/`
  Generated artifacts (results, plots, analysis outputs). Ignored by git.
- `requirements.txt`
  Python dependencies (NumPy, Gurobi, etc).
- `README.md`
  Hola.

---

## Core workflow (high level)

The typical flow for a single experiment run is:

1) Load config (generic or bgap override).
2) Generate an instance (offline steps, online steps, local feasibility constraints, costs) from the config (e.g., via. BGAPInstanceGenerator, GenericInstanceGenerator, or a custom generator using the BaseInstanceGenerator framework).
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
- `FeasibilityGenerationConfig`: feasibility edge probabilities (uniform or exp_bin mode).
- `CostConfig`: assignment costs, fallback cost, eviction penalty model.
- `SlackConfig`: global slack, optionally applied to online phase.
- `UtilizationPricingConfig`: settings for utilization-priced heuristic.
- `DLAConfig`: settings for Dynamic Learning Algorithm (DLA).
- `PricingSimulationConfig`: shared sampled LP pricing settings (num_samples, fallback override) (used for initializing prices of certain price-based policies)
- `RollingMILPConfig`: rolling-horizon MILP controls (single or batch rollout mode).
- `PrimalDualConfig`: primal-dual online policy controls (step-size schedule, normalization, lambda0 init).
- `SolverConfig`: MILP settings and warm-start heuristic selection (only "CABFD" and "none" supported).
- `HeuristicConfig`: scalarization for vector sizes and residuals (eg for residual scalarizatino in CaBfd).
- `EvalConfig`: list of seeds, offline utilization tracking flag.
- `GenerationConfig`: instance generator selection ("generic" or "bgap").

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

### `configs/presets/*.yaml`
Optional tuning presets (for example initialization/mode comparisons).
They are not selected automatically by default runners; pass them explicitly via
`--config` or `--base-config` when you want those settings.

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

- `BaseInstanceGenerator` (abstract base):
  - `from_config(cfg)`: factory that selects `GenericInstanceGenerator` or `BGAPInstanceGenerator`.
  - `generate_full_instance(cfg, seed, T_onl=...)`: generates offline + online in one pass.
  - `generate_offline_instance(cfg, seed)`: wrapper that calls `generate_full_instance` with `T_onl=0`.
  - `resample_online_phase(cfg, instance, seed=...)`: returns a copy of `instance` with freshly sampled online steps (used for pricing).
  - `sample_cap_matrices(cfg, rng, count, n, m, phase=...)`: samples capacity matrices for pricing/rolling MILP.

- `GenericInstanceGenerator`: full m×n capacity matrices per step.

Generation details:
- Offline feasibility uses `cfg.feasibility.p_off` (or exp_bin params) and is encoded in A_t^{feas}, b_t.
- Online feasibility uses `cfg.feasibility.p_onl` and `_ensure_row_feasible`.
- Fallback feasibility for offline and online is controlled by
  `fallback_allowed_offline` and `fallback_allowed_online` via A_t^{feas} rows.

### `generic/data/generator_utils.py`
Low-level sampling helpers shared by all generators:
- `_coerce_b`: builds capacity vector b from config (sampled or fixed).
- `_ensure_row_feasible`: guarantees at least one feasible option per step (to avoid trivial cases).
- `_build_feas_constraints`: builds A_t^{feas}, b_t (one-hot + forbidden options).
- `_sample_feas_mask_by_option`: samples feasibility masks (uniform or exp_bin).
- `_sample_assignment_costs`: samples costs from Beta distribution.
- `_normalize_beta_params`, `_normalize_bounds`, `_cap_params_for_phase`: normalization helpers.

### BGAP vs. generic problems
BGAP is encoded in the instance generator choice and block utilities, not in
the experiment runner. In particular:

- `generation.generator: "generic"` uses `GenericInstanceGenerator` and samples
  full `m x n` capacity matrices.
  For a generic (non-bgap) instance, set `generation.generator: "generic"`.
- `generation.generator: "bgap"` uses `BGAPInstanceGenerator` and
  enforces the block structure (`m = n * d`).
- `bgap/core/block_utils.py` provides bgap-specific helpers such as
  `extract_volume` and `split_capacities`.
- `bgap/online/policies/*` are heuristics that assume the block structure.

`bgap/experiments/run_param_sweep.py` only orchestrates scenarios and pipelines for bgap specific ceteris-paribus sweeps;
it does not define the problem structure itself.

To run a generic pipeline, use `scripts/run_eval.py` (or `python -m generic.experiments.run_eval`)
with `configs/generic/generic.yaml`. To run bgap experiments, use
the merged config

### `generic/data/offline_milp_assembly.py`
Builds the canonical MILP for the offline stage:
minimize c^T x subject to A x <= b. (we chose this representation for generality and to reflect many opt. problems; e.g., = can be expressed as <= and >=)

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
`cfg.solver.warm_start_heuristic`. Currently `"CABFD"` and `"none"` are implemented.

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
- `utilization_priced.py` (UTIL):
  conservative planning based on bin utilization prices (penalize high load bins).
- `_policy_utils.py`:
  shared helpers: `sorted_steps_by_volume`, `init_loads_and_caps`, `objective_with_fallback`.

---

## Online phase

### `generic/online/policies.py`
Defines the `BaseOnlinePolicy` interface (`select_action`, `begin_instance`), plus re-exports `PolicyInfeasibleError`.

Also contains the generic policy implementations:
- `PrimalDualPolicy`: online primal-dual policy. Each arriving step is placed by solving a one-step MILP with price-modified costs; dual prices are updated after each decision.
- `SimDualPolicy`: static-price variant — initializes lambda via sampled LP pricing (`sim_lp`), then keeps prices fixed. Delegates to `PrimalDualPolicy` with zero step size.
- `RollingHorizonMILPPolicy`: solves a MILP over the current step + sampled remaining steps (single or batch rollout mode).
- `GenericDynamicLearningPolicy`: dynamic price-updating policy using a geometric phase schedule. Recomputes prices by solving a fractional LP over observed steps. No evictions (generic version).

### `generic/online/policy_utils.py`
Shared helpers for online policies:
- `PolicyInfeasibleError`: raised when a policy cannot place the arriving step.
- `current_cost_row`: returns the assignment cost row for a step.
- `remaining_capacities`: computes b - usage, with optional slack.
- `lookup_assignment_cost`: returns the cost for a specific step→option pair.

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
  reassignments triggers fallback handling if available (not an error — the solver routes it to fallback).
- Fallback placement is controlled by:
  - `fallback_is_enabled`
  - `fallback_allowed_online`
  - per-step A_t^{feas} rows that allow or forbid the fallback option.
- `cfg.costs.stop_online_on_first_failure`: if True, stop at first unplaceable step; if False, continue.

### `generic/online/state_utils.py` (generic core)
Generic state mutation helpers used by the online solver:
- `clone_state`, `build_cap_lookup`
- `apply_decision`, `add_to_load`, `remove_from_load`
- `count_fallback_steps`

### `bgap/online/state_utils.py` (bgap helpers)
BGAP-specific simulation helpers used by online heuristics:
- `PlacementContext`: a snapshot of loads and assignments for planning.
- `build_context`: builds a context with effective capacities (slack-aware).
- `execute_placement`: simulate placing an item into a bin; can evict/reassign.
  Mutates the context on success — policies rebuild the context per candidate to isolate attempts.
- `eviction_penalty`: compute eviction penalty (per item or per usage).
- `eviction_order_desc`: offline items in a bin sorted by size (descending), used as eviction order.
- `select_reassignment_bin`: choose a reassignment destination for an evicted offline item (cost or residual mode).
- `candidate_bins`: feasible regular bins for an online step.
- `effective_capacities`, `offline_volumes`, `TOLERANCE`.

---

## Online policies (bgap)

Located in `bgap/online/policies/`:

- `cost_best_fit.py` (`CostAwareBestFitOnlinePolicy`):
  chooses bin with lowest incremental cost (cost aware), tie-break by residual (best fit). Two-pass: first without eviction, then with eviction if needed and allowed.
- `best_fit.py` (`BestFitOnlinePolicy`):
  greedy best-fit by residual capacity. Two-pass: first fit-without-eviction, then overflow bins with eviction allowed.
- `sim_base.py` (`SimBasePolicy`):
  legacy policy kept for reproducibility of older runs; expects file-based precomputed dual prices and is not used in default pipelines -> was replaced by SimDual generic policy.
- `dynamic_learning.py` (`DynamicLearningPolicy`):
  extends `GenericDynamicLearningPolicy` with bgap-specific eviction logic. When no regular bin fits (capacity exceeded), tries evictions for each candidate bin.

---

## Pricing modules

### `generic/online/pricing.py`
Computes sampled LP dual prices used by price-based policies:

- uses the realized offline state to compute residual capacities,
- builds a fractional LP for a sampled set of online steps (same horizon, new seed),
- supports separate switches for sampling online A/feasibility vs. online costs:
  - `cfg.pricing_sim.sample_online_caps` controls resampling of A_t^{cap} and feasibility,
  - `cfg.costs.observe_future_online_costs` controls whether online costs are resampled or taken from the realized instance,
- allows fallback in the pricing LP (to guarantee successful computation) when a fallback option exists, penalized by `cfg.costs.huge_fallback`,
  controlled by `cfg.pricing_sim.fallback_allowed_online_for_pricing`,
- pricing is in-memory (`compute_resource_prices`), used by `SimDualPolicy`
  and `PrimalDualPolicy` when `lambda0_init == "sim_lp"`.

### Dynamic Learning pricing
`GenericDynamicLearningPolicy` (and `DynamicLearningPolicy`) have their own LP-based pricing for each phase. Uses observed steps up to the phase boundary t_k, with a scaled residual capacity. Also allows fallback in the pricing LP and uses `cfg.costs.huge_fallback` as penalty. Phase schedule is geometric: t_k = ceil(ε · T_onl · 2^k).

---

## Experiments and evaluation

### `generic/experiments/pipeline_registry.py`
Defines available offline solvers and online policies as import strings and
registers the default pipelines. Provides:
- `PipelineSpec` (name + offline solver + online policy).
- `default_registry()` to build all combinations.

Registered offline solvers: `bgap_milp` (bgap MILP), `cabfd` (CABFD heuristic), `util` (UtilizationPriced).
Registered online policies: `rolling_horizon_milp`, `cost_best_fit`, `sim_dual`, `dynamic_learning`, `primal_dual`.

### `generic/experiments/run_eval.py`
Run a single pipeline across multiple seeds and output an aggregated JSON.

Flow:
- load config,
- generate instance (offline + online),
- solve offline (MILP from A,b,c if solver supports `solve_from_data`, otherwise via `solve`),
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
**Thesis experiments** can be reconstructured using `run_param_sweep.py` and the generic config copy.

### `bgap/experiments/scenarios.py`
Defines scenario families for parameter sweeps:
- ratio sweeps (T_off vs T_onl),
- coefficient variance families,
- graph sparsity variations,
- reshuffling scenarios.

Each scenario can be combined with feasibility variants (uniform or exp_bin) via `FEAS_VARIANTS_ACTIVE`.
Scenarios are appended to the global `SCENARIO_SWEEP` list at module import time.
`select_scenarios(names)` filters by name and raises on unknown names.

### `bgap/experiments/run_param_sweep.py`
Uses bgap config + scenarios to run the generic evaluation pipeline across
many settings and writes outputs to `outputs/bgap/results/param_sweep/<scenario>/`.

Flags:
- `--skip-optimal`: skip the full-horizon optimal benchmark per scenario.
- `--only-optimal`: run only the optimal benchmark (no pipelines).

Example:
```
python scripts/run_param_sweep.py \
  --base-config configs/bgap/bgap.yaml \
  --scenarios baseline_midvar_off40_on60_uniform \
  --pipelines bgap_milp+cost_best_fit \
  --seeds 1 2 3
```

---

## Grid search modules

### `generic/experiments/grid_search/primal_dual_grid_search.py`
Grid search over `PrimalDualPolicy` hyperparameters:
- Three grid profiles: `raw` (no normalization), `norm_update`, `norm_update_costs`.
- Sweeps: `eta_mode`, `eta0`, `eta_decay`, `eta_min`, `normalize_update`, `normalize_costs`, `use_remaining_capacity_target`, `cost_scale_mode`, `lambda0_init`, `pricing_num_samples`, `pricing_sample_online_caps`.
- Supports checkpoint/resume via `--resume-dir`.
- Outputs `results.json`, `results.csv`, and `best.json` to the output directory.

```
python scripts/grid_search/run_primal_dual_grid_search.py \
  --config configs/generic/generic.yaml \
  --horizon 300 \
  --seeds 1 2 3 5
```

### `bgap/experiments/grid_search/dla_grid_search.py`
Grid search over `DynamicLearningPolicy` hyperparameters:
- Sweeps: `epsilon`, `min_phase_len`, `use_offline_slack` across multiple horizons.
- Outputs `results.json`, `results.csv`, `combo_results.csv`, and `best.json`.

```
python scripts/grid_search/run_dla_grid_search.py \
  --base-config configs/bgap/bgap.yaml \
  --horizons 60 100 150
```

### `bgap/experiments/grid_search/util_pricing_grid_search.py`
Grid search over `UtilizationPricedDecreasing` hyperparameters (offline heuristic).

---

## Output JSON structure (overview)

`generic/experiments/run_eval.py` produces summaries that include:
- per-run offline/online status, objective, runtime,
- aggregate means across seeds (over completed runs only),
- status counts and failure counts,
- optional `offline_util_per_bin` (per-bin utilization after offline phase, if `track_offline_util_per_bin` is enabled),
- penalized objectives (`*_objective_penalized`) using `fail_penalty_per_item`.

`run_multiple_evals.py` adds a `pipeline` field and writes one JSON per pipeline.
`optimal_benchmark.py` writes an optimal full-horizon JSON.
`run_param_sweep.py` adds `scenario`, `scenario_description`, and `problem` fields to each output.

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

- `bgap/data/instance_generators_legacy.py`: old block-structured generator, kept as backup.
- `bgap/online/policies/sim_base.py` (`SimBasePolicy`): legacy policy that reads file-based prices; not used in default pipelines.
