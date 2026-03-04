from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from bgap.core.config import load_config
from bgap.experiments.scenarios import apply_config_overrides, select_scenarios
from bgap.offline.solver import OfflineMILPSolver
from bgap.online.policies.dynamic_learning import DynamicLearningPolicy
from generic.experiments.run_eval import run_eval
from generic.online.policies import PrimalDualPolicy


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'.")


def _parse_args() -> argparse.Namespace:
    scenario_choices = [scenario.name for scenario in select_scenarios(None)]
    parser = argparse.ArgumentParser(
        description=(
            "Quick, low-budget mode-specific tuning for PrimalDual and DynamicLearning "
            "used in the dedicated pricing-mode comparison plot."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/bgap/bgap.yaml"),
        help="Base BGAP config path.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        type=str,
        default=[
            "baseline_midvar_off0_on100_uniform",
            "baseline_midvar_off0_on100_expbin_a2",
        ],
        choices=scenario_choices,
        help=(
            "Scenario subset used for fast tuning. "
            "Default runs both uniform and exponential feasibility."
        ),
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=scenario_choices,
        help=(
            "Deprecated alias for --scenarios with a single value. "
            "If set, overrides --scenarios."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Small seed subset for fast tuning (default: 1 2 3).",
    )
    parser.add_argument(
        "--pd-eta0s",
        nargs="*",
        type=float,
        default=[1e-5, 3e-5, 1e-4],
        help="PrimalDual eta0 candidates (constant mode, raw profile).",
    )
    parser.add_argument(
        "--pd-offline-util-init-scales",
        nargs="*",
        type=float,
        default=[1e-5, 3e-5, 1e-4],
        help="PrimalDual offline-util init scales (used only for lambda0_init=offline_util).",
    )
    parser.add_argument(
        "--pd-simlp-init-scales",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="PrimalDual sim_lp init scales (used only for lambda0_init=sim_lp).",
    )
    parser.add_argument(
        "--pd-pricing-num-samples",
        nargs="*",
        type=int,
        default=[10, 25],
        help="pricing_sim.num_samples candidates for PD sim_lp mode.",
    )
    parser.add_argument(
        "--pd-pricing-sample-online-caps",
        nargs="*",
        type=_parse_bool,
        default=[True, False],
        help="pricing_sim.sample_online_caps candidates for PD sim_lp mode.",
    )
    parser.add_argument(
        "--dla-epsilons",
        nargs="*",
        type=float,
        default=[0.01, 0.05],
        help="DLA epsilon candidates.",
    )
    parser.add_argument(
        "--dla-min-phase-lens",
        nargs="*",
        type=int,
        default=[10, 25],
        help="DLA min_phase_len candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bgap/results/quick_mode_tuning"),
        help="Output directory root.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run folder tag (default: timestamp).",
    )
    return parser.parse_args()


def _objective(summary: Dict[str, Any]) -> float | None:
    agg = summary.get("aggregate", {})
    val = agg.get("total_objective_penalized_mean", agg.get("total_objective_mean"))
    if val is None:
        return None
    return float(val)


def _is_valid(summary: Dict[str, Any]) -> bool:
    agg = summary.get("aggregate", {})
    return (
        _objective(summary) is not None
        and int(agg.get("offline_failures", 0)) == 0
        and int(agg.get("online_failures", 0)) == 0
    )


def _best_by_mode(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not row.get("valid") or row.get("objective") is None:
            continue
        mode = str(row["mode"])
        current = best.get(mode)
        if current is None or float(row["objective"]) < float(current["objective"]):
            best[mode] = row
    return best


def _aggregate_candidate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    objective_vals = [float(row["objective"]) for row in rows if row.get("objective") is not None]
    runtime_vals = [
        float(row["online_runtime_mean"])
        for row in rows
        if row.get("online_runtime_mean") is not None
    ]
    offline_failures = sum(int(row.get("offline_failures", 0) or 0) for row in rows)
    online_failures = sum(int(row.get("online_failures", 0) or 0) for row in rows)
    all_objectives_present = len(objective_vals) == len(rows) and len(rows) > 0

    objective = None
    if all_objectives_present:
        objective = float(sum(objective_vals) / len(objective_vals))
    runtime_mean = None
    if runtime_vals:
        runtime_mean = float(sum(runtime_vals) / len(runtime_vals))

    return {
        "objective": objective,
        "online_runtime_mean": runtime_mean,
        "offline_failures": offline_failures,
        "online_failures": online_failures,
        "valid": all_objectives_present and offline_failures == 0 and online_failures == 0,
    }


def main() -> None:
    args = _parse_args()
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.scenario is None and not args.scenarios:
        raise ValueError("At least one scenario is required.")
    if not args.pd_eta0s:
        raise ValueError("At least one --pd-eta0s value is required.")
    if not args.pd_offline_util_init_scales:
        raise ValueError("At least one --pd-offline-util-init-scales value is required.")
    if not args.pd_simlp_init_scales:
        raise ValueError("At least one --pd-simlp-init-scales value is required.")
    if not args.pd_pricing_num_samples:
        raise ValueError("At least one --pd-pricing-num-samples value is required.")
    if any(int(v) < 1 for v in args.pd_pricing_num_samples):
        raise ValueError("All --pd-pricing-num-samples values must be >= 1.")
    if not args.pd_pricing_sample_online_caps:
        raise ValueError("At least one --pd-pricing-sample-online-caps value is required.")
    if not args.dla_epsilons:
        raise ValueError("At least one --dla-epsilons value is required.")
    if not args.dla_min_phase_lens:
        raise ValueError("At least one --dla-min-phase-lens value is required.")

    selected_scenarios = [args.scenario] if args.scenario is not None else list(args.scenarios)
    # Preserve order but remove duplicates.
    selected_scenarios = list(dict.fromkeys(selected_scenarios))

    run_tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base_config)
    scenario_lookup = {scenario.name: scenario for scenario in select_scenarios(None)}

    scenario_defs = [scenario_lookup[name] for name in selected_scenarios]
    scenario_cfgs: Dict[str, Any] = {}
    for scenario in scenario_defs:
        cfg = apply_config_overrides(base_cfg, scenario.overrides)
        cfg.eval.seeds = tuple(args.seeds)
        cfg.solver.use_warm_start = False
        # Unneeded for this tuning and adds overhead.
        cfg.eval.track_offline_util_per_bin = False
        scenario_cfgs[scenario.name] = cfg

    seeds = list(args.seeds)
    offline_solver_name = "bgap.offline.solver.OfflineMILPSolver"

    pd_modes = ["zero", "offline_util", "sim_lp"]
    default_pd_pricing_num_samples = max(1, int(base_cfg.pricing_sim.num_samples))
    default_pd_pricing_sample_online_caps = bool(base_cfg.pricing_sim.sample_online_caps)
    pd_candidates: List[Dict[str, Any]] = []
    for mode in pd_modes:
        for eta0 in args.pd_eta0s:
            if mode == "offline_util":
                for init_scale in args.pd_offline_util_init_scales:
                    pd_candidates.append(
                        {
                            "mode": mode,
                            "eta0": float(eta0),
                            "offline_util_init_scale": float(init_scale),
                            "sim_lp_init_scale": None,
                            "pricing_num_samples": int(default_pd_pricing_num_samples),
                            "pricing_sample_online_caps": bool(default_pd_pricing_sample_online_caps),
                        }
                    )
            elif mode == "sim_lp":
                for init_scale in args.pd_simlp_init_scales:
                    for num_samples in args.pd_pricing_num_samples:
                        for sample_caps in args.pd_pricing_sample_online_caps:
                            pd_candidates.append(
                                {
                                    "mode": mode,
                                    "eta0": float(eta0),
                                    "offline_util_init_scale": None,
                                    "sim_lp_init_scale": float(init_scale),
                                    "pricing_num_samples": int(num_samples),
                                    "pricing_sample_online_caps": bool(sample_caps),
                                }
                            )
            else:
                pd_candidates.append(
                    {
                        "mode": mode,
                        "eta0": float(eta0),
                        "offline_util_init_scale": None,
                        "sim_lp_init_scale": None,
                        "pricing_num_samples": int(default_pd_pricing_num_samples),
                        "pricing_sample_online_caps": bool(default_pd_pricing_sample_online_caps),
                    }
                )

    pd_rows: List[Dict[str, Any]] = []
    for candidate in pd_candidates:
        mode = str(candidate["mode"])
        eta0 = float(candidate["eta0"])
        scenario_rows: List[Dict[str, Any]] = []
        for scenario_name in selected_scenarios:
            cfg = copy.deepcopy(scenario_cfgs[scenario_name])
            cfg.primal_dual.lambda0_init = mode
            cfg.primal_dual.eta_mode = "constant"
            cfg.primal_dual.eta0 = eta0
            cfg.primal_dual.eta_decay = 0.0
            cfg.primal_dual.eta_min = 0.0
            cfg.primal_dual.normalize_update = False
            cfg.primal_dual.normalize_costs = False
            cfg.primal_dual.use_remaining_capacity_target = True
            cfg.primal_dual.cost_scale_mode = "assign_mean"
            cfg.primal_dual.offline_util_init_scale = None
            cfg.primal_dual.sim_lp_init_scale = 1.0
            if mode == "offline_util":
                cfg.primal_dual.offline_util_init_scale = float(candidate["offline_util_init_scale"])
            if mode == "sim_lp":
                cfg.primal_dual.sim_lp_init_scale = float(candidate["sim_lp_init_scale"])
            cfg.pricing_sim.num_samples = int(candidate["pricing_num_samples"])
            cfg.pricing_sim.sample_online_caps = bool(candidate["pricing_sample_online_caps"])

            summary = run_eval(
                cfg,
                offline_solver_cls=OfflineMILPSolver,
                online_policy_cls=PrimalDualPolicy,
                seeds=seeds,
                T_onl=None,
                offline_solver_name=offline_solver_name,
                online_policy_name="generic.online.policies.PrimalDualPolicy",
            )
            agg = summary.get("aggregate", {})
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "objective": _objective(summary),
                    "online_runtime_mean": agg.get("online_runtime_mean"),
                    "offline_failures": agg.get("offline_failures"),
                    "online_failures": agg.get("online_failures"),
                    "valid": _is_valid(summary),
                }
            )

        agg_row = _aggregate_candidate(scenario_rows)
        pd_rows.append(
            {
                "mode": mode,
                "eta0": eta0,
                "offline_util_init_scale": candidate["offline_util_init_scale"],
                "sim_lp_init_scale": candidate["sim_lp_init_scale"],
                "pricing_num_samples": int(candidate["pricing_num_samples"]),
                "pricing_sample_online_caps": bool(candidate["pricing_sample_online_caps"]),
                "objective": agg_row["objective"],
                "online_runtime_mean": agg_row["online_runtime_mean"],
                "offline_failures": agg_row["offline_failures"],
                "online_failures": agg_row["online_failures"],
                "valid": agg_row["valid"],
                "by_scenario": scenario_rows,
            }
        )

    dla_modes = ["zero", "offline_util", "sim_lp"]
    dla_rows: List[Dict[str, Any]] = []
    for mode in dla_modes:
        for eps in args.dla_epsilons:
            for min_phase_len in args.dla_min_phase_lens:
                scenario_rows = []
                for scenario_name in selected_scenarios:
                    cfg = copy.deepcopy(scenario_cfgs[scenario_name])
                    cfg.dla.lambda0_init = mode
                    cfg.dla.epsilon = float(eps)
                    cfg.dla.min_phase_len = int(min_phase_len)
                    cfg.dla.use_offline_slack = True
                    cfg.dla.log_prices = False

                    summary = run_eval(
                        cfg,
                        offline_solver_cls=OfflineMILPSolver,
                        online_policy_cls=DynamicLearningPolicy,
                        seeds=seeds,
                        T_onl=None,
                        offline_solver_name=offline_solver_name,
                        online_policy_name="bgap.online.policies.dynamic_learning.DynamicLearningPolicy",
                    )
                    agg = summary.get("aggregate", {})
                    scenario_rows.append(
                        {
                            "scenario": scenario_name,
                            "objective": _objective(summary),
                            "online_runtime_mean": agg.get("online_runtime_mean"),
                            "offline_failures": agg.get("offline_failures"),
                            "online_failures": agg.get("online_failures"),
                            "valid": _is_valid(summary),
                        }
                    )

                agg_row = _aggregate_candidate(scenario_rows)
                dla_rows.append(
                    {
                        "mode": mode,
                        "epsilon": float(eps),
                        "min_phase_len": int(min_phase_len),
                        "objective": agg_row["objective"],
                        "online_runtime_mean": agg_row["online_runtime_mean"],
                        "offline_failures": agg_row["offline_failures"],
                        "online_failures": agg_row["online_failures"],
                        "valid": agg_row["valid"],
                        "by_scenario": scenario_rows,
                    }
                )

    best_pd = _best_by_mode(pd_rows)
    best_dla = _best_by_mode(dla_rows)

    meta = {
        "base_config": str(args.base_config),
        "scenarios": selected_scenarios,
        "scenario_descriptions": {
            scenario.name: scenario.description for scenario in scenario_defs
        },
        "seeds": seeds,
        "pd_grid": {
            "modes": pd_modes,
            "eta0s": [float(v) for v in args.pd_eta0s],
            "offline_util_init_scales": [float(v) for v in args.pd_offline_util_init_scales],
            "simlp_init_scales": [float(v) for v in args.pd_simlp_init_scales],
            "pricing_num_samples": [int(v) for v in args.pd_pricing_num_samples],
            "pricing_sample_online_caps": [bool(v) for v in args.pd_pricing_sample_online_caps],
            "candidate_count": int(len(pd_candidates)),
            "fixed": {
                "eta_mode": "constant",
                "normalize_update": False,
                "normalize_costs": False,
                "use_remaining_capacity_target": True,
                "cost_scale_mode": "assign_mean",
            },
        },
        "dla_grid": {
            "modes": dla_modes,
            "epsilons": [float(v) for v in args.dla_epsilons],
            "min_phase_lens": [int(v) for v in args.dla_min_phase_lens],
            "fixed": {
                "use_offline_slack": True,
            },
        },
    }

    (out_dir / "pd_results_quick.json").write_text(json.dumps({"meta": meta, "results": pd_rows}, indent=2))
    (out_dir / "best_pd_quick.json").write_text(json.dumps({"meta": meta, "best_by_mode": best_pd}, indent=2))
    (out_dir / "dla_results_quick.json").write_text(json.dumps({"meta": meta, "results": dla_rows}, indent=2))
    (out_dir / "best_dla_quick.json").write_text(json.dumps({"meta": meta, "best_by_mode": best_dla}, indent=2))

    print(f"Wrote quick mode tuning outputs to {out_dir}")
    print(f"  - {out_dir / 'best_pd_quick.json'}")
    print(f"  - {out_dir / 'best_dla_quick.json'}")


if __name__ == "__main__":
    main()
