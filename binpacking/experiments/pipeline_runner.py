from __future__ import annotations

from typing import Optional, Tuple
import copy

from generic.config import Config
from generic.experiments.pipeline import PipelineSpec
from binpacking.experiments.utils import (
    build_pipeline_summary,
    save_combined_result,
    build_empty_offline_solution,
)
from pathlib import Path
from generic.online.online_solver import OnlineSolver
from binpacking.online.prices import compute_prices
from generic.online.state_utils import count_fallback_items
from binpacking.data.instance_generators import generate_instance_with_online
from generic.models import AssignmentState, Instance
from generic.offline.models import OfflineSolutionInfo
from generic.online.models import OnlineSolutionInfo


def run_pipeline(
    cfg: Config,
    spec: PipelineSpec,
    *,
    seed: Optional[int] = None,
    output_dir: str = "binpacking/results",
    instance: Optional[Instance] = None,
    offline_solution: Optional[Tuple[AssignmentState, OfflineSolutionInfo]] = None,
) -> Tuple[dict, str]:
    """
    Execute a single offline+online pipeline described by `spec`.

    Parameters
    ----------
    cfg:
        Base configuration. (Caller may provide a copy if reuse is required.)
    spec:
        Pipeline specification containing factories and labels.
    seed:
        Seed used for instance generation; defaults to the first entry in cfg.eval.seeds.
    output_dir:
        Directory where the JSON summary will be written.
    instance:
        Optional pre-generated instance (deep-copied internally) to keep runs aligned.
    offline_solution:
        Optional cached offline result (state, info) to skip re-solving the offline part.

    Returns
    -------
    summary, path:
        JSON-friendly summary dictionary and the file path it was written to.
    """
    chosen_seed = cfg.eval.seeds[0] if seed is None else seed
    if instance is None:
        base_instance = generate_instance_with_online(cfg, seed=chosen_seed)
    else:
        base_instance = copy.deepcopy(instance)

    offline_count = len(base_instance.offline_items)
    online_count = len(base_instance.online_items)

    if offline_count == 0:
        offline_state, offline_info = build_empty_offline_solution(base_instance)
    elif offline_solution is not None:
        offline_state = copy.deepcopy(offline_solution[0])
        offline_info = copy.deepcopy(offline_solution[1])
    else:
        offline_solver = spec.offline_factory(cfg)
        offline_state, offline_info = offline_solver.solve(base_instance)

    # Short-circuit when there are no online items: skip pricing/policy construction.
    if online_count == 0:
        final_state = copy.deepcopy(offline_state)
        online_info = OnlineSolutionInfo(
            status="NO_ITEMS",
            runtime=0.0,
            total_cost=0.0,
            fallback_items=count_fallback_items(final_state, base_instance),
            evicted_offline=0,
            decisions=[],
        )
        summary = build_pipeline_summary(
            spec.name,
            chosen_seed,
            cfg,
            base_instance,
            offline_state,
            final_state,
            offline_info,
            online_info,
            offline_method=spec.offline_label,
            online_method=spec.online_label,
            slack_config=cfg.slack,
            dla_config=cfg.dla,
        )
        problem = summary["problem"]
        prefix = (
            f"pipeline_{spec.name.replace('+', '_').replace(' ', '_')}_"
            f"N{problem['N']}_Moff{problem['M_off']}_Mon{problem['M_on']}_seed{chosen_seed}"
        )
        path = save_combined_result(prefix, summary, output_dir)
        return summary, path

    aux_path: Path | None = None
    if online_count > 0:
        if spec.online_label == "PrimalDual":
            aux_path = Path(output_dir) / f"prices_{spec.online_label}_{spec.name}_seed{chosen_seed}.json"
            aux_path.parent.mkdir(parents=True, exist_ok=True)
            # Sample an independent instance from the same distribution to avoid lookahead
            # bias when computing prices.
            pricing_seed = chosen_seed + 1  # deterministic but different instance
            pricing_instance = generate_instance_with_online(cfg, seed=pricing_seed)
            if len(pricing_instance.offline_items) == 0:
                pricing_offline_state, _ = build_empty_offline_solution(pricing_instance)
            else:
                pricing_offline_solver = spec.offline_factory(cfg)
                pricing_offline_state, _ = pricing_offline_solver.solve(pricing_instance)
            compute_prices(cfg, pricing_instance, pricing_offline_state, aux_path)
        elif spec.online_label == "DynamicLearning" and cfg.dla.log_prices:
            aux_path = Path(cfg.dla.output_dir) / f"dla_log_{spec.name}_seed{chosen_seed}.json"
            aux_path.parent.mkdir(parents=True, exist_ok=True)
            
            
    # Instantiate online policy, passing freshly computed prices when required.
    if spec.online_label == "PrimalDual":
        if aux_path is None or not aux_path.exists():
            raise FileNotFoundError(
                f"PrimalDual prices file missing for seed {chosen_seed} at {aux_path}. "
                "Ensure prices are precomputed (online_count > 0) before running."
            )
        online_policy = spec.online_factory(cfg, aux_path)
    elif spec.online_label == "BalancedPrices":
        # placeholder for potential other pd approaches
        placeholder = 0
    elif spec.online_label == "DynamicLearning":
        online_policy = spec.online_factory(cfg, aux_path if cfg.dla.log_prices else None)
    else:
        online_policy = spec.online_factory(cfg)
    online_solver = OnlineSolver(cfg, online_policy)
    final_state, online_info = online_solver.run(base_instance, offline_state)

    summary = build_pipeline_summary(
        spec.name,
        chosen_seed,
        cfg,
        base_instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
        offline_method=spec.offline_label,
        online_method=spec.online_label,
        slack_config=cfg.slack,
        dla_config=cfg.dla,
    )

    problem = summary["problem"]
    prefix = (
        f"pipeline_{spec.name.replace('+', '_').replace(' ', '_')}_"
        f"N{problem['N']}_Moff{problem['M_off']}_Mon{problem['M_on']}_seed{chosen_seed}"
    )
    path = save_combined_result(prefix, summary, output_dir)
    return summary, path
