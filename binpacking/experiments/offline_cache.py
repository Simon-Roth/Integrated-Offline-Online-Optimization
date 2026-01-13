from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import hashlib
import numpy as np

from generic.models import AssignmentState
from generic.offline.models import OfflineSolutionInfo

CACHE_DIR = Path("binpacking/results/offline_cache")


def compute_config_signature(config_path: Path) -> str:
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_cached_full_horizon(
    config_sig: str,
    seed: int,
) -> Optional[Tuple[AssignmentState, OfflineSolutionInfo]]:
    path = _full_horizon_path(config_sig, seed)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        load = data["load"].astype(float)
        assigned_keys = data["assigned_keys"].astype(int)
        assigned_vals = data["assigned_vals"].astype(int)
        assigned_bin = {int(k): int(v) for k, v in zip(assigned_keys, assigned_vals)}
        offline_state = AssignmentState(load=load, assigned_bin=assigned_bin, offline_evicted=set())
        info = OfflineSolutionInfo(
            status=str(data["info_status"].item()),
            obj_value=float(data["info_obj_value"].item()),
            mip_gap=float(data["info_mip_gap"].item()),
            runtime=float(data["info_runtime"].item()),
            assignments=data["info_assignments"].astype(int),
        )
        return offline_state, info


def save_cached_full_horizon(
    config_sig: str,
    seed: int,
    state: AssignmentState,
    info: OfflineSolutionInfo,
) -> None:
    path = _full_horizon_path(config_sig, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    assigned_keys = np.array(list(state.assigned_bin.keys()), dtype=int)
    assigned_vals = np.array(list(state.assigned_bin.values()), dtype=int)
    np.savez(
        path,
        load=state.load.astype(float, copy=True),
        assigned_keys=assigned_keys,
        assigned_vals=assigned_vals,
        info_status=np.array(info.status),
        info_obj_value=np.array(info.obj_value),
        info_mip_gap=np.array(info.mip_gap),
        info_runtime=np.array(info.runtime),
        info_assignments=info.assignments.astype(int, copy=True),
    )


def _full_horizon_path(config_sig: str, seed: int) -> Path:
    return CACHE_DIR / f"{config_sig}_seed{seed}_full_horizon.npz"
