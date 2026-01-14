from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from generic.config import Config

OfflineFactory = Callable[[Config], object]
OnlineFactory = Callable[[Config], object]


@dataclass(frozen=True)
class PipelineSpec:
    """
    Lightweight description of an offline+online experiment.

    Attributes
    ----------
    name:
        Display name for logs / JSON summary.
    offline_label:
        Label stored in the JSON for the offline component.
    online_label:
        Label stored in the JSON for the online component.
    offline_factory:
        Callable that returns a solver/heuristic with a ``solve(instance)`` method.
    online_factory:
        Callable that returns a policy implementing ``select_bin(...)``.
    """

    name: str
    offline_label: str
    online_label: str
    offline_factory: OfflineFactory
    online_factory: OnlineFactory
    offline_cache_key: str | None = None
