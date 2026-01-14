from __future__ import annotations

from typing import Dict, Iterator, List

from generic.experiments.pipeline import PipelineSpec


class PipelineRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, PipelineSpec] = {}

    def register(self, spec: PipelineSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Pipeline '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> PipelineSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs))
            raise KeyError(
                f"Unknown pipeline '{name}'. Known pipelines: {known}"
            ) from exc

    def list(self) -> List[PipelineSpec]:
        return list(self._specs.values())

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._specs

    def __getitem__(self, name: str) -> PipelineSpec:
        return self.get(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)
