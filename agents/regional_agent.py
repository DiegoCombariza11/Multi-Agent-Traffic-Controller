from typing import Dict, List

import numpy as np


class RegionalAgent:
    """Regional coordination layer that can override RL actions when congestion spikes."""

    def __init__(
        self,
        region_id: str,
        intersections: List[str],
        queue_threshold: int = 30,
        min_intervention_steps: int = 10,
        override_phase: int = 0,
    ) -> None:
        self.region_id = region_id
        self.intersections = intersections
        self.queue_threshold = queue_threshold
        self.min_intervention_steps = min_intervention_steps
        self.override_phase = override_phase

        self.intervening = False
        self.remaining_steps = 0

    def get_regional_queue(self, info: Dict[str, float]) -> float:
        """Estimate congestion summing stopped vehicles per intersection."""

        if not info:
            return 0

        return sum(info.get(f"{tl}_stopped", 0) for tl in self.intersections)

    def should_intervene(self, regional_queue: float) -> bool:
        return regional_queue >= self.queue_threshold

    def apply_regional_action(self, actions: np.ndarray, tl_index_map: Dict[str, int]) -> None:
        """Override the RL actions for the intersections this region controls."""

        if actions is None:
            return

        for tl in self.intersections:
            idx = tl_index_map.get(tl)
            if idx is None:
                continue

            if actions.ndim == 2:
                actions[:, idx] = self.override_phase
            else:
                actions[idx] = self.override_phase

    def step(self, info: Dict[str, float], actions: np.ndarray, tl_index_map: Dict[str, int]) -> bool:
        """Update intervention state and mutate the action vector when needed."""

        regional_queue = self.get_regional_queue(info)

        if self.intervening:
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self.intervening = False
            else:
                self.apply_regional_action(actions, tl_index_map)
                return True

        if self.should_intervene(regional_queue):
            self.intervening = True
            self.remaining_steps = self.min_intervention_steps
            self.apply_regional_action(actions, tl_index_map)
            return True

        return False
