from typing import Dict, List

import numpy as np


class RegionalAgent:
    """Regional coordination layer that can override RL actions when congestion spikes."""

    def __init__(
        self,
        region_id: str,
        intersections: List[str],
        queue_threshold: int = 10,
        min_intervention_steps: int = 20,
        override_phase: int = 0,
        release_threshold: int | None = None,
    ) -> None:
        self.region_id = region_id
        self.intersections = intersections
        self.queue_threshold = queue_threshold
        self.min_intervention_steps = min_intervention_steps
        self.override_phase = override_phase
        # Hysteresis: while intervening we keep intervention until queue drops below release_threshold
        if release_threshold is None:
            release_threshold = max(0, queue_threshold - 1)
        self.release_threshold = release_threshold

        self.intervening = False
        self.remaining_steps = 0

    def get_regional_queue(self, info: Dict[str, float]) -> float:
        """Estimate congestion summing accumulated waiting time per intersection."""

        if not info:
            return 0
        total = 0
        missing_keys = False
        for tl in self.intersections:
            k = f"{tl}_accumulated_waiting_time"
            if k in info:
                try:
                    total += float(info.get(k, 0) or 0)
                except Exception:
                    total += 0
            else:
                missing_keys = True
        # Fallback: if specific per-TL keys are missing, use system aggregate
        if total == 0 and missing_keys:
            total = float(info.get("system_total_waiting_time", 0) or 0)
        return total

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
        print(f"[DEBUG] RegionalAgent '{self.region_id}': regional_queue={regional_queue}, intervening={self.intervening}, remaining_steps={self.remaining_steps}")

        if self.intervening:
            # If queue still above release threshold, maintain intervention without decrementing
            if regional_queue >= self.release_threshold:
                self.apply_regional_action(actions, tl_index_map)
                return True
            # Otherwise decrement remaining steps
            self.remaining_steps -= 1
            if self.remaining_steps > 0:
                self.apply_regional_action(actions, tl_index_map)
                return True
            else:
                self.intervening = False

        if self.should_intervene(regional_queue):
            self.intervening = True
            self.remaining_steps = self.min_intervention_steps
            self.apply_regional_action(actions, tl_index_map)
            print(f"[DEBUG] RegionalAgent '{self.region_id}': *** STARTED intervention (queue={regional_queue} >= {self.queue_threshold})")
            return True

        return False
