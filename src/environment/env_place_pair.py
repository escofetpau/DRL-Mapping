from typing import Optional
import numpy as np
import wandb

from src.utils.constants import N_CORES, CORE_CAPACITY
from random import sample, seed
from src.environment.utils import (
    pack_action,
    unpack_action,
    has_pair,
    is_qbit_placed,
    is_direct_capacity_violation,
    is_missing_space_for_interaction_violation,
    is_no_space_for_future_gates_violation,
)

from src.environment.base_env import BaseGraphSeriesEnv


seed(42)


class GraphSeriesEnvPlacePair(BaseGraphSeriesEnv):
    def __init__(
        self,
        circuit_config,
        action_type,
        weights_reward,
        mask_full_cores=True,
        n_cores=N_CORES,
    ):

        super().__init__(
            circuit_config,
            action_type,
            weights_reward,
            mask_full_cores=True,
            n_cores=N_CORES,
            core_capacity=CORE_CAPACITY,
        )

    def _take_action(self, action: int) -> tuple[int, bool]:

        def get_violations(qbit, core):
            direct_capacity_violation = is_direct_capacity_violation(
                core, self.core_capacities
            )
            missing_space_for_interaction_violation = (
                is_missing_space_for_interaction_violation(
                    qbit, core, self.core_capacities, self.interactions
                )
            )
            no_space_for_future_gates = is_no_space_for_future_gates_violation(
                qbit,
                core,
                self.core_capacities,
                self.new_allocation,
                self.interactions,
            )

            return (
                direct_capacity_violation,
                missing_space_for_interaction_violation,
                no_space_for_future_gates,
            )

        def get_valid_action(qbit) -> int | None:
            """"""
            return next(
                (
                    qbit * self.n_cores + core if self.action_type == 'L' else core
                    for core in sample(range(self.n_cores), self.n_cores)
                    if not any(get_violations(qbit, core))
                ),
                None,
            )

        intervention = False
        qbit, core = unpack_action(action, self.qbit_idx, self.action_type, self.n_cores)

        violations = get_violations(qbit, core)

        if not any(violations):
            intervention = True
            # print('invalid action', action)
            actual_action: Optional[int] = get_valid_action(qbit)
            assert actual_action is not None, "Truncation happened!"

        else:
            actual_action = action

        qbit, core = unpack_action(actual_action, self.qbit_idx, self.action_type, self.n_cores)
        self.core_capacities[core] -= 1
        self._set_new_placement(qbit, core)

        if has_pair(qbit, self.interactions):
            neighbour = np.argmax(self.interactions[qbit])
            self.core_capacities[core] -= 1
            self._set_new_placement(neighbour, core)

        nl_comm = np.sum(np.abs(self.old_allocation - self.new_allocation)) // 2

        self.nl_com = nl_comm
        self.intervention = intervention
        self.direct_capacity_violation = violations[0]
        self.missing_space_for_interaction_violation = violations[1]
        self.no_space_for_future_gates_violation = violations[2]

        return actual_action, False
    

    def env_mask(self):
        """Mask for the MaskablePPO"""
        mask = (
            np.ones(self.n_qbits * self.n_cores, dtype=bool)
            if self.action_type == 'L'
            else np.ones(self.n_cores, dtype=bool)
        )

        if self.action_type == 'L':
            for qbit in range(self.n_qbits):  # ban qbits already placed
                if is_qbit_placed(qbit, self.new_allocation):
                    for core in range(self.n_cores):
                        mask[qbit * self.n_cores + core] = False

        if self.mask_full_cores:  # ban full cores
            for core, capacity in enumerate(self.core_capacities):
                if capacity <= 0:
                    for qbit in range(self.n_qbits):
                        if self.action_type == 'L':
                            mask[qbit * self.n_cores + core] = False
                        else:
                            mask[core] = False

        return mask
