from typing import Optional
import numpy as np
import wandb

from src.utils.constants import N_CORES
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


class GraphSeriesEnvPlaceCore(BaseGraphSeriesEnv):
    def __init__(
        self,
        circuit_config,
        action_type,
        mask_full_cores=True,
        n_cores=N_CORES,
        weights_reward={
            "nonlocal": 1,
            "capacity": 30,
            "intervention": 40,
            "slice_idx": 50,
        },
    ):
        
        super().__init__(
            circuit_config, action_type, mask_full_cores, n_cores, weights_reward
        )

    def _take_action(self, action: int) -> tuple[int, int, bool, bool, bool]:

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
                self.action_type,
                self.core_capacities,
                self.new_allocation,
                self.interactions,
                self.n_qbits,
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
                    qbit * self.n_cores + core if self.qbit_idx is None else core
                    for core in sample(range(self.n_cores), self.n_cores)
                    if not any(get_violations(qbit, core))
                ),
                None,
            )

        intervention = False

        violations = get_violations(self.qbit_idx, action)

        if not any(violations):
            intervention = True
            # print('invalid action', action)
            actual_action: Optional[int] = get_valid_action(self.qbit_idx)
            assert actual_action is not None, "Truncation happened!"

        else:
            actual_action = action

        self.core_capacities[action] -= 1
        self.new_allocation[self.qbit_idx, action] = 1
        #set_new_placement només funcinoa per PlacePair!
        # self._set_new_placement(actual_action)

        
        #Això és per les reserves? 
        
        if has_pair(self.qbit_idx, self.interactions):
            neighbour = np.argmax(self.interactions[self.qbit_idx])
            self.core_capacities[action] -= 1


            self._set_new_placement(
                pack_action(neighbour, action, self.action_type, self.n_cores)
            )

        nl_comm = sum(abs(self.old_allocation - self.new_allocation)) // 2
        self.qbit_idx += 1
        return actual_action, nl_comm, intervention, violations[0], False
    

    def env_mask(self):
        """Mask for the MaskablePPO"""
        mask = (
            np.ones(self.n_qbits * self.n_cores, dtype=bool)
            if self.qbit_idx is None
            else np.ones(self.n_cores, dtype=bool)
        )

        if self.qbit_idx is None:
            for qbit in range(self.n_qbits):  # ban qbits already placed
                if is_qbit_placed(qbit, self.new_allocation):
                    for core in range(self.n_cores):
                        mask[qbit * self.n_cores + core] = False

        if self.mask_full_cores:  # ban full cores
            for core, capacity in enumerate(self.core_capacities):
                if capacity <= 0:
                    for qbit in range(self.n_qbits):
                        if self.qbit_idx is None:
                            mask[qbit * self.n_cores + core] = False
                        else:
                            mask[core] = False

        return mask
