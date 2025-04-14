from typing import Optional
import numpy as np

from random import sample, seed

from src.environment.base_env import BaseGraphSeriesEnv
from src.environment.utils import (
    get_nl_com,
    unpack_action,
    has_pair,
    is_qubit_placed,
    is_direct_capacity_violation,
    is_missing_space_for_interaction_violation,
    is_no_space_for_future_gates_violation,
)


seed(42)


class GraphSeriesEnvPlacePair(BaseGraphSeriesEnv):
    def __init__(
        self,
        circuit_config: dict,
        action_type: str,
        n_qubits: int,
        n_cores: int,
        weights_reward: dict,
        mask_full_cores: bool = True,
    ):
        '''
            - mask_full_cores: If True, masks full cores. Avoids direct_capacity_violatio. If False, does not mask cores.
        '''

        self.mask_full_cores = mask_full_cores

        super().__init__(
            circuit_config,
            action_type,
            weights_reward,
            n_qubits,
            n_cores,
            core_capacity=n_qubits//n_cores,
        )

    def _take_action(self, action: int) -> tuple[int, bool]:

        def get_violations(qubit: int, core: int) -> tuple[bool, bool, bool]:
            direct_capacity_violation = is_direct_capacity_violation(
                core, self.core_capacities
            )
            missing_space_for_interaction_violation = (
                is_missing_space_for_interaction_violation(
                    qubit, core, self.core_capacities, self.interactions
                )
            )
            no_space_for_future_gates = is_no_space_for_future_gates_violation(
                qubit,
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

        def get_valid_action(qubit: int) -> int | None:
            """"""
            return next(
                (
                    qubit * self.n_cores + core if self.action_type == 'L' else core
                    for core in sample(range(self.n_cores), self.n_cores)
                    if not any(get_violations(qubit, core))
                ),
                None,
            )

        intervention = False
        qubit, core = unpack_action(action, self.qubit_idx, self.action_type, self.n_cores)

        violations = get_violations(qubit, core)

        if any(violations):
            intervention = True
            actual_action: Optional[int] = get_valid_action(qubit)
            assert actual_action is not None, "Truncation happened!"

        else:
            actual_action = action

        qubit, core = unpack_action(actual_action, self.qubit_idx, self.action_type, self.n_cores)
        self.core_capacities[core] -= 1
        self._set_new_placement(qubit, core)

        
        self.nl_com = get_nl_com(self.old_allocation, self.new_allocation, qubit)
        self.intervention = intervention
        self.direct_capacity_violation = violations[0]
        if self.direct_capacity_violation:
            print(f"Direct capacity violation for qubit {qubit} and core {core}")
        self.missing_space_for_interaction_violation = violations[1]
        self.no_space_for_future_gates_violation = violations[2]

        if has_pair(qubit, self.interactions):
            neighbour = np.argmax(self.interactions[qubit])
            self.core_capacities[core] -= 1
            self._set_new_placement(neighbour, core)
            self.nl_com += get_nl_com(self.old_allocation, self.new_allocation, neighbour)


        return actual_action, False
    

    def env_mask(self) -> np.ndarray:
        """Mask for the MaskablePPO"""
        mask = (
            np.ones(self.n_qubits * self.n_cores, dtype=bool)
            if self.action_type == 'L'
            else np.ones(self.n_cores, dtype=bool)
        )

        if self.action_type == 'L':
            for qubit in range(self.n_qubits):  # ban qubits already placed
                if is_qubit_placed(qubit, self.new_allocation):
                    for core in range(self.n_cores):
                        mask[qubit * self.n_cores + core] = False

        if self.mask_full_cores:  # ban full cores
            for core, capacity in enumerate(self.core_capacities):
                if capacity == 0:
                    if self.action_type == 'L':
                        for qubit in range(self.n_qubits):
                            mask[qubit * self.n_cores + core] = False
                    else:
                        mask[core] = False

        return mask
