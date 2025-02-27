import numpy as np
from src.utils.constants import ACTION_TYPES


def has_pair(qbit, interactions):
    return sum(interactions[qbit]) > 0


def is_qbit_placed(qbit, new_allocation):
    return any(new_allocation[qbit])


def is_qbit_placed_vectorized(new_allocation):
    return np.any(new_allocation, axis=1)


def is_last_qbit(qbit, new_allocation, n_qbits):
    return all(any(new_allocation[q]) for q in range(n_qbits) if q != qbit)


def unpack_action(action, curr_qbit_idx, n_cores):
    """action = qbit * n_cores + core"""
    qbit = action // n_cores if curr_qbit_idx is None else curr_qbit_idx
    core = action % n_cores if curr_qbit_idx is None else action
    return qbit, core


def pack_action(qbit, core, action_type, n_cores):
    if action_type == ACTION_TYPES[1]:
        return qbit * n_cores + core
    return core


### Violations
def is_direct_capacity_violation(core, core_capacities):
    return core_capacities[core] == 0


def is_interaction_violation(qbit, core, qbit_reservations):
    return qbit_reservations[qbit] != -1 and qbit_reservations[qbit] != core


def is_missing_space_for_interaction_violation(qbit, core, core_capacities, interactions):
    return has_pair(qbit, interactions) and core_capacities[core] < 2


def is_direct_reservation_capacity_violation(core, core_capacities, core_reservations):
    return (
        core_reservations[core] > 0
        and core_capacities[core] - core_reservations[core] < 1
    )


def is_interaction_reservation_capacity_violation(
    qbit, core, interactions, core_capacities, core_reservations
):
    return (
        core_reservations[core] > 0
        and core_capacities[core] - core_reservations[core] - has_pair(qbit, interactions) < 1
    )


def is_no_space_for_future_gates_violation(
    qbit,
    core,
    action_type,
    core_capacities,
    new_allocation,
    interactions,
    n_qbits,
    core_reservations=None,
    qbit_reservations=None,
):
    aux = core_capacities.copy()
    aux[core] -= 1  # suposem que coloquem qbit a core

    full_gate_capacity = (
        np.sum((aux - core_reservations) // 2)
        if core_reservations is not None
        else np.sum(aux) // 2
    )

    if action_type == ACTION_TYPES[0]:  # sergi
        future_qbits_mask = np.arange(n_qbits) > qbit
        if qbit_reservations is not None:
            future_qbits_mask = future_qbits_mask & (qbit_reservations == -1)
    else:  # laia
        future_qbits_mask = (
            (~is_qbit_placed_vectorized(new_allocation))
            & (np.arange(n_qbits) != qbit)
        )
        if qbit_reservations is not None:
            future_qbits_mask = future_qbits_mask & (qbit_reservations == -1)

    remaining_full_gates = np.sum(interactions[:n_qbits][future_qbits_mask]) // 2


    return full_gate_capacity < remaining_full_gates
