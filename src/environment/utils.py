import numpy as np

def get_nl_com(old_allocation, new_allocation, qbit):
    return np.sum(np.abs(old_allocation[qbit] - new_allocation[qbit])) // 2


def has_pair(qbit, interactions):
    return sum(interactions[qbit]) > 0


def is_qbit_placed(qbit, new_allocation):
    return any(new_allocation[qbit])


def is_qbit_placed_vectorized(new_allocation):
    return np.any(new_allocation, axis=1)


def is_last_qbit(qbit, new_allocation):
    n_qbits = new_allocation.shape[0]
    return all(any(new_allocation[q]) for q in range(n_qbits) if q != qbit)


def unpack_action(action, curr_qbit_idx, action_type, n_cores):
    """action = qbit * n_cores + core"""
    qbit = action // n_cores if action_type == 'L' else curr_qbit_idx
    core = action % n_cores if action_type == 'L' else action
    return qbit, core


def pack_action(qbit, core, action_type, n_cores):
    if action_type == 'L':
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
    core_capacities,
    new_allocation,
    interactions,
    core_reservations=None,
    qbit_reservations=None,
):
    n_qbits = new_allocation.shape[0]
    aux = core_capacities.copy()
    aux[core] -= 1  # suposem que coloquem qbit a core
    full_gate_capacity = (
        np.sum((aux - core_reservations) // 2)
        if core_reservations is not None
        else np.sum(a=np.array(aux) // 2)
    )

    future_qbits_mask = (
        (~is_qbit_placed_vectorized(new_allocation))
        & (np.arange(n_qbits) != qbit)
    )
    if qbit_reservations is not None:
        future_qbits_mask = future_qbits_mask & (qbit_reservations == -1)

    remaining_full_gates = np.sum(interactions[:n_qbits][future_qbits_mask]) // 2

    return full_gate_capacity < remaining_full_gates
